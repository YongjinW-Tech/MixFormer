from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, out_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.w_gate = nn.Linear(dim, hidden_dim)
        self.w_value = nn.Linear(dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_value(x)
        x = self.dropout(x)
        return self.w_out(x)


class HeadMixing(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, head_dim = x.shape
        if head_dim % num_heads != 0:
            raise ValueError(
                f"head_dim must be divisible by num_heads for HeadMixing, got head_dim={head_dim}, num_heads={num_heads}"
            )
        # 对应论文 3.3.1 中的 HeadMixing：
        # X' ∈ R^{N×D} -> R^{N×N×D/N} -> 交换前两个维度 -> 再展平回 R^{N×D}。
        # 这里保留 batch 维，因此实际张量是 B×N×D。
        chunk_dim = head_dim // num_heads
        x = x.reshape(batch_size, num_heads, num_heads, chunk_dim)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, num_heads, head_dim)


def masked_average(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = embeddings * mask.unsqueeze(-1)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return weighted.sum(dim=-2) / denom


class QueryMixer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_multiplier: float,
        dropout: float = 0.0,
        enable_ui_decoupling: bool = False,
        num_user_heads: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.enable_ui_decoupling = enable_ui_decoupling
        self.num_user_heads = num_user_heads
        self.norm_in = RMSNorm(head_dim)
        self.norm_head = RMSNorm(head_dim)
        self.head_mixing = HeadMixing()
        hidden_dim = int(head_dim * ffn_multiplier)
        self.per_head_ffn = nn.ModuleList(
            [SwiGLUFeedForward(head_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_heads)]
        )

        if enable_ui_decoupling:
            if not (0 < num_user_heads < num_heads):
                raise ValueError("num_user_heads must be in (0, num_heads) when UI decoupling is enabled.")
            if head_dim % num_heads != 0:
                raise ValueError("head_dim must be divisible by num_heads when UI decoupling is enabled.")
            mask = torch.ones(num_heads, head_dim)
            chunk_dim = head_dim // num_heads
            mask[:num_user_heads, num_user_heads * chunk_dim :] = 0.0
            self.register_buffer("ui_mask", mask, persistent=False)
        else:
            self.register_buffer("ui_mask", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 论文公式 (3): P = HeadMixing(Norm(X)) + X
        mixed = self.head_mixing(self.norm_in(x))
        if self.enable_ui_decoupling and self.ui_mask is not None:
            # 对应论文 3.4 的 user-item decoupling 思想：
            # 对 user-side heads 屏蔽来自 item-side heads 的混合信号。
            mixed = mixed * self.ui_mask.unsqueeze(0)
        p = mixed + x

        outputs = []
        for head_idx in range(self.num_heads):
            head = p[:, head_idx, :]
            # 论文公式 (4): q_i = SwiGLUFFN_i(Norm(p_i)) + p_i
            # 这里每个 head 单独一套 FFN，用来保留异构特征头的专门化能力。
            head = self.per_head_ffn[head_idx](self.norm_head(head)) + head
            outputs.append(head)
        return torch.stack(outputs, dim=1)


class CrossAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, model_dim: int, ffn_multiplier: float, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.seq_norm = RMSNorm(model_dim)
        self.seq_ffn = SwiGLUFeedForward(
            model_dim,
            hidden_dim=int(model_dim * ffn_multiplier),
            out_dim=model_dim,
            dropout=dropout,
        )
        self.key_projs = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(num_heads)])
        self.value_projs = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, seq_repr: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 论文公式 (5): h_t = SwiGLUFFN^(l)(Norm(s_t)) + s_t
        # 每一层都对行为序列做一次独立 FFN 变换，再进入 K/V 投影。
        seq_hidden = self.seq_ffn(self.seq_norm(seq_repr)) + seq_repr
        batch_size, seq_len, _ = seq_hidden.shape
        seq_heads = seq_hidden.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        keys = []
        values = []
        for head_idx in range(self.num_heads):
            head_hidden = seq_heads[:, :, head_idx, :]
            # 论文公式 (6)(7): 先切出第 i 个 head 的 h_t^i，再分别映射到 k_t^i 和 v_t^i。
            keys.append(self.key_projs[head_idx](head_hidden))
            values.append(self.value_projs[head_idx](head_hidden))
        keys = torch.stack(keys, dim=2)
        values = torch.stack(values, dim=2)

        # 论文公式 (8):
        # z_i = sum_t softmax(q_i^T k_t^i / sqrt(D)) v_t^i + q_i
        scores = torch.einsum("bnd,btnd->bnt", q, keys) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~seq_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        z = torch.einsum("bnt,btnd->bnd", attn, values) + q
        return z, seq_hidden


class OutputFusion(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, ffn_multiplier: float, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm = RMSNorm(head_dim)
        hidden_dim = int(head_dim * ffn_multiplier)
        self.per_head_ffn = nn.ModuleList(
            [SwiGLUFeedForward(head_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_heads)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs = []
        for head_idx in range(self.num_heads):
            head = z[:, head_idx, :]
            # 论文公式 (9): o_i = SwiGLUFFN_i(Norm(z_i)) + z_i
            # 每个 head 独立融合“高阶非序列语义 + 条件化序列聚合结果”。
            head = self.per_head_ffn[head_idx](self.norm(head)) + head
            outputs.append(head)
        return torch.stack(outputs, dim=1)


class MixFormerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        model_dim: int,
        ffn_multiplier: float,
        dropout: float = 0.0,
        enable_ui_decoupling: bool = False,
        num_user_heads: int = 0,
    ):
        super().__init__()
        self.query_mixer = QueryMixer(
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            enable_ui_decoupling=enable_ui_decoupling,
            num_user_heads=num_user_heads,
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            model_dim=model_dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.output_fusion = OutputFusion(
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )

    def forward(self, x_heads: torch.Tensor, seq_repr: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 单个 MixFormer block 的执行顺序与论文 3.3 完全一致：
        # Query Mixer -> Cross Attention -> Output Fusion
        q = self.query_mixer(x_heads)
        z, seq_hidden = self.cross_attention(q, seq_repr, seq_mask)
        o = self.output_fusion(z)
        return o, seq_hidden
