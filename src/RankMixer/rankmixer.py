import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenization(nn.Module):
    """
    Semantic Tokenization 语义分组切分 + 投影
        1. 将输入特征按语义分组后拼接成一个总向量 e_input
        2. 将 e_input 切分为 T 份
        3. 每一份通过独立的线性投影映射到 D 维
        得到 T 个 feature tokens (公式 2)
    输入:
        x: shape = (B, input_dim)
    输出:
        tokens: shape = (B, T, D)
    """
    def __init__(self, input_dim: int, num_tokens: int, token_dim: int):
        super().__init__()
        if input_dim % num_tokens != 0:
            raise ValueError(
                f"input_dim={input_dim} 不能被 num_tokens={num_tokens} 整除"
            )

        self.input_dim = input_dim
        self.num_tokens = num_tokens   # T
        self.token_dim = token_dim     # D
        self.split_dim = input_dim // num_tokens

        self.proj_layers = nn.ModuleList([
            nn.Linear(self.split_dim, token_dim) for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        splits = torch.split(x, self.split_dim, dim=-1)

        tokens = []
        for i in range(self.num_tokens):
            token_i = self.proj_layers[i](splits[i])   # (B, D)
            tokens.append(token_i)

        # (B, T, D)
        tokens = torch.stack(tokens, dim=1)
        return tokens


class MultiHeadTokenMixing(nn.Module):
    """
    多头 Token Mixing

    论文流程：
    1. 每个 token x_t ∈ R^D 被均匀切成 H 个 head
    2. 对于第 h 个 head，把所有 token 的第 h 个子向量拼接起来
    3. 得到新的 token s_h
    4. 论文中设置 H = T，从而输出 token 数与输入 token 数一致，
       便于残差连接。对应论文公式 (3)(4)(5)。 [oai_citation:4‡Zhu 等 - 2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf](sediment://file_000000005c0871f784ad090e30387f30)

    输入:
        x: (B, T, D)

    输出:
        mixed: (B, T, D)
    """
    def __init__(self, num_tokens: int, token_dim: int, num_heads: int):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={token_dim} 必须能被 num_heads={num_heads} 整除。"
            )
        if num_heads != num_tokens:
            raise ValueError(
                "严格按 RankMixer 论文主干实现时，需要满足 num_heads == num_tokens。"
            )

        self.num_tokens = num_tokens   # T
        self.token_dim = token_dim     # D
        self.num_heads = num_heads     # H
        self.head_dim = token_dim // num_heads  # D / H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape

        # Step 1: (B, T, D) -> (B, T, H, D/H)
        x = x.view(B, T, self.num_heads, self.head_dim)

        # Step 2: (B, T, H, D/H) -> (B, H, T, D/H)
        x = x.permute(0, 2, 1, 3).contiguous()

        # Step 3: 对每个 head，把所有 token 的该 head 子向量拼接
        # (B, H, T, D/H) -> (B, H, T * D/H)
        mixed = x.view(B, self.num_heads, T * self.head_dim)

        # 由于 H = T，且 T * (D/H) = D
        # 所以输出为 (B, T, D)
        return mixed


class PerTokenFFN(nn.Module):
    """
    每个 token 独立参数的前馈网络（Per-token FFN）

    论文强调：
    - 不同 token 有各自独立的 FFN 参数
    - 这不同于 Transformer 里所有 token 共享一个 FFN
    对应论文公式 (6)(7)(8)(9)。 [oai_citation:5‡Zhu 等 - 2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf](sediment://file_000000005c0871f784ad090e30387f30)

    输入:
        x: (B, T, D)

    输出:
        y: (B, T, D)
    """
    def __init__(self, num_tokens: int, token_dim: int, expansion_ratio: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden_dim = token_dim * expansion_ratio

        self.fc1_layers = nn.ModuleList([
            nn.Linear(token_dim, self.hidden_dim) for _ in range(num_tokens)
        ])
        self.fc2_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, token_dim) for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_tokens):
            # 取第 i 个 token: (B, D)
            token_i = x[:, i, :]

            # 独立 FFN
            h = self.fc1_layers[i](token_i)
            h = F.gelu(h)
            h = self.fc2_layers[i](h)

            outputs.append(h)

        # (B, T, D)
        y = torch.stack(outputs, dim=1)
        return y


class RankMixerBlock(nn.Module):
    """
    单个 RankMixer Block

    对应论文公式 (1):  [oai_citation:6‡Zhu 等 - 2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf](sediment://file_000000005c0871f784ad090e30387f30)
        S_{n-1} = LN(TokenMixing(X_{n-1}) + X_{n-1})
        X_n     = LN(PFFN(S_{n-1}) + S_{n-1})
    """
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        expansion_ratio: int = 4
    ):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_heads=num_heads
        )
        self.per_token_ffn = PerTokenFFN(
            num_tokens=num_tokens,
            token_dim=token_dim,
            expansion_ratio=expansion_ratio
        )

        self.norm1 = nn.LayerNorm(token_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(token_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token Mixing + Residual + Norm
        mixed = self.token_mixing(x)   # (B, T, D)
        x = self.norm1(x + mixed)

        # Per-token FFN + Residual + Norm
        ffn_out = self.per_token_ffn(x)  # (B, T, D)
        x = self.norm2(x + ffn_out)

        return x


class RankMixer(nn.Module):
    """
    RankMixer 主干（Dense 版）

    整体流程：
        input vector
          -> Semantic Tokenization
          -> L 个 RankMixer Block
          -> mean pooling
          -> output representation

    对应论文 3.1 总体架构。 [oai_citation:7‡Zhu 等 - 2025 - RankMixer Scaling Up Ranking Models in Industrial Recommenders.pdf](sediment://file_000000005c0871f784ad090e30387f30)
    """
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        expansion_ratio: int = 4
    ):
        super().__init__()

        self.semantic_tokenization = SemanticTokenization(
            input_dim=input_dim,
            num_tokens=num_tokens,
            token_dim=token_dim
        )

        self.blocks = nn.ModuleList([
            RankMixerBlock(
                num_tokens=num_tokens,
                token_dim=token_dim,
                num_heads=num_heads,
                expansion_ratio=expansion_ratio
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, input_dim) -> (B, T, D)
        x = self.semantic_tokenization(x)

        # 经过 L 层 RankMixer Block
        for block in self.blocks:
            x = block(x)

        # mean pooling: (B, T, D) -> (B, D)
        x = x.mean(dim=1)
        return x


if __name__ == "__main__":
    # 示例
    batch_size = 4
    input_dim = 512

    model = RankMixer(
        input_dim=input_dim,
        num_layers=2,
        num_tokens=8,
        token_dim=512,
        num_heads=8,          # 严格按论文，要求 num_heads == num_tokens
        expansion_ratio=4
    )

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print("输入形状:", x.shape)   # (4, 512)
    print("输出形状:", y.shape)   # (4, 512)
    print(model)