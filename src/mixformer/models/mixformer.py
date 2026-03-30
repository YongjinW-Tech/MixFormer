from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .layers import MixFormerBlock, RMSNorm, masked_average


class NonSequentialHeadEncoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        embedding_dims: Dict[str, int],
        enable_ui_decoupling: bool = False,
        num_user_heads: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embedding_dims = embedding_dims
        self.enable_ui_decoupling = enable_ui_decoupling
        self.num_user_heads = num_user_heads
        self.num_item_heads = num_heads - num_user_heads

        self.user_field_order = ["user_id", "gender", "age", "occupation"]
        self.item_field_order = ["item_id", "genre", "hour", "weekday"]

        self.total_non_seq_dim = sum(embedding_dims[field] for field in self.user_field_order + self.item_field_order)

        if enable_ui_decoupling:
            self.user_dim = sum(embedding_dims[field] for field in self.user_field_order)
            self.item_dim = sum(embedding_dims[field] for field in self.item_field_order)
            if self.user_dim % self.num_user_heads != 0:
                raise ValueError("User-side non-sequential dimension must be divisible by num_user_heads.")
            if self.item_dim % self.num_item_heads != 0:
                raise ValueError("Item-side non-sequential dimension must be divisible by num_item_heads.")
            user_slice_dim = self.user_dim // self.num_user_heads
            item_slice_dim = self.item_dim // self.num_item_heads
            self.user_projs = nn.ModuleList([nn.Linear(user_slice_dim, head_dim) for _ in range(self.num_user_heads)])
            self.item_projs = nn.ModuleList([nn.Linear(item_slice_dim, head_dim) for _ in range(self.num_item_heads)])
        else:
            if self.total_non_seq_dim % num_heads != 0:
                raise ValueError("Total non-sequential embedding dimension must be divisible by num_heads.")
            slice_dim = self.total_non_seq_dim // num_heads
            self.shared_projs = nn.ModuleList([nn.Linear(slice_dim, head_dim) for _ in range(num_heads)])

    def _encode_split(self, concat: torch.Tensor, num_splits: int, projections: nn.ModuleList) -> torch.Tensor:
        slice_dim = concat.shape[-1] // num_splits
        heads = []
        for idx in range(num_splits):
            start = idx * slice_dim
            end = (idx + 1) * slice_dim
            # 对应论文公式 (2)：将拼接后的非序列特征按连续区间切分，再逐 head 投影到 D 维。
            heads.append(projections[idx](concat[:, start:end]))
        return torch.stack(heads, dim=1)

    def forward(self, feature_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.enable_ui_decoupling:
            # 对应论文 3.4.1：先显式区分 user-side 和 item-side 非序列特征，
            # 再分别投影到 N_U / N_G 个 heads。
            user_concat = torch.cat([feature_embeddings[field] for field in self.user_field_order], dim=-1)
            item_concat = torch.cat([feature_embeddings[field] for field in self.item_field_order], dim=-1)
            user_heads = self._encode_split(user_concat, self.num_user_heads, self.user_projs)
            item_heads = self._encode_split(item_concat, self.num_item_heads, self.item_projs)
            return torch.cat([user_heads, item_heads], dim=1)

        # 对应论文 3.2：标准 MixFormer 将全部非序列特征拼接后统一 split 成 N 个 heads。
        concat = torch.cat([feature_embeddings[field] for field in self.user_field_order + self.item_field_order], dim=-1)
        return self._encode_split(concat, self.num_heads, self.shared_projs)


class MixFormerModel(nn.Module):
    def __init__(self, meta: Dict, model_cfg: Dict):
        super().__init__()
        self.meta = meta
        self.model_cfg = model_cfg
        self.num_heads = model_cfg["num_heads"]
        self.head_dim = model_cfg["head_dim"]
        self.model_dim = self.num_heads * self.head_dim
        self.max_seq_len = model_cfg["max_seq_len"]
        self.dropout = nn.Dropout(model_cfg["dropout"])

        non_seq_dims = model_cfg["non_seq_embedding_dims"]
        seq_dims = model_cfg["seq_embedding_dims"]

        self.user_id_emb = nn.Embedding(meta["num_users"] + 1, non_seq_dims["user_id"], padding_idx=0)
        self.gender_emb = nn.Embedding(meta["num_genders"] + 1, non_seq_dims["gender"], padding_idx=0)
        self.age_emb = nn.Embedding(meta["num_ages"] + 1, non_seq_dims["age"], padding_idx=0)
        self.occupation_emb = nn.Embedding(meta["num_occupations"] + 1, non_seq_dims["occupation"], padding_idx=0)
        self.item_id_emb = nn.Embedding(meta["num_items"] + 1, non_seq_dims["item_id"], padding_idx=0)
        self.genre_emb = nn.Embedding(meta["num_genres"] + 1, non_seq_dims["genre"], padding_idx=0)
        self.hour_emb = nn.Embedding(24 + 1, non_seq_dims["hour"], padding_idx=0)
        self.weekday_emb = nn.Embedding(7 + 1, non_seq_dims["weekday"], padding_idx=0)

        self.rating_emb = nn.Embedding(meta["num_ratings"] + 1, seq_dims["rating"], padding_idx=0)
        self.position_emb = nn.Embedding(model_cfg["max_seq_len"] + 1, seq_dims["position"], padding_idx=0)

        seq_input_dim = (
            non_seq_dims["item_id"]
            + seq_dims["rating"]
            + non_seq_dims["hour"]
            + non_seq_dims["weekday"]
            + non_seq_dims["genre"]
            + seq_dims["position"]
        )
        self.seq_input_proj = nn.Linear(seq_input_dim, self.model_dim)

        self.non_seq_encoder = NonSequentialHeadEncoder(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            embedding_dims=non_seq_dims,
            enable_ui_decoupling=model_cfg.get("enable_ui_decoupling", False),
            num_user_heads=model_cfg.get("num_user_heads", self.num_heads // 2),
        )

        self.blocks = nn.ModuleList(
            [
                MixFormerBlock(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    model_dim=self.model_dim,
                    ffn_multiplier=model_cfg["ffn_multiplier"],
                    dropout=model_cfg["dropout"],
                    enable_ui_decoupling=model_cfg.get("enable_ui_decoupling", False),
                    num_user_heads=model_cfg.get("num_user_heads", self.num_heads // 2),
                )
                for _ in range(model_cfg["num_layers"])
            ]
        )
        self.final_norm = RMSNorm(self.model_dim)
        self.task_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.SiLU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(self.model_dim // 2, 1),
        )

    def _embed_multi_hot_genres(self, genre_ids: torch.Tensor, genre_mask: torch.Tensor) -> torch.Tensor:
        genre_emb = self.genre_emb(genre_ids)
        return masked_average(genre_emb, genre_mask)

    def _build_non_seq_heads(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 这里对应论文 3.2 的 non-sequential feature embedding：
        # user / item / context 特征先分别查表，再进入 feature embedding and split layer。
        feature_embeddings = {
            "user_id": self.user_id_emb(batch["user_id"]),
            "gender": self.gender_emb(batch["gender"]),
            "age": self.age_emb(batch["age"]),
            "occupation": self.occupation_emb(batch["occupation"]),
            "item_id": self.item_id_emb(batch["target_item_id"]),
            "genre": self._embed_multi_hot_genres(batch["target_genres"], batch["target_genre_mask"]),
            "hour": self.hour_emb(batch["target_hour"] + 1),
            "weekday": self.weekday_emb(batch["target_weekday"] + 1),
        }
        return self.non_seq_encoder(feature_embeddings)

    def _build_sequence_repr(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 这里对应论文 3.2 的 sequential feature embedding：
        # 每个历史 action 由 item / rating / time / genre / position 等信息组成，
        # 拼接后投影到 N*D 维，作为 block 内 Cross Attention 的序列输入。
        hist_item = self.item_id_emb(batch["hist_item_ids"])
        hist_rating = self.rating_emb(batch["hist_ratings"])
        hist_hour = self.hour_emb(batch["hist_hours"] + 1)
        hist_weekday = self.weekday_emb(batch["hist_weekdays"] + 1)
        hist_genre = self._embed_multi_hot_genres(batch["hist_genres"], batch["hist_genre_mask"])
        hist_position = self.position_emb(batch["hist_positions"])
        seq_concat = torch.cat(
            [hist_item, hist_rating, hist_hour, hist_weekday, hist_genre, hist_position],
            dim=-1,
        )
        return self.seq_input_proj(self.dropout(seq_concat))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 先构建非序列 query heads，再构建行为序列表示。
        x_heads = self._build_non_seq_heads(batch)
        seq_repr = self._build_sequence_repr(batch)
        seq_mask = batch["hist_mask"]

        # 多层堆叠，体现论文里“统一 backbone 中逐层进行高阶特征交互 + 条件化序列聚合”的思想。
        for block in self.blocks:
            x_heads, seq_repr = block(x_heads, seq_repr, seq_mask)

        # 论文中这里接 task-specific heads；
        # 公开数据复现版本采用单任务二分类打分头。
        fused = x_heads.reshape(x_heads.size(0), -1)
        fused = self.final_norm(fused)
        logits = self.task_head(fused).squeeze(-1)
        return logits
