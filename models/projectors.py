"""
models/projectors.py — 多模態投影層（Pointwise v2 Plan B + 消融實驗支援）

Plan B 主要改動（相比原 Pointwise v2）：
  - prefix 順序改為：[VIDEO, LTP, TEXT_CLIP, MUSIC]（causal mask 正確版）
  - modality_type_emb ID 對應更新

消融實驗擴充（v3）：
  - forward() 新增 active_modalities 參數
  - 只投影並拼接 active 模態的 embedding
  - prefix 長度從固定 4 變為動態（3 或 4）
  - modality_type_emb 也動態選取對應 ID

模態順序與 ID 對應（固定，不隨消融改變）：
  順序：video → ltp → text → music（CANONICAL_ORDER）
  ID：  VIDEO=0, LTP=2, TEXT_CLIP=3, MUSIC=1

消融後的 prefix 範例（維持相對順序）：
  w/o ltp：  [VIDEO, TEXT_CLIP, MUSIC] → IDs [0, 3, 1]
  w/o video：[LTP, TEXT_CLIP, MUSIC]   → IDs [2, 3, 1]
  w/o text： [VIDEO, LTP, MUSIC]       → IDs [0, 2, 1]
  w/o music：[VIDEO, LTP, TEXT_CLIP]   → IDs [0, 2, 3]

投影器結構（LLaVA v1.5 消融實驗驗證的兩層 MLP）：
  LayerNorm → Linear → GELU → Dropout → Linear → LayerNorm
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ── 模態固定順序與 ID 映射（全域常數）────────────────────────────────────────
CANONICAL_ORDER = ["video", "ltp", "text", "music"]
MODALITY_ID = {"video": 0, "ltp": 2, "text": 3, "music": 1}


class ModalityProjector(nn.Module):
    """通用模態投影器：in_dim → llama_hidden_dim（4096）"""

    def __init__(self, in_dim: int, out_dim: int = 4096,
                 hidden_dim: int = 2048, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalProjectors(nn.Module):
    """
    四種模態的獨立投影器集合（Pointwise v2 Plan B + 消融實驗支援）。

    ★ 所有 4 個投影器永遠初始化，消融時只是不呼叫對應投影器。
      原因：保持所有 exp 的 checkpoint（projectors.pt）key 完全一致，
           方便比較與 reload。

    Plan B prefix 順序：[VIDEO:0][LTP:1][TEXT_CLIP:2][MUSIC:3]
      MUSIC 在 pos3，causal mask 下可 attend VIDEO + LTP + TEXT_CLIP
    """

    def __init__(
        self,
        video_dim: int = 768,
        music_dim: int = 768,
        text_dim: int = 512,
        ltp_dim: int = 256,
        llama_hidden_dim: int = 4096,
        projector_hidden_dim: int = 2048,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.video_proj = ModalityProjector(video_dim, llama_hidden_dim, projector_hidden_dim, dropout)
        self.music_proj = ModalityProjector(music_dim, llama_hidden_dim, projector_hidden_dim, dropout)
        self.ltp_proj   = ModalityProjector(ltp_dim,  llama_hidden_dim, projector_hidden_dim, dropout)
        self.text_proj  = ModalityProjector(text_dim,  llama_hidden_dim, projector_hidden_dim, dropout)

        # ★ Embedding 大小固定為 4，即使消融時只用到 3 個
        # 保持所有 exp 的 projectors.pt 的 modality_type_emb 維度一致
        self.modality_type_emb = nn.Embedding(4, llama_hidden_dim)
        nn.init.trunc_normal_(self.modality_type_emb.weight, std=0.02)

    def forward(
        self,
        video_feat: torch.Tensor,        # (B, 768)
        music_candidates: torch.Tensor,  # (B, 1, 768)
        ltp_feat: torch.Tensor,          # (B, 256)
        text_feat: torch.Tensor,         # (B, 512)
        active_modalities: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Args:
            active_modalities: 要注入的模態列表。
                               None = 全部 4 種（向後相容，等同主實驗）。
                               傳入 config.active_modalities 即可。

        Returns:
            prefix: (B, n_active, 4096)
                    n_active = len(active_modalities)
                    完整實驗=4，消融實驗=3
        """
        if active_modalities is None:
            active_modalities = CANONICAL_ORDER

        device = video_feat.device

        # ── 只計算 active 模態的投影 ─────────────────────────────────────────
        emb_map: dict = {}
        if "video" in active_modalities:
            emb_map["video"] = self.video_proj(video_feat).unsqueeze(1)          # (B,1,4096)
        if "ltp" in active_modalities:
            emb_map["ltp"]   = self.ltp_proj(ltp_feat).unsqueeze(1)              # (B,1,4096)
        if "text" in active_modalities:
            emb_map["text"]  = self.text_proj(text_feat).unsqueeze(1)            # (B,1,4096)
        if "music" in active_modalities:
            emb_map["music"] = self.music_proj(
                music_candidates.squeeze(1)                                       # (B,768)
            ).unsqueeze(1)                                                        # (B,1,4096)

        # ── 按 CANONICAL_ORDER 拼接（保持相對順序）───────────────────────────
        embs = [emb_map[m] for m in CANONICAL_ORDER if m in active_modalities]
        prefix = torch.cat(embs, dim=1)   # (B, n_active, 4096)

        # ── 動態 Modality Type Embedding ─────────────────────────────────────
        modality_ids = torch.tensor(
            [MODALITY_ID[m] for m in CANONICAL_ORDER if m in active_modalities],
            dtype=torch.long, device=device,
        )
        type_emb = self.modality_type_emb(modality_ids)   # (n_active, 4096)
        prefix = prefix + type_emb.unsqueeze(0)            # (B, n_active, 4096)

        return prefix
