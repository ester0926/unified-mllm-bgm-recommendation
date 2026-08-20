"""
用途：建立、讀取與檢查 LTP 偏好向量。
輸入：使用者 profile、metadata 或已整理好的偏好資料。
輸出：LTP 向量與相關索引檔。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# P_ltp 融合投影器（需與 ModelConfig.ltp_dim 對齊）
# ─────────────────────────────────────────────────────────────────────────────

class LTPFusionProjector(nn.Module):
    """
    將 implicit(768) + explicit(512) 融合投影至 P_ltp(512)。
    此模組作為 P_ltp 的「前處理器」，獨立於主模型訓練。
    """

    def __init__(self, implicit_dim: int = 768, explicit_dim: int = 512,
                 output_dim: int = 512):
        super().__init__()
        in_dim = implicit_dim + explicit_dim  # 1280

        self.fusion = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, implicit: torch.Tensor, explicit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            implicit : (B, 768) — AST 聚合的行為嵌入
            explicit : (B, 512) — CLIP-T 編碼的語義特徵
        Returns:
            (B, 512) — P_ltp
        """
        combined = torch.cat([implicit, explicit], dim=-1)  # (B, 1280)
        return self.fusion(combined)                         # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# PersonaX 原型性-多樣性平衡採樣
# ─────────────────────────────────────────────────────────────────────────────

def personax_sampling(
    history_features: np.ndarray,   # (N, 768) 用戶歷史音樂的 AST 特徵
    history_weights: Optional[np.ndarray] = None,  # (N,) 互動強度權重（播放次數等）
    n_proto: int = 5,               # 原型性採樣數量（代表核心品味）
    n_diversity: int = 3,           # 多樣性採樣數量（涵蓋探索行為）
    n_clusters: int = 8,            # K-Means 聚類數量
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PersonaX 風格的代表性曲目抽取。

    步驟：
    1. 使用 K-Means 將歷史音樂分群（捕捉不同音樂品味面向）
    2. 原型性採樣：從整體中心出發，選最接近的 n_proto 首（核心偏好）
    3. 多樣性採樣：在每個 cluster 中各抽一首最具代表性的曲目（覆蓋範圍）
    4. 計算加權平均，得到 implicit 偏好向量

    Returns:
        selected_features : (n_proto+n_diversity, 768) 代表性曲目特徵
        aggregated_vec    : (768,) 加權聚合的 implicit 向量
    """
    N, D = history_features.shape
    if N == 0:
        return np.zeros((1, D)), np.zeros(D)

    # 若歷史太短，直接使用全部
    if N <= n_proto + n_diversity:
        agg = np.average(history_features, axis=0,
                         weights=history_weights if history_weights is not None else None)
        return history_features, agg

    # ── K-Means 聚類 ──────────────────────────────────────────────────────
    n_clusters = min(n_clusters, N)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(history_features)
    cluster_centers = kmeans.cluster_centers_  # (n_clusters, D)

    # ── 原型性採樣：距離全域中心最近的曲目 ───────────────────────────────
    if history_weights is not None:
        # 加權中心
        global_center = np.average(history_features, axis=0, weights=history_weights)
    else:
        global_center = history_features.mean(axis=0)

    dists_to_center = np.linalg.norm(history_features - global_center, axis=1)
    proto_indices = np.argsort(dists_to_center)[:n_proto]

    # ── 多樣性採樣：每個 cluster 選距離 cluster center 最近的曲目 ────────
    diversity_indices = []
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        if not cluster_mask.any():
            continue
        cluster_feats = history_features[cluster_mask]
        cluster_orig_indices = np.where(cluster_mask)[0]
        dists = np.linalg.norm(cluster_feats - cluster_centers[c], axis=1)
        best_local = np.argmin(dists)
        diversity_indices.append(cluster_orig_indices[best_local])
        if len(diversity_indices) >= n_diversity:
            break

    # ── 合併代表性曲目索引 ─────────────────────────────────────────────
    all_selected = list(proto_indices) + diversity_indices
    all_selected = list(dict.fromkeys(all_selected))  # 去重保序
    selected_features = history_features[all_selected]

    # ── 加權聚合（播放次數加權）──────────────────────────────────────────
    if history_weights is not None:
        selected_weights = history_weights[all_selected]
        selected_weights = selected_weights / selected_weights.sum()
        aggregated_vec = np.average(selected_features, axis=0, weights=selected_weights)
    else:
        aggregated_vec = selected_features.mean(axis=0)

    return selected_features, aggregated_vec


# ─────────────────────────────────────────────────────────────────────────────
# RecLLM 自然語言 Persona 生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_persona_text(
    selected_tracks: List[Dict],    # 代表性曲目的 metadata（genre, tempo, mood 等）
    model_name: str = "gpt-3.5-turbo",
    max_tokens: int = 150,
) -> str:
    """
    使用 LLM 從代表性曲目 metadata 生成自然語言偏好描述。

    RecLLM 風格的 Persona 生成 prompt，包含：
    - 顯著事實抽取（最常出現的 genre, mood）
    - 偏好衝突調節（處理矛盾的歷史行為）
    - 可供 LLaMA 推理的文字格式

    Returns:
        str: 自然語言偏好描述（最多 77 tokens，CLIP 上限）
    """
    # 統計最常出現的特徵
    genres = [t.get("genre", "unknown") for t in selected_tracks if t]
    moods = [t.get("mood", "unknown") for t in selected_tracks if t]
    tempos = [t.get("tempo_category", "medium") for t in selected_tracks if t]

    genre_count = {}
    for g in genres:
        genre_count[g] = genre_count.get(g, 0) + 1
    top_genres = sorted(genre_count, key=genre_count.get, reverse=True)[:3]

    # 簡化版：直接組合統計結果生成 Persona 文字
    # 完整版應呼叫 OpenAI API 或本地 LLM
    persona_text = (
        f"User prefers {', '.join(top_genres)} music. "
        f"Preferred mood: {', '.join(set(moods[:3]))}. "
        f"Common tempo: {max(set(tempos), key=tempos.count) if tempos else 'medium'}."
    )

    # 截斷至 CLIP 可接受的長度（約 77 tokens ≈ 300 字元）
    return persona_text[:280]


def encode_persona_text(persona_text: str, clip_text_model=None) -> np.ndarray:
    """
    用 CLIP Text Encoder 將 Persona 文字編碼為 512 維向量。

    Args:
        persona_text    : 自然語言偏好描述
        clip_text_model : 已載入的 CLIP 模型（None 則回傳零向量）

    Returns:
        (512,) numpy array — 顯性語義特徵向量
    """
    if clip_text_model is None:
        logger.warning("CLIP model not provided, returning zero vector for explicit feat")
        return np.zeros(512, dtype=np.float32)

    try:
        import clip
        text_tokens = clip.tokenize([persona_text], truncate=True)
        with torch.no_grad():
            text_features = clip_text_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"CLIP 編碼失敗: {e}")
        return np.zeros(512, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# P_ltp 向量建構主函數
# ─────────────────────────────────────────────────────────────────────────────

def build_ltp_vector(
    user_history: List[Dict],       # 用戶歷史音樂列表，每項含 ast_features(768)
    ltp_projector: LTPFusionProjector,
    clip_text_model=None,
    n_proto: int = 5,
    n_diversity: int = 3,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    完整建構單一用戶的 P_ltp 向量。

    Args:
        user_history  : 用戶音樂歷史（含 ast_features 欄位）
        ltp_projector : 已訓練的融合投影器
        clip_text_model: CLIP 模型（用於 explicit 特徵）
        n_proto, n_diversity: PersonaX 採樣參數

    Returns:
        (512,) numpy array — P_ltp 混合偏好向量
    """
    if not user_history:
        return np.zeros(512, dtype=np.float32)

    # ── 步驟 1：PersonaX 採樣，得到 implicit 偏好向量 ────────────────────
    history_features = np.stack([
        h["ast_features"] for h in user_history
        if "ast_features" in h and h["ast_features"] is not None
    ])  # (N, 768)

    history_weights = np.array([
        h.get("play_count", 1) for h in user_history
        if "ast_features" in h
    ], dtype=np.float32)

    selected_feats, implicit_vec = personax_sampling(
        history_features=history_features,
        history_weights=history_weights,
        n_proto=n_proto,
        n_diversity=n_diversity,
    )
    # implicit_vec: (768,)

    # ── 步驟 2：RecLLM Persona 生成，得到 explicit 特徵向量 ──────────────
    selected_tracks_meta = [
        user_history[i] for i in range(min(len(selected_feats), len(user_history)))
    ]
    persona_text = generate_persona_text(selected_tracks_meta)
    explicit_vec = encode_persona_text(persona_text, clip_text_model)
    # explicit_vec: (512,)

    # ── 步驟 3：融合投影至 512 維 P_ltp ──────────────────────────────────
    ltp_projector.eval()
    with torch.no_grad():
        implicit_t = torch.tensor(implicit_vec, dtype=torch.float32).unsqueeze(0).to(device)
        explicit_t = torch.tensor(explicit_vec, dtype=torch.float32).unsqueeze(0).to(device)
        ltp_vec = ltp_projector(implicit_t, explicit_t)
        # ltp_vec: (1, 512)

    return ltp_vec.squeeze(0).cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 批量建構並寫入 HDF5
# ─────────────────────────────────────────────────────────────────────────────

def batch_build_ltp(
    user_history_dict: Dict[str, List[Dict]],  # {user_id: [track_records...]}
    output_dir: str,
    ltp_projector: LTPFusionProjector,
    clip_text_model=None,
    device: torch.device = torch.device("cpu"),
):
    """
    批量為所有用戶建構 P_ltp 向量，儲存為 {user_id}.npy 格式。

    使用方式：
        在資料預處理階段執行此函數，生成的 .npy 檔案會在
        UnifiedMLLMDataset.__getitem__ 中讀取。
    """
    os.makedirs(output_dir, exist_ok=True)

    for user_id, history in user_history_dict.items():
        ltp_vec = build_ltp_vector(
            user_history=history,
            ltp_projector=ltp_projector,
            clip_text_model=clip_text_model,
            device=device,
        )
        save_path = os.path.join(output_dir, f"{user_id}_ltp.npy")
        np.save(save_path, ltp_vec)

    logger.info(f"P_ltp 向量已儲存至: {output_dir} ({len(user_history_dict)} 用戶)")
