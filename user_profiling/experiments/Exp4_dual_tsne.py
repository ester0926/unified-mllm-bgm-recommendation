"""
實驗四：雙空間 t-SNE 偏好對齊可視化 (v8)
========================================================
佈局：1 行 4 欄

  Panel 1  Text Space t-SNE（定性）
  Panel 2  Cross-Modal Alignment（跨模態對齊，p<0.001）
  Panel 3  Cross-Modal Hit@K 檢索測試（全庫 ~74,000 首）← 主要改動
  Panel 4  Audio Space t-SNE + Convex Hull + 錨定率

v8 改動摘要（vs v7）：

  Panel 3 query 修正（關鍵）：
    v7：query = P_ltp_implicit（Core Music 偏好中心）
        → Core Music ≠ Target Music，找不到正確答案（Hit = 0%）

    v8：query = W_explicit(raw_text[i])（文字偏好投影到 256D）
        pool  = W_implicit(target_ast[j])（全庫 ~74,000 首音訊投影）
        → 和 Panel 2 完全一致的跨模態框架
        → 問題變成：「給定用戶的文字偏好，能否從全庫找到對應的目標音樂？」
        → Panel 2 的 Δμ=+0.072 已確認跨模態信號強，Hit@K 應顯著優於隨機

  另外修正：移除殘留的舊版執行呼叫，確保腳本只執行一次

使用方式：
  cd "<repo_root>"
  python experiments/Exp4_dual_tsne_v8.py
  （若要取代 Exp4_dual_tsne.py，請用本檔案的全部內容覆蓋舊檔）
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# ★ CONFIG — 只需修改這裡
# ============================================================
class CFG:
    # ── MuseChat HDF5 資料夾（全部 106 個檔案）──────────────
    MUSECHAT_DIR = Path(r"data/optimized_musechat_features_float16_v3")

    # ── LTP Pipeline 路徑 ────────────────────────────────────
    BASE_LTP    = Path(r"data/user_profiling/long_term_preference")
    HISTORY_DIR = BASE_LTP / "stage2_history/personax"
    PROFILES    = BASE_LTP / "stage4_recLLM/profiles.jsonl"

    # ── Stage 5 輸出路徑 ─────────────────────────────────────
    STAGE5_DIR          = Path(r"data/user_profiling/stage5_output")
    PROJECTION_WEIGHTS  = STAGE5_DIR / "projection_weights.pt"
    PLTP_EXPLICIT       = STAGE5_DIR / "preference_vectors_explicit_only.h5"
    PLTP_IMPLICIT       = STAGE5_DIR / "preference_vectors_implicit_only.h5"

    # ── CLIP Text Encoder ────────────────────────────────────
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    # ── 輸出 ─────────────────────────────────────────────────
    OUTPUT_DIR  = Path(r"user_profiling\experiments\visualization_outputs")
    OUTPUT_FILE = OUTPUT_DIR / "exp4_dual_tsne.png"

    # ── 取樣數量 ─────────────────────────────────────────────
    N_USERS         = 40   # 視覺化用戶總數
    N_CORE_PER_USER = 5    # 每個用戶最多取幾首 core music
    N_SHOW_USERS    = 12   # Panel 4 標色的用戶數（建議 10~15）

    # ── t-SNE 參數 ───────────────────────────────────────────
    PERPLEXITY_TEXT  = 15
    PERPLEXITY_AUDIO = 20
    PCA_DIM          = 50

    # ── 投影層維度（需與 Stage 5 一致）──────────────────────
    CLIP_DIM    = 512    # CLIP text encoder output
    AST_DIM     = 768    # AST audio encoder output
    PROJ_DIM    = 256    # W_explicit / W_implicit output（Stage 5 OUTPUT_DIM）

    # ── 視覺參數 ─────────────────────────────────────────────
    COLORS = {
        'raw_dialogue':  '#2E86AB',
        'llm_pref':      '#F6AE2D',
        'paired_sim':    '#27AE60',
        'random_sim':    '#E74C3C',
        'other_core':    '#CCCCCC',
        'other_target':  '#999999',
        'bg':            '#F8F9FA',
        'panel_bg':      '#FFFFFF',
        'title':         '#1A1A2E',
        'grid':          '#E0E0E0',
    }

    # ── Hit@K 參數（Panel 3）───────────────────────────────
    # 檢索池大小 = N_USERS（每個用戶的 target music 各作為一個選項）
    # 若想加入額外 distractor，可調高此數（最多不超過 N_USERS）
    HITK_TOPK   = [1, 5, 10, 20]   # 要報告的 K 值

    MUSIC_ID_FROM = 'prefix'


# ============================================================
# 工具函數
# ============================================================

def get_cjk_font() -> FontProperties:
    from matplotlib import font_manager as fm
    candidates = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
                  'PingFang TC', 'Noto Sans CJK TC']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return FontProperties(family=name)
    return FontProperties(family='DejaVu Sans')


def music_id_from_key(pair_key: str) -> str:
    parts = pair_key.rsplit('_', 1)
    if len(parts) != 2:
        return pair_key
    return parts[0] if CFG.MUSIC_ID_FROM == 'prefix' else parts[1]


def pool_text(feat: np.ndarray) -> np.ndarray:
    """text_features [77, 512] → Masked Mean Pooling → [512]"""
    feat  = feat.astype(np.float32)
    norms = np.linalg.norm(feat, axis=1)
    mask  = (norms > 0.01).astype(np.float32)
    if mask.sum() == 0:
        mask = np.ones(len(feat), dtype=np.float32)
    return (feat * mask[:, None]).sum(axis=0) / max(mask.sum(), 1e-9)


def pool_audio(feat: np.ndarray) -> np.ndarray:
    """target_music_all_cls [12, 768] → mean pool → [768]"""
    return feat.astype(np.float32).mean(axis=0)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A [N,D], B [N,D] → [N,N]"""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_n @ B_n.T


def paired_vs_random_stats(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    計算 cosine sim matrix，回傳：
      paired_sims  [N]          : diagonal（同 pair 的相似度）
      random_sims  [N*(N-1)]    : off-diagonal（跨 pair 的相似度）
      p_value                   : Mann-Whitney U（paired > random）
    """
    sim_mat     = cosine_sim_matrix(A, B)
    N           = len(A)
    paired_sims = np.array([sim_mat[i, i] for i in range(N)])
    random_sims = np.array([sim_mat[i, j] for i in range(N)
                            for j in range(N) if i != j])
    _, p_value  = mannwhitneyu(paired_sims, random_sims, alternative='greater')
    return paired_sims, random_sims, p_value


def build_full_pool(
    pltp_imp_map: Dict[str, np.ndarray],   # 全庫 84,150 筆 P_ltp implicit
    ast_index:    Dict[str, np.ndarray],   # 全庫 AST 向量索引
    proj:         'ProjectionWeights',
) -> Tuple[List[str], np.ndarray]:
    """
    對全庫所有有 AST 向量的 music_id 建立 W_implicit 投影 pool。

    策略：
      1. 以 pltp_imp_map 的所有 key（84,150 個 music_id）為基準
      2. 在 ast_index 中查找對應的 AST 768D 向量
      3. 批次過 W_implicit 投影到 256D
      4. 回傳 (pool_ids [M_pool], pool_proj [M_pool, 256])

    記憶體估算：84,150 × 256 × float32 ≈ 86MB，可接受
    """
    logger.info("建立全庫音樂 pool（W_implicit 投影）...")
    pool_ids:  List[str]        = []
    pool_ast:  List[np.ndarray] = []

    for mid, _ in pltp_imp_map.items():
        if mid in ast_index:
            pool_ids.append(mid)
            pool_ast.append(ast_index[mid])

    if not pool_ids:
        raise RuntimeError(
            "全庫 pool 為空！請確認 ast_index 和 pltp_imp_map 的 music_id 格式一致。"
        )

    ast_arr   = np.array(pool_ast, dtype=np.float32)        # [M_pool, 768]
    pool_proj = proj.project_implicit(ast_arr)               # [M_pool, 256]

    logger.info(f"全庫 pool 建立完成：{len(pool_ids)} 首音樂 → pool shape {pool_proj.shape}")
    return pool_ids, pool_proj


def compute_hitk_full(
    pltp_imp:   np.ndarray,   # [N, 256]  P_ltp implicit（query）
    pool_ids:   List[str],    # [M_pool]  全庫 music_id
    pool_proj:  np.ndarray,   # [M_pool, 256]  W_implicit(ast)（key）
    target_ids: List[str],    # [N]  每個用戶的正確答案 music_id
    topk_list:  List[int],
) -> Tuple[Dict[int, float], float]:
    """
    全庫 Hit@K 檢索測試。

    對每個用戶 i：
      1. 計算 pltp_imp[i] 與 pool_proj 所有向量的 cosine sim
      2. 降序排名，看 target_ids[i] 是否在 Top-K 內
      3. 計算 MRR（Mean Reciprocal Rank）

    隨機基準：Hit@K_random = K / M_pool
    """
    N      = len(pltp_imp)
    M_pool = len(pool_ids)

    # 建立 music_id → pool index 的快速查找表
    id_to_idx = {mid: i for i, mid in enumerate(pool_ids)}

    # 正規化向量（cosine sim = dot product after L2-norm）
    q_norm = pltp_imp / (np.linalg.norm(pltp_imp, axis=1, keepdims=True) + 1e-9)
    k_norm = pool_proj / (np.linalg.norm(pool_proj, axis=1, keepdims=True) + 1e-9)

    hitk_results:    Dict[int, float] = {k: 0 for k in topk_list}
    reciprocal_ranks: List[float]     = []
    n_found = 0

    logger.info(f"Hit@K 檢索中（N={N} queries，pool={M_pool}）...")
    for i in tqdm(range(N), desc="Hit@K 全庫搜索"):
        correct_id = target_ids[i]
        if correct_id not in id_to_idx:
            # 正確答案不在 pool 中（理論上不應發生）
            reciprocal_ranks.append(0.0)
            continue
        n_found += 1
        correct_idx = id_to_idx[correct_id]

        sims  = q_norm[i] @ k_norm.T               # [M_pool]
        ranks = np.argsort(-sims)                   # 降序排名

        rank_of_correct = int(np.where(ranks == correct_idx)[0][0]) + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank_of_correct)

        for k in topk_list:
            if correct_idx in set(ranks[:k]):
                hitk_results[k] += 1

    # 轉為比例
    denom = max(n_found, 1)
    hitk_results = {k: v / denom for k, v in hitk_results.items()}
    mrr = float(np.mean(reciprocal_ranks))

    logger.info(f"Hit@K 完成：{n_found}/{N} 個查詢的正確答案在 pool 中")
    return hitk_results, mrr, M_pool


def compute_intra_user_consistency(
    pltp_exp: np.ndarray,   # [N, 256]  P_ltp explicit_only
    pltp_imp: np.ndarray,   # [N, 256]  P_ltp implicit_only
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    用戶內一致性：同一用戶的 explicit vs implicit P_ltp cosine sim
    應大於跨用戶版本。

    回傳：
      intra_sims  [N]       : diagonal（同用戶）
      inter_sims  [N*(N-1)] : off-diagonal（跨用戶）
      p_value               : Mann-Whitney U（intra > inter）
    """
    return paired_vs_random_stats(pltp_exp, pltp_imp)

def point_in_hull(point: np.ndarray, hull_pts: np.ndarray) -> bool:
    if len(hull_pts) < 3:
        return False
    try:
        hull = ConvexHull(hull_pts)
        return bool(
            np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-9)
        )
    except Exception:
        return False


def run_tsne(matrix: np.ndarray, perplexity: int) -> np.ndarray:
    n       = matrix.shape[0]
    pca_dim = min(CFG.PCA_DIM, n - 1, matrix.shape[1])
    if pca_dim >= 2:
        matrix = PCA(n_components=pca_dim, random_state=42).fit_transform(matrix)
    perp = min(perplexity, max(2, n // 3))
    return TSNE(n_components=2, perplexity=perp, n_iter=2000,
                random_state=42, init='pca',
                learning_rate='auto').fit_transform(matrix)


def draw_convex_hull(ax, points: np.ndarray, color: str, alpha: float = 0.11):
    if len(points) < 3:
        return
    try:
        hull     = ConvexHull(points)
        hull_pts = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                color=color, alpha=alpha, zorder=1)
        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                color=color, linewidth=0.9, alpha=0.55, zorder=2)
    except Exception:
        pass


# ============================================================
# Step 1: 載入 Stage 5 投影層權重
# ============================================================

class ProjectionWeights:
    """
    從 projection_weights.pt 載入 W_explicit 和 W_implicit，
    提供 numpy forward（不需要 GPU）。

    Stage 5 儲存格式（projection_weights.pt）：
      {
        'W_explicit': state_dict  # {'weight': [256,512], 'bias': [256]}
        'W_implicit': state_dict  # {'weight': [256,768], 'bias': [256]}
      }
    """
    def __init__(self, path: Path, device: str = 'cpu'):
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 projection_weights.pt：{path}\n"
                f"請先執行 stage5_preference_representation_v4.py"
            )
        ckpt = torch.load(path, map_location=device)
        # Stage 5 用 torch.save({'W_explicit': linear.state_dict(), ...}) 儲存
        # 所以 ckpt['W_explicit'] 是一個 state_dict，key 為 'weight' / 'bias'
        self._w_exp = ckpt['W_explicit']['weight'].float().numpy()  # [256, 512]
        self._b_exp = ckpt['W_explicit']['bias'].float().numpy()    # [256]
        self._w_imp = ckpt['W_implicit']['weight'].float().numpy()  # [256, 768]
        self._b_imp = ckpt['W_implicit']['bias'].float().numpy()    # [256]
        logger.info(f"投影層權重載入完成：W_explicit {self._w_exp.shape}，"
                    f"W_implicit {self._w_imp.shape}")

    def project_explicit(self, x: np.ndarray) -> np.ndarray:
        """x [N, 512] → [N, 256]"""
        return x @ self._w_exp.T + self._b_exp

    def project_implicit(self, x: np.ndarray) -> np.ndarray:
        """x [N, 768] → [N, 256]"""
        return x @ self._w_imp.T + self._b_imp


# ============================================================
# Step 2: 載入 Stage 5 P_ltp（explicit / implicit）
# ============================================================

def load_pltp_all(path: Path) -> Dict[str, np.ndarray]:
    """
    將整個 preference_vectors_*.h5 載入記憶體。
    （84150 × 256 × float32 ≈ 86MB，可接受）
    回傳 {music_id: vec [256D]}。
    """
    if not path.exists():
        logger.warning(f"P_ltp 檔案不存在：{path}")
        return {}
    result: Dict[str, np.ndarray] = {}
    with h5py.File(path, 'r') as f:
        grp = f.get('preference_vectors', f)
        for mid in grp.keys():
            result[mid] = grp[mid][()].astype(np.float32)
    logger.info(f"P_ltp 全量載入：{len(result)} 筆  ({path.name})")
    return result


# ============================================================
# Step 3: 建立 Stage 4 索引
# ============================================================

def build_profile_index() -> Dict[str, str]:
    """回傳 {music_id: llm_text}（優先 salient_facts）"""
    index: Dict[str, str] = {}
    if not CFG.PROFILES.exists():
        logger.warning(f"Stage 4 profiles 不存在: {CFG.PROFILES}")
        return index
    import jsonlines
    with jsonlines.open(CFG.PROFILES, 'r') as reader:
        for obj in reader:
            mid = obj.get('music_id') or obj.get('target_music', '')
            if not mid:
                continue
            facts = obj.get('salient_facts', [])
            if facts:
                text = '. '.join(
                    f.get('fact', '') for f in facts
                    if isinstance(f, dict) and f.get('fact')
                ).strip()
            else:
                text = obj.get('summary_text', '').strip()
            if mid and text:
                index[mid] = text
    logger.info(f"Stage 4 索引：{len(index)} 筆")
    return index


# ============================================================
# Step 4: 建立 AST 索引（全 106 HDF5）
# ============================================================

def build_ast_index() -> Dict[str, np.ndarray]:
    """掃描全部 HDF5，以 video_id / candidate_id / pair_key 三種 key 建立 AST 索引"""
    h5_files = sorted(CFG.MUSECHAT_DIR.glob("musechat_features_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"找不到任何 HDF5：{CFG.MUSECHAT_DIR}")
    logger.info(f"掃描 {len(h5_files)} 個 HDF5 建立 AST 索引...")
    index: Dict[str, np.ndarray] = {}
    for fpath in tqdm(h5_files, desc="建立 AST 索引"):
        try:
            with h5py.File(fpath, 'r') as f:
                grp = f.get('pairs', f)
                for key in grp.keys():
                    try:
                        sub = grp[key]
                        if 'target_music_all_cls' not in sub:
                            continue
                        vec   = pool_audio(sub['target_music_all_cls'][()])
                        parts = key.rsplit('_', 1)
                        if len(parts) == 2:
                            index.setdefault(parts[0], vec)
                            index.setdefault(parts[1], vec)
                        index.setdefault(key, vec)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"無法讀取 {fpath.name}: {e}")
    logger.info(f"AST 索引：{len(index)} 個 key")
    return index


# ============================================================
# Step 5: 讀取 Stage 2 history
# ============================================================

def load_core_music_ids(music_id: str) -> List[str]:
    path = CFG.HISTORY_DIR / f"{music_id}__history.json"
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            hist = json.load(fh)
        core_sbs = hist.get('balanced_history', {}).get('core_sbs', [])
        return [x.get('music_id', '') for x in core_sbs if x.get('music_id')]
    except Exception:
        return []


# ============================================================
# Step 6: CLIP 編碼
# ============================================================

def encode_with_clip(texts: List[str], model, tokenizer,
                     device: str, batch_size: int = 64) -> np.ndarray:
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch   = texts[i: i + batch_size]
            inputs  = tokenizer(batch, return_tensors='pt', padding=True,
                                truncation=True, max_length=77).to(device)
            outputs = model(**inputs)
            if outputs.pooler_output is not None:
                vecs = outputs.pooler_output.cpu().float().numpy()
            else:
                hidden = outputs.last_hidden_state
                mask   = inputs['attention_mask'].unsqueeze(-1).float()
                vecs   = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vecs   = vecs.cpu().float().numpy()
            all_vecs.append(vecs)
    return np.vstack(all_vecs)


# ============================================================
# Step 7: 掃描 HDF5，組裝所有需要的向量
# ============================================================

def load_all_data(
    profile_index: Dict[str, str],
    ast_index:     Dict[str, np.ndarray],
    n_users:       int,
    clip_model, clip_tokenizer, device: str,
    proj: ProjectionWeights,
    pltp_exp_map:  Dict[str, np.ndarray],
    pltp_imp_map:  Dict[str, np.ndarray],
):
    """
    回傳：
      raw_clip      [N, 512]   HDF5 text_features → CLIP masked mean pool
      llm_clip      [N, 512]   salient_facts → CLIP（Panel 1 用）
      raw_proj      [N, 256]   raw_clip → W_explicit（Panel 2 左側）
      tgt_proj      [N, 256]   target_ast → W_implicit（Panel 2 右側 & Panel 3 pool）
      pltp_exp      [N, 256]   P_ltp explicit_only（用戶內一致性用）
      pltp_imp      [N, 256]   P_ltp implicit_only（Panel 3 query）
      core_audio    [M, 768]   Core Historical Music AST（Panel 4 用）
      tgt_audio     [N, 768]   Target Music AST（Panel 4 用）
      user_labels   [M]        core music → user index（Panel 4 用）
      matched_ids   [N]
    """
    h5_files = sorted(CFG.MUSECHAT_DIR.glob("musechat_features_*.h5"))
    logger.info(f"讀取 {len(h5_files)} 個 MuseChat HDF5...")

    # 自動偵測 music_id 位置
    with h5py.File(h5_files[0], 'r') as f:
        pg          = f.get('pairs', f)
        sample_keys = list(pg.keys())[:min(200, len(pg))]
    profile_ids = set(profile_index.keys())
    pfx = sum(1 for k in sample_keys if k.split('_')[0]     in profile_ids)
    sfx = sum(1 for k in sample_keys if k.rsplit('_',1)[-1] in profile_ids)
    CFG.MUSIC_ID_FROM = 'prefix' if pfx >= sfx else 'suffix'
    logger.info(f"  music_id 位置：{CFG.MUSIC_ID_FROM} (prefix={pfx} suffix={sfx})")

    # ── 第一輪：收集候選 ──────────────────────────────────────
    LIMIT = n_users * 5
    cand_ids, cand_raw_clip, cand_tgt_ast, cand_core_ids = [], [], [], []
    seen: set = set()

    for fpath in tqdm(h5_files, desc="收集候選"):
        if len(cand_ids) >= LIMIT:
            break
        try:
            with h5py.File(fpath, 'r') as f:
                pairs_grp = f.get('pairs', f)
                for key in pairs_grp.keys():
                    if len(cand_ids) >= LIMIT:
                        break
                    try:
                        mid = music_id_from_key(key)
                        if mid not in profile_ids or mid in seen:
                            continue
                        sub = pairs_grp[key]
                        if 'text_features'        not in sub: continue
                        if 'target_music_all_cls' not in sub: continue
                        seen.add(mid)
                        cand_ids.append(mid)
                        cand_raw_clip.append(pool_text(sub['text_features'][()]))
                        cand_tgt_ast.append(pool_audio(sub['target_music_all_cls'][()]))
                        cand_core_ids.append(load_core_music_ids(mid))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"無法讀取 {fpath.name}: {e}")

    logger.info(f"候選 pair 數：{len(cand_ids)}")

    # ── 第二輪：篩選需要四種 P_ltp 和 core music 都存在的用戶 ─
    raw_clip_list, llm_texts = [], []
    tgt_ast_list             = []
    pltp_exp_list            = []
    pltp_imp_list            = []
    core_list, tgt_list      = [], []
    user_labels: List[int]   = []
    matched_ids: List[str]   = []

    for mid, raw_clip, tgt_ast, core_ids in zip(
            cand_ids, cand_raw_clip, cand_tgt_ast, cand_core_ids):
        if len(matched_ids) >= n_users:
            break
        # 四個條件都要滿足
        if mid not in pltp_exp_map:
            continue
        if mid not in pltp_imp_map:
            continue
        core_vecs = [ast_index[cid] for cid in core_ids if cid in ast_index]
        if not core_vecs:
            continue

        u = len(matched_ids)
        raw_clip_list.append(raw_clip)
        llm_texts.append(profile_index[mid])
        tgt_ast_list.append(tgt_ast)
        pltp_exp_list.append(pltp_exp_map[mid])
        pltp_imp_list.append(pltp_imp_map[mid])
        for v in core_vecs[:CFG.N_CORE_PER_USER]:
            core_list.append(v)
            user_labels.append(u)
        tgt_list.append(tgt_ast)    # Panel 4 用原始 AST
        matched_ids.append(mid)

    if not matched_ids:
        raise RuntimeError(
            "找不到匹配用戶，請確認路徑：\n"
            f"  MuseChat HDF5  : {CFG.MUSECHAT_DIR}\n"
            f"  Stage 4        : {CFG.PROFILES}\n"
            f"  Stage 2        : {CFG.HISTORY_DIR}\n"
            f"  P_ltp explicit : {CFG.PLTP_EXPLICIT}\n"
            f"  P_ltp implicit : {CFG.PLTP_IMPLICIT}"
        )

    logger.info(f"最終取樣：{len(matched_ids)} 個用戶，{len(core_list)} 首 core music")

    # ── 陣列轉換 ─────────────────────────────────────────────
    raw_clip_arr = np.array(raw_clip_list, dtype=np.float32)    # [N, 512]
    tgt_ast_arr  = np.array(tgt_ast_list,  dtype=np.float32)    # [N, 768]
    pltp_exp_arr = np.array(pltp_exp_list, dtype=np.float32)    # [N, 256]
    pltp_imp_arr = np.array(pltp_imp_list, dtype=np.float32)    # [N, 256]
    core_arr     = np.array(core_list,     dtype=np.float32)    # [M, 768]
    tgt_arr      = np.array(tgt_list,      dtype=np.float32)    # [N, 768]

    # ── CLIP 編碼 LLM salient_facts（Panel 1 用）────────────
    logger.info("CLIP 編碼 LLM salient_facts...")
    llm_clip_arr = encode_with_clip(llm_texts, clip_model, clip_tokenizer, device)

    # ── W_explicit 投影 raw_text（Panel 2 用）────────────────
    logger.info("W_explicit 投影 text_features → 256D...")
    raw_proj_arr = proj.project_explicit(raw_clip_arr)          # [N, 256]

    # ── W_implicit 投影 target_ast（Panel 3 用）─────────────
    logger.info("W_implicit 投影 target_music_ast → 256D...")
    tgt_proj_arr = proj.project_implicit(tgt_ast_arr)           # [N, 256]

    return (
        raw_clip_arr,   # [N, 512]  Panel 1 raw
        llm_clip_arr,   # [N, 512]  Panel 1 llm
        raw_proj_arr,   # [N, 256]  Panel 2 左側
        tgt_proj_arr,   # [N, 256]  Panel 2 右側 & Panel 3 pool key
        pltp_exp_arr,   # [N, 256]  用戶內一致性
        pltp_imp_arr,   # [N, 256]  Panel 3 query
        core_arr,       # [M, 768]  Panel 4 core
        tgt_arr,        # [N, 768]  Panel 4 target
        user_labels,    # [M]       Panel 4
        matched_ids,    # [N]
    )


# ============================================================
# Step 8: 繪圖（1×4）
# ============================================================

def draw_sim_panel(
    ax,
    paired_sims:  np.ndarray,
    random_sims:  np.ndarray,
    p_value:      float,
    title:        str,
    xlabel:       str,
    paired_label: str,
    random_label: str,
    C:            dict,
):
    """通用的 cosine similarity 分布子圖繪製函數（Panel 2 / Panel 3 共用）"""
    ax.set_facecolor(C['panel_bg'])
    ax.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, axis='y', zorder=0)

    mu_pair = paired_sims.mean()
    mu_rand = random_sims.mean()
    all_vals = np.concatenate([paired_sims, random_sims])
    bins = np.linspace(all_vals.min() - 0.02, all_vals.max() + 0.02, 32)

    ax.hist(random_sims, bins=bins, color=C['random_sim'], alpha=0.55,
            label=f'{random_label}  (μ={mu_rand:.3f})', zorder=2, density=True)
    ax.hist(paired_sims, bins=bins, color=C['paired_sim'],  alpha=0.75,
            label=f'{paired_label}  (μ={mu_pair:.3f})', zorder=3, density=True)
    ax.axvline(mu_rand, color=C['random_sim'], linewidth=1.8,
               linestyle='--', zorder=4)
    ax.axvline(mu_pair, color=C['paired_sim'],  linewidth=1.8,
               linestyle='-',  zorder=5)

    ax.set_title(title, fontsize=9.5, fontweight='bold',
                 color=C['title'], pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('Density', fontsize=8.5)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9,
              edgecolor=C['grid'])

    # p-value 標注
    sig   = p_value < 0.05
    color = C['paired_sim'] if sig else C['random_sim']
    label = '✓ Significant' if sig else '✗ Not significant'
    p_str = f'p = {p_value:.2e}' if p_value >= 1e-4 else 'p < 1e-4'
    ax.text(0.97, 0.97,
            f'Mann-Whitney U\n{p_str}\n{label}\nΔμ = {mu_pair - mu_rand:+.4f}',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      edgecolor=color, alpha=0.95))


def plot_all(
    raw_clip:    np.ndarray,   # [N, 512]
    llm_clip:    np.ndarray,   # [N, 512]
    raw_proj:    np.ndarray,   # [N, 256]  W_explicit(text)
    tgt_proj:    np.ndarray,   # [N, 256]  W_implicit(target_ast)
    pltp_exp:    np.ndarray,   # [N, 256]  P_ltp explicit_only
    pltp_imp:    np.ndarray,   # [N, 256]  P_ltp implicit_only
    core_audio:  np.ndarray,   # [M, 768]
    tgt_audio:   np.ndarray,   # [N, 768]
    user_labels: List[int],    # [M]
    pool_ids:    List[str],    # [M_pool]  全庫 music_id
    pool_proj:   np.ndarray,   # [M_pool, 256]  W_implicit(ast) 全庫投影
    target_ids:  List[str],    # [N]  每個用戶的 target music_id
):
    C   = CFG.COLORS
    N   = len(raw_clip)
    M   = len(core_audio)
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False})
    get_cjk_font()

    n_show     = min(CFG.N_SHOW_USERS, N)
    palette    = plt.cm.get_cmap('tab20', n_show)
    usr_colors = [mcolors.to_hex(palette(i)) for i in range(n_show)]

    # ── 預先計算所有統計 ─────────────────────────────────────

    # Panel 2（修正版）：跨模態對齊
    # W_explicit(text[i]) · W_implicit(target_ast[i]) > 隨機跨用戶對
    logger.info("Panel 2：計算跨模態 cosine similarity（W_exp text vs W_imp audio）...")
    p2_paired, p2_random, p2_pval = paired_vs_random_stats(raw_proj, tgt_proj)

    # Panel 3：Cross-Modal Hit@K 全庫檢索測試（v8 修正版）
    # query  = W_explicit(raw_text[i])   ← 文字偏好投影（和 Panel 2 一致的跨模態方向）
    # pool   = W_implicit(target_ast[j]) ← 全庫音訊投影
    # 正確答案 = j == i（同一用戶的目標音樂）
    logger.info("Panel 3：計算 Cross-Modal Hit@K（全庫 pool，query=W_exp(text)）...")
    hitk_results, mrr, pool_size = compute_hitk_full(
        raw_proj, pool_ids, pool_proj, target_ids, CFG.HITK_TOPK
    )
    random_baselines = {k: k / pool_size for k in CFG.HITK_TOPK}

    # 用戶內一致性（終端機輸出用）
    logger.info("計算用戶內一致性（explicit vs implicit P_ltp）...")
    intra_sims, inter_sims, intra_pval = compute_intra_user_consistency(
        pltp_exp, pltp_imp
    )

    # Panel 1：t-SNE
    logger.info("Panel 1：Text Space t-SNE...")
    text_all = normalize(np.vstack([raw_clip, llm_clip]))
    text_2d  = run_tsne(text_all, CFG.PERPLEXITY_TEXT)
    raw_2d, llm_2d = text_2d[:N], text_2d[N:]

    # Panel 4：t-SNE
    logger.info("Panel 4：Audio Space t-SNE...")
    audio_all = normalize(np.vstack([core_audio, tgt_audio]))
    audio_2d  = run_tsne(audio_all, CFG.PERPLEXITY_AUDIO)
    core_2d, tgt_2d = audio_2d[:M], audio_2d[M:]

    logger.info("Panel 4：計算錨定率...")
    contained, n_contained = [], 0
    for u in range(N):
        c_mask = [i for i, lbl in enumerate(user_labels) if lbl == u]
        inside = point_in_hull(tgt_2d[u], core_2d[c_mask]) if len(c_mask) >= 3 else False
        contained.append(inside)
        if inside:
            n_contained += 1
    contain_rate = n_contained / N

    # ── 佈局：1 × 4 ─────────────────────────────────────────
    fig = plt.figure(figsize=(26, 7.5), facecolor=C['bg'])
    gs  = fig.add_gridspec(1, 4, wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    fig.suptitle(
        'Experiment 4  ·  P_ltp Effectiveness Validation  (N={} users)'.format(N),
        fontsize=12, fontweight='bold', color=C['title'], y=1.02,
    )

    # ════════════════════════════════════════════════════════
    # Panel 1：Text Space t-SNE（定性）
    # ════════════════════════════════════════════════════════
    ax1.set_facecolor(C['panel_bg'])
    ax1.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, zorder=0)
    for i in range(N):
        ax1.plot([raw_2d[i,0], llm_2d[i,0]], [raw_2d[i,1], llm_2d[i,1]],
                 color='#BBBBBB', linewidth=0.4, alpha=0.4, zorder=1)
    ax1.scatter(raw_2d[:,0], raw_2d[:,1],
                c=C['raw_dialogue'], s=55, marker='s', alpha=0.8,
                edgecolors='white', linewidths=0.5, zorder=3,
                label=f'Raw Dialogue (N={N})')
    ax1.scatter(llm_2d[:,0], llm_2d[:,1],
                c=C['llm_pref'], s=80, marker='*', alpha=0.88,
                edgecolors='white', linewidths=0.4, zorder=4,
                label=f'LLM Explicit Pref (N={N})')
    ax1.set_title('Panel 1 · Text Space t-SNE\n(CLIP 512D → t-SNE 2D)',
                  fontsize=9.5, fontweight='bold', color=C['title'], pad=6)
    ax1.set_xlabel('t-SNE Dim 1', fontsize=8.5, color='#555')
    ax1.set_ylabel('t-SNE Dim 2', fontsize=8.5, color='#555')
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.9,
               edgecolor=C['grid'])
    ax1.text(0.03, 0.97,
             'Square → Star: paired user\nShort lines = semantic closeness\n'
             '(qualitative, CLIP domain effect limits separation)',
             transform=ax1.transAxes, fontsize=7.5, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6',
                       edgecolor='#F6AE2D', alpha=0.92))

    # ════════════════════════════════════════════════════════
    # Panel 2：Cross-Modal Alignment（修正版）
    # W_exp(text[i]) · W_imp(audio[i]) > 跨用戶
    # ════════════════════════════════════════════════════════
    draw_sim_panel(
        ax2,
        p2_paired, p2_random, p2_pval,
        title  = 'Panel 2 · Cross-Modal Alignment\n'
                 'W_exp(text) · W_imp(audio)  [256D]',
        xlabel = 'Cosine Similarity  (shared 256D space)',
        paired_label = 'Paired  (same user text × audio)',
        random_label = 'Random  (cross user text × audio)',
        C = C,
    )
    ax2.text(0.03, 0.03,
             'Validates: InfoNCE training correctly\naligns text & audio in 256D space\n'
             '→ Same user\'s text & music closer\n   than random cross-user pairs',
             transform=ax2.transAxes, fontsize=7.5, va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FFF4',
                       edgecolor='#27AE60', alpha=0.9))

    # ════════════════════════════════════════════════════════
    # Panel 3：Hit@K 檢索測試（新增，最強驗證）
    # P_ltp_implicit 作為 query，W_implicit(target_ast) 作為 pool
    # ════════════════════════════════════════════════════════
    ax3.set_facecolor(C['panel_bg'])
    ax3.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, axis='y', zorder=0)

    ks          = CFG.HITK_TOPK
    hit_vals    = [hitk_results[k] * 100 for k in ks]
    rand_vals   = [random_baselines[k] * 100 for k in ks]
    x           = np.arange(len(ks))
    bar_w       = 0.35

    bars_hit  = ax3.bar(x - bar_w/2, hit_vals,  bar_w,
                        color=C['paired_sim'], alpha=0.85,
                        label='P_ltp Retrieval', zorder=3)
    bars_rand = ax3.bar(x + bar_w/2, rand_vals, bar_w,
                        color=C['random_sim'],  alpha=0.55,
                        label='Random Baseline', zorder=3)

    # 數值標注
    for bar, val in zip(bars_hit, hit_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold', color=C['paired_sim'])
    for bar, val in zip(bars_rand, rand_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=7.5, color=C['random_sim'])

    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Hit@{k}' for k in ks], fontsize=9)
    ax3.set_ylabel('Hit Rate (%)', fontsize=9)
    ax3.set_title(
        'Panel 3 · Cross-Modal Hit@K Retrieval\n'
        f'Query: W_exp(text)  Pool: {pool_size:,} songs  MRR={mrr:.4f}',
        fontsize=9.5, fontweight='bold', color=C['title'], pad=6,
    )
    ax3.legend(loc='upper left', fontsize=8.5, framealpha=0.9,
               edgecolor=C['grid'])

    # 提升倍數標注（最大 K）
    max_k     = max(ks)
    max_hit   = hitk_results[max_k] * 100
    max_rand  = random_baselines[max_k] * 100
    lift      = max_hit / max_rand if max_rand > 0 else 0
    lift_color = C['paired_sim'] if lift >= 2.0 else '#E67E22'
    ax3.text(0.97, 0.97,
             f'Hit@{max_k} lift: ×{lift:.1f}\n'
             f'vs random baseline\n'
             f'MRR = {mrr:.4f}',
             transform=ax3.transAxes, fontsize=9, va='top', ha='right',
             fontweight='bold', color=lift_color,
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                       edgecolor=lift_color, alpha=0.95))

    ax3.text(0.03, 0.03,
             f'Query: W_explicit(raw_text)  [256D]\n'
             f'Pool:  W_implicit(target_ast) [256D]\n'
             f'Pool size: {pool_size:,} songs (full library)\n'
             f'Random Hit@{max(CFG.HITK_TOPK)} ≈ {random_baselines[max(CFG.HITK_TOPK)]*100:.3f}%',
             transform=ax3.transAxes, fontsize=7.5, va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB',
                       edgecolor='#2E86AB', alpha=0.9))

    # ════════════════════════════════════════════════════════
    # Panel 4：Audio Space t-SNE + Convex Hull + 錨定率
    # ════════════════════════════════════════════════════════
    ax4.set_facecolor(C['panel_bg'])
    ax4.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, zorder=0)

    for u in range(n_show):
        color  = usr_colors[u]
        c_mask = [i for i, lbl in enumerate(user_labels) if lbl == u]
        if not c_mask:
            continue
        c_pts = core_2d[c_mask]
        t_pt  = tgt_2d[u:u+1]
        draw_convex_hull(ax4, np.vstack([c_pts, t_pt]), color=color)
        ax4.scatter(c_pts[:,0], c_pts[:,1],
                    color=color, s=40, marker='o', alpha=0.72,
                    edgecolors='white', linewidths=0.4, zorder=3)
        edge_c = '#111111' if (u < len(contained) and contained[u]) else '#FF4444'
        ax4.scatter(t_pt[:,0], t_pt[:,1],
                    color=color, s=140, marker='D', alpha=0.95,
                    edgecolors=edge_c, linewidths=1.2, zorder=5)

    for u in range(n_show, N):
        c_mask = [i for i, lbl in enumerate(user_labels) if lbl == u]
        if c_mask:
            ax4.scatter(core_2d[c_mask,0], core_2d[c_mask,1],
                        color=C['other_core'], s=20, marker='o',
                        alpha=0.28, zorder=2)
        edge_c = '#111111' if (u < len(contained) and contained[u]) else '#FF4444'
        ax4.scatter(tgt_2d[u,0], tgt_2d[u,1],
                    color=C['other_target'], s=60, marker='D',
                    alpha=0.38, edgecolors=edge_c, linewidths=0.7, zorder=4)

    from matplotlib.lines import Line2D
    ax4.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#27AE60',
               markersize=8, label=f'Core Music (N={M})'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='#111', markersize=9, label='Target – inside hull  ✓'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='#FF4444', markersize=9, label='Target – outside hull  ✗'),
    ], loc='upper right', fontsize=7.5, framealpha=0.9, edgecolor=C['grid'])

    ax4.set_title('Panel 4 · Audio Space t-SNE\n'
                  '(AST 768D → t-SNE 2D, same color = same user)',
                  fontsize=9.5, fontweight='bold', color=C['title'], pad=6)
    ax4.set_xlabel('t-SNE Dim 1', fontsize=8.5, color='#555')
    ax4.set_ylabel('t-SNE Dim 2', fontsize=8.5, color='#555')

    a_color = '#27AE60' if contain_rate >= 0.4 else '#E67E22'
    ax4.text(0.03, 0.97,
             f'Containment Rate\n{contain_rate*100:.1f}%  ({n_contained}/{N})\n'
             f'Target inside Core hull',
             transform=ax4.transAxes, fontsize=9.5, va='top',
             fontweight='bold', color=a_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor=a_color, alpha=0.95))
    ax4.text(0.03, 0.03,
             f'Color hull = top {n_show}/{N} users\n'
             f'Black edge ✓ inside  |  Red edge ✗ outside',
             transform=ax4.transAxes, fontsize=7.5, va='bottom', color='#444',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8EE',
                       edgecolor='#2DC653', alpha=0.88))

    # ── 儲存 ─────────────────────────────────────────────────
    plt.tight_layout(pad=1.5)
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CFG.OUTPUT_FILE, dpi=150, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()
    logger.info(f"✅ 圖表儲存：{CFG.OUTPUT_FILE}")

    # ── 終端機摘要 ───────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"📊 Experiment 4  關鍵數字（N={N} 個用戶）")
    print(f"{'='*68}")

    print(f"\n  [Panel 2 · Cross-Modal Alignment]")
    print(f"  W_exp(text) · W_imp(audio) 配對 cosine sim : {p2_paired.mean():.4f}")
    print(f"  W_exp(text) · W_imp(audio) 隨機 cosine sim : {p2_random.mean():.4f}")
    print(f"  Δμ = {p2_paired.mean()-p2_random.mean():+.4f}   p = {p2_pval:.2e}  "
          f"{'✓ 顯著' if p2_pval < 0.05 else '✗ 不顯著'}")

    print(f"\n  [Panel 3 · Cross-Modal Hit@K 全庫檢索測試]  Pool = {pool_size:,} 首")
    print(f"  Query: W_explicit(raw_text)  Pool key: W_implicit(target_ast)")
    print(f"  {'K':<8} {'Hit%':>10} {'Random Hit%':>14} {'Lift':>8}")
    print(f"  {'-'*46}")
    for k in CFG.HITK_TOPK:
        h = hitk_results[k] * 100
        r = random_baselines[k] * 100
        l = h / r if r > 0 else 0
        print(f"  Hit@{k:<4} {h:>9.2f}% {r:>13.4f}% {l:>7.0f}×")
    rnd_mrr = sum(1/r for r in range(1, min(pool_size+1, 10001))) / pool_size
    print(f"  MRR = {mrr:.4f}   (random MRR ≈ {rnd_mrr:.4f})")

    print(f"\n  [用戶內一致性 · P_ltp explicit vs implicit]")
    print(f"  同用戶 intra cosine sim mean : {intra_sims.mean():.4f}")
    print(f"  跨用戶 inter cosine sim mean : {inter_sims.mean():.4f}")
    print(f"  Δμ = {intra_sims.mean()-inter_sims.mean():+.4f}   p = {intra_pval:.2e}  "
          f"{'✓ 顯著' if intra_pval < 0.05 else '✗ 不顯著'}")

    print(f"\n  [Panel 4 · Audio Space Behavioral Anchoring]")
    print(f"  錨定率 : {contain_rate*100:.1f}%  ({n_contained}/{N})")
    print(f"  Core music 平均 : {M/N:.1f} 首/用戶")
    print(f"{'='*68}\n")

# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=== 實驗四：雙空間 t-SNE 偏好對齊（v8）===")

    # 1. 初始化裝置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"裝置: {device}")

    # 2. 載入投影層權重
    proj = ProjectionWeights(CFG.PROJECTION_WEIGHTS, device='cpu')

    # 3. 載入 CLIP
    logger.info("載入 CLIP Text Encoder...")
    clip_tok   = CLIPTokenizer.from_pretrained(CFG.CLIP_MODEL)
    clip_model = CLIPTextModel.from_pretrained(CFG.CLIP_MODEL).to(device)
    clip_model.eval()
    logger.info("CLIP 載入完成")

    # 4. 建立索引
    profile_index = build_profile_index()
    if not profile_index:
        raise RuntimeError("Stage 4 profiles 為空")
    ast_index = build_ast_index()

    # 5. 全量載入 P_ltp（explicit + implicit）
    logger.info("全量載入 P_ltp explicit_only...")
    pltp_exp_map = load_pltp_all(CFG.PLTP_EXPLICIT)
    logger.info("全量載入 P_ltp implicit_only...")
    pltp_imp_map = load_pltp_all(CFG.PLTP_IMPLICIT)

    # 6. 建立全庫 Hit@K pool（84,150 首 × 256D ≈ 86MB）
    pool_ids, pool_proj = build_full_pool(pltp_imp_map, ast_index, proj)

    # 7. 載入當次取樣的向量
    results = load_all_data(
        profile_index, ast_index, CFG.N_USERS,
        clip_model, clip_tok, device,
        proj, pltp_exp_map, pltp_imp_map,
    )
    (raw_clip, llm_clip,
     raw_proj, tgt_proj,
     pltp_exp, pltp_imp,
     core_audio, tgt_audio,
     user_labels, matched_ids) = results

    # 8. 繪圖（含 Hit@K 全庫計算）
    plot_all(
        raw_clip, llm_clip,
        raw_proj, tgt_proj,
        pltp_exp, pltp_imp,
        core_audio, tgt_audio,
        user_labels,
        pool_ids, pool_proj,
        matched_ids,
    )

    print(f"✅ 輸出：{CFG.OUTPUT_FILE}")


if __name__ == "__main__":
    main()