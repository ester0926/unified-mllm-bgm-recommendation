"""
實驗三：隱性特徵的語義基礎化權重 (Semantic Grounding Weights)
============================================================
目標：用真實資料展示 Softmax 加權機制能有效「過濾雜訊」。
      公式：w_i = softmax(β × sim(Semantic_Seed(m_i), P_explicit))，β=2.0

自動選樣策略：
  掃描 Stage 2 history 目錄，找出 core_sbs 中 genre 最多樣的一首
  target music 作為展示範例（最能呈現權重差異）。

使用方式：
  cd "<repo_root>"
  python exp3_semantic_weights.py

輸出：
  visualization_outputs/exp3_semantic_grounding_weights.png
"""

import json
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ★ CONFIG
# ============================================================
class CFG:
    BASE_DIR    = Path(r"data/user_profiling")

    PROFILES    = BASE_DIR / "long_term_preference/stage4_recLLM/profiles.jsonl"
    HISTORY_DIR = BASE_DIR / "long_term_preference/stage2_history/personax"

    OUTPUT_DIR  = BASE_DIR / "experiments/visualization_outputs"
    OUTPUT_FILE = OUTPUT_DIR / "exp3_semantic_grounding_weights.png"

    CLIP_MODEL  = "openai/clip-vit-base-patch32"

    BETA        = 2.0   # Softmax 溫度參數（與 Stage 5 相同）

    # 自動選樣：若不想自動選，把 TARGET_MUSIC_ID 改為具體的 music_id 字串
    # 例如：TARGET_MUSIC_ID = "dQw4w9WgXcQ"
    TARGET_MUSIC_ID: Optional[str] = None   # None = 自動選最多樣的

    # 掃描候選數量上限（加速）
    MAX_SCAN_HISTORIES = 3000

    COLORS = {
        'match':     '#27AE60',   # 符合偏好
        'mismatch':  '#E74C3C',   # 不符合 / 雜訊
        'baseline':  '#95A5A6',   # uniform baseline
        'bg':        '#F8F9FA',
        'text':      '#2C3E50',
        'explicit':  '#2D6BE4',
    }


# ============================================================
# Step 1: 自動選樣——找最適合展示 Semantic Grounding 效果的樣本
# ============================================================

def _is_ascii_or_cjk(text: str) -> bool:
    """
    檢查字串是否只含 ASCII 可顯示字元 或 CJK 統一漢字。
    排除泰文、阿拉伯文、韓文、日文假名等需要額外字型的字元範圍。
    允許範圍：
      - ASCII 可顯示字元 (U+0020–U+007E)
      - CJK 統一漢字 (U+4E00–U+9FFF)
      - CJK 擴展 A (U+3400–U+4DBF)
      - 全形標點 (U+FF00–U+FFEF)
      - 常見標點符號 (U+2000–U+206F)
    """
    for ch in text:
        cp = ord(ch)
        if (0x0020 <= cp <= 0x007E or   # ASCII 可顯示
            0x4E00 <= cp <= 0x9FFF or   # CJK 統一漢字
            0x3400 <= cp <= 0x4DBF or   # CJK 擴展 A
            0xFF00 <= cp <= 0xFFEF or   # 全形字元
            0x2000 <= cp <= 0x206F or   # 一般標點
            ch in ' \t\n\r'):
            continue
        return False
    return True


def _score_sample(core_sbs: List[Dict]) -> float:
    """
    對一個 core_sbs 計算「適合展示 Semantic Grounding」的分數。
    理想樣本：有 1 個明顯少數 genre（雜訊）+ 其餘多數 genre 相同或相近。

    評分邏輯：
      - majority_ratio：多數 genre 佔 core_sbs 的比例（越高越好，代表偏好清晰）
      - outlier_gap   ：多數 genre 出現次數 − 少數 genre 出現次數（越大越好）
      - 最終分數 = majority_ratio × outlier_gap
    """
    from collections import Counter
    genres = [item.get('genre', 'unknown').lower().strip() for item in core_sbs]
    counts = Counter(genres)
    if len(counts) < 2:
        return 0.0   # 全部同 genre，沒有對比

    sorted_counts = sorted(counts.values(), reverse=True)
    majority_cnt = sorted_counts[0]
    minority_cnt = sorted_counts[-1]
    n = len(core_sbs)

    majority_ratio = majority_cnt / n
    outlier_gap    = majority_cnt - minority_cnt

    return majority_ratio * outlier_gap


def find_best_sample_music_id() -> str:
    """
    掃描 history 目錄，找最適合展示 Semantic Grounding 的樣本：

    篩選條件（必須全部滿足）：
      1. core_sbs 至少 5 首
      2. 每首都有非空 semantic_seed
      3. genre 至少 2 種（有對比才有意義）
      4. 所有 title 只含 ASCII 或 CJK 字元（避免泰文、阿拉伯文等字型問題）

    排序依據：
      majority_ratio × outlier_gap 最高者優先
      → 即「有明確多數 genre + 有明確少數 genre」的樣本
    """
    history_files = sorted(CFG.HISTORY_DIR.glob("*__history.json"))
    if not history_files:
        raise FileNotFoundError(f"找不到 history 檔案: {CFG.HISTORY_DIR}")

    logger.info(f"掃描 {min(len(history_files), CFG.MAX_SCAN_HISTORIES)} 個 history 檔案...")

    candidates = []   # list of (score, n_majority, mid, genre_summary)

    for path in tqdm(history_files[:CFG.MAX_SCAN_HISTORIES], desc="掃描 history"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                hist = json.load(f)

            core_sbs = hist.get('balanced_history', {}).get('core_sbs', [])
            if len(core_sbs) < 5:
                continue

            # 條件 2：每首都要有 semantic_seed
            if any(not item.get('semantic_seed', '').strip() for item in core_sbs):
                continue

            # 條件 2b：不允許任何一首 genre 為 unknown / 空字串
            if any(item.get('genre', 'unknown').lower().strip() in ('unknown', '', 'none')
                   for item in core_sbs):
                continue

            # 條件 3：至少 2 種 genre
            from collections import Counter
            genres = [item.get('genre', 'unknown').lower().strip() for item in core_sbs]
            genre_counts = Counter(genres)
            if len(genre_counts) < 2:
                continue

            # 條件 4：所有 title 只含 ASCII 或 CJK
            titles_ok = all(
                _is_ascii_or_cjk(item.get('title', ''))
                for item in core_sbs
            )
            if not titles_ok:
                continue

            score = _score_sample(core_sbs)
            if score <= 0:
                continue

            sorted_counts = sorted(genre_counts.values(), reverse=True)
            genre_summary = ", ".join(
                f"{g}×{c}" for g, c in genre_counts.most_common()
            )
            candidates.append((score, sorted_counts[0], hist.get('target_music'),
                                genre_summary, len(core_sbs)))

        except Exception:
            continue

    if not candidates:
        # fallback：放寬條件，只要有 core_sbs 且 ASCII/CJK 即可
        logger.warning("嚴格條件找不到樣本，放寬為僅過濾字元...")
        for path in history_files[:CFG.MAX_SCAN_HISTORIES]:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    hist = json.load(f)
                core_sbs = hist.get('balanced_history', {}).get('core_sbs', [])
                if not core_sbs:
                    continue
                if all(_is_ascii_or_cjk(item.get('title', '')) for item in core_sbs):
                    mid = hist.get('target_music')
                    logger.info(f"Fallback 選定: {mid}")
                    return mid
            except Exception:
                continue
        raise RuntimeError("找不到任何合適的樣本")

    # 依 score 降序排列，取最高分
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

    best = candidates[0]
    score, n_majority, best_mid, genre_summary, n_core = best

    logger.info(f"選定 target_music_id: {best_mid}")
    logger.info(f"  Genre 分布: {genre_summary}  (共 {n_core} 首)")
    logger.info(f"  Grounding 展示分數: {score:.3f}")
    logger.info(f"  前 5 名候選:")
    for i, (sc, nm, mid, gs, nc) in enumerate(candidates[:5]):
        logger.info(f"    [{i+1}] {mid}  score={sc:.3f}  genres={gs}")

    return best_mid


# ============================================================
# Step 2: 載入資料
# ============================================================

def load_profile(music_id: str) -> Optional[Dict]:
    """從 Stage 4 profiles.jsonl 找出此 music_id 的 profile"""
    import jsonlines
    with jsonlines.open(CFG.PROFILES, 'r') as reader:
        for obj in reader:
            mid = obj.get('music_id') or obj.get('target_music')
            if mid == music_id:
                return obj
    return None


def load_history(music_id: str) -> Optional[Dict]:
    path = CFG.HISTORY_DIR / f"{music_id}__history.json"
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_explicit_text(profile: Dict) -> str:
    """與 Stage 5 / 實驗二相同邏輯"""
    facts = profile.get('salient_facts', [])
    if facts:
        if isinstance(facts[0], dict):
            texts = [f.get('fact', '') for f in facts]
        else:
            texts = list(facts)
        text = ". ".join(t for t in texts if t)
        if text.strip():
            return text
    return profile.get('summary_text', '')


# ============================================================
# Step 3: CLIP-T 編碼 & 相似度計算
# ============================================================

def encode_single(
    text: str,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
) -> np.ndarray:
    """編碼單一文本 → (512,) float32"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        emb = model(**inputs).pooler_output.cpu().numpy().flatten()
    return emb


def encode_batch(
    texts: List[str],
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
) -> np.ndarray:
    """編碼文本列表 → (N, 512) float32"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        embs = model(**inputs).pooler_output.cpu().numpy()
    return embs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (512,), b: (N, 512) → (N,) cosine sim"""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


def softmax_beta(sims: np.ndarray, beta: float) -> np.ndarray:
    """w_i = softmax(β × sim_i)，數值穩定版"""
    x = beta * sims
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


# ============================================================
# Step 4: 繪圖
# ============================================================

def get_cjk_font():
    """
    偵測系統中可用的 CJK 字型，回傳 FontProperties 物件。
    優先順序：Windows 常見字型 → 跨平台字型 → fallback DejaVu Sans。
    同時把該字型設為 matplotlib 的全域 sans-serif，讓所有文字都能顯示。
    """
    from matplotlib import font_manager as fm
    from matplotlib.font_manager import FontProperties

    # Windows / macOS / Linux 上常見的支援 CJK 字型候選清單
    candidates = [
        'Microsoft YaHei',    # 微軟雅黑 (Windows)
        'Microsoft JhengHei', # 微軟正黑體 (Windows TW)
        'SimHei',             # 黑體 (Windows)
        'SimSun',             # 宋體 (Windows)
        'MingLiU',            # 細明體 (Windows TW)
        'PingFang TC',        # macOS 繁中
        'PingFang SC',        # macOS 簡中
        'Hiragino Sans',      # macOS 日文
        'Noto Sans CJK TC',   # Linux / Google Fonts 繁中
        'Noto Sans CJK SC',   # Linux / Google Fonts 簡中
        'Noto Sans CJK JP',   # Linux / Google Fonts 日文
        'WenQuanYi Zen Hei',  # Linux 文泉驛
        'WenQuanYi Micro Hei',
        'IPAGothic',          # Linux IPA
        'IPAPGothic',
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for name in candidates:
        if name in available:
            # 把它加入 sans-serif fallback，讓全局文字也能 fallback 到它
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            logger.info(f"CJK 字型選用: {name}")
            return FontProperties(family=name)

    logger.warning("找不到 CJK 字型，CJK 字元可能顯示為方塊。"
                   "建議安裝 Noto Sans CJK 或 Microsoft YaHei。")
    return FontProperties(family='DejaVu Sans')


def plot_weights(
    core_sbs:      List[Dict],
    sims:          np.ndarray,
    weights:       np.ndarray,
    weights_no_beta: np.ndarray,
    explicit_text: str,
    music_id:      str,
    save_path:     Path,
):
    """
    三欄布局：
    - 主圖 (左大)：w_i 長條圖
    - 輔圖 (右上)：原始 CLIP cosine similarity 橫條
    - 底圖 (全寬)：β=0 vs β=2.0 消融對比
    """
    # ── 字型設定：偵測 CJK 字型供含中日韓字元的標籤使用 ──────────────
    cjk_font = get_cjk_font()
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.unicode_minus': False,
    })
    C = CFG.COLORS
    n = len(core_sbs)

    # --- 組裝標籤 ---
    # 注意：labels 需要在 weights 計算完後才能確定顏色，
    # 先建立標籤，顏色在繪圖前依 weights vs uniform 決定。
    labels = []
    genre_set = {}

    for item in core_sbs:
        title = item.get('title', 'Unknown')[:20]
        genre = item.get('genre', 'unknown').lower().strip()
        genre_set[genre] = genre_set.get(genre, 0) + 1
        labels.append(f"{title}\n({genre})")

    # ── 顏色依「是否低於 uniform baseline」決定 ──────────────────────
    # 高於 baseline（語義較契合，被強化）→ 綠色
    # 低於 baseline（語義較偏離，被壓制）→ 紅色
    # 這樣顏色與 Semantic Grounding 計算結果完全一致，
    # 不依賴 genre 標籤頻率，避免「標籤少數 ≠ 語義雜訊」的矛盾。
    
    uniform = 1.0 / n

    bar_colors = [
        C['mismatch'] if w < uniform else C['match']
        for w in weights
    ]

    # 輔圖 y 軸標籤的 [!]/[ok] 同樣改為依權重判斷
    def tag_by_weight(i):
        return '[!]' if weights[i] < uniform else '[ok]'

    x = np.arange(n)

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(C['bg'])
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[3, 1.2],
                  width_ratios=[3, 1.5],
                  hspace=0.48, wspace=0.32)

    # =============== 主圖：w_i 長條 ===============
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_facecolor(C['bg'])

    bars = ax_main.bar(x, weights * 100, color=bar_colors, width=0.58,
                       edgecolor='white', linewidth=0.8, zorder=3)

    # Uniform baseline
    ax_main.axhline(uniform * 100, color=C['baseline'], linestyle='--',
                    linewidth=1.8, zorder=2,
                    label=f'Uniform Baseline ({uniform*100:.1f}%)')

    # 柱頂數值
    for i, (bar, w) in enumerate(zip(bars, weights)):
        is_noise = bar_colors[i] == C['mismatch']
        ax_main.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2,
                     f'{w*100:.1f}%',
                     ha='center', va='bottom', fontsize=9,
                     fontweight='bold',
                     color=C['mismatch'] if is_noise else C['text'])

    # 標注雜訊柱（最低 weight 的那個）
    noise_idx = int(np.argmin(weights))
    noise_bar = bars[noise_idx]
    suppress_ratio = weights[noise_idx] / uniform

    ax_main.annotate(
        f'  Noise Suppressed\n'
        f'  w = {weights[noise_idx]*100:.2f}%\n'
        f'  ({suppress_ratio:.2f}× baseline)',
        xy=(noise_bar.get_x() + noise_bar.get_width() / 2,
            noise_bar.get_height()),
        xytext=(noise_bar.get_x() - 0.8, max(weights) * 100 * 0.72),
        fontsize=9.5, color=C['mismatch'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C['mismatch'],
                        connectionstyle='arc3,rad=-0.3', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=C['mismatch'], alpha=0.9)
    )

    ax_main.set_xticks(x)
    # 逐一設定 tick label 以套用 CJK 字型
    ax_main.set_xticklabels(labels, fontsize=8.5)
    for tick in ax_main.get_xticklabels():
        tick.set_fontproperties(cjk_font)
        tick.set_fontsize(8.5)
    ax_main.set_ylabel('Semantic Grounding Weight (%)', fontsize=11)
    ax_main.set_title(
        'Experiment 3: Semantic Grounding Weights\n'
        r'$w_i = \mathrm{softmax}(\beta \cdot \mathrm{sim}(\mathrm{Semantic\_Seed}(m_i),\ P_{\mathrm{explicit}}))$'
        f',  β={CFG.BETA}',
        fontsize=12, fontweight='bold', color=C['text'], pad=12
    )
    ax_main.set_ylim(0, max(weights) * 100 * 1.35)
    ax_main.grid(axis='y', color='#E0E0E0', linewidth=0.8, zorder=0)
    ax_main.tick_params(axis='x', length=0)
    ax_main.legend(fontsize=9.5, framealpha=0.85, edgecolor='#CCC')

    # 圖例 patches
    match_patch    = mpatches.Patch(color=C['match'],    label='Semantically Aligned  (w > baseline)')
    mismatch_patch = mpatches.Patch(color=C['mismatch'], label='Semantically Suppressed  (w < baseline)')
    ax_main.legend(handles=[match_patch, mismatch_patch,
                             plt.Line2D([],[],color=C['baseline'],linestyle='--',
                                        linewidth=1.8, label=f'Uniform Baseline ({uniform*100:.1f}%)')],
                   fontsize=9, framealpha=0.85, edgecolor='#CCC', loc='upper right')

    # =============== 輔圖：Cosine Similarity ===============
    ax_sim = fig.add_subplot(gs[0, 1])
    ax_sim.set_facecolor(C['bg'])

    h_colors = [C['mismatch'] if bar_colors[i] == C['mismatch'] else C['explicit']
                for i in range(n)]
    ax_sim.barh(x, sims, color=h_colors, height=0.55,
                edgecolor='white', linewidth=0.8, zorder=3)

    short_labels = []
    for i, item in enumerate(core_sbs):
        genre = item.get('genre', 'unknown').lower().strip()
        tag   = tag_by_weight(i)
        short_labels.append(f"{tag} {item.get('title','')[:16]}")

    ax_sim.set_yticks(x)
    ax_sim.set_yticklabels(short_labels, fontsize=8.5)
    for tick in ax_sim.get_yticklabels():
        tick.set_fontproperties(cjk_font)
        tick.set_fontsize(8.5)
    ax_sim.set_xlabel('CLIP Cosine Similarity\nwith Explicit Profile', fontsize=9.5)
    ax_sim.set_title('Input\nSimilarity Scores', fontsize=10.5, fontweight='bold')
    ax_sim.set_xlim(-0.1, 1.05)
    ax_sim.axvline(0, color='#BBB', linewidth=0.8)
    ax_sim.grid(axis='x', color='#E0E0E0', linewidth=0.8, zorder=0)

    for i, s in enumerate(sims):
        ax_sim.text(max(s, 0) + 0.02, i, f'{s:.3f}',
                    va='center', fontsize=8.5,
                    color=C['mismatch'] if bar_colors[i] == C['mismatch'] else C['text'])

    # =============== 底圖：β=0 vs β=2.0 消融對比 ===============
    ax_abl = fig.add_subplot(gs[1, :])
    ax_abl.set_facecolor(C['bg'])

    w_ab = 0.35
    bars_no = ax_abl.bar(x - w_ab / 2, weights_no_beta * 100,
                          width=w_ab, color='#BDC3C7',
                          label='β=0 (Uniform, No Grounding)', zorder=3,
                          edgecolor='white', linewidth=0.6)
    bars_w  = ax_abl.bar(x + w_ab / 2, weights * 100,
                          width=w_ab, color=bar_colors,
                          label=f'β={CFG.BETA} (Semantic Grounding)', zorder=3,
                          edgecolor='white', linewidth=0.6)

    ax_abl.set_xticks(x)
    short_abl = []
    for i, item in enumerate(core_sbs):
        genre = item.get('genre', 'unknown').lower().strip()
        tag   = '[Suppressed]' if weights[i] < uniform else f'#{i+1}'
        short_abl.append(tag)
    ax_abl.set_xticklabels(short_abl, fontsize=9)
    ax_abl.set_ylabel('Weight (%)', fontsize=10)
    ax_abl.set_title(
        f'Ablation: Effect of β-Scaled Softmax  '
        f'(β=0 = uniform weights, β={CFG.BETA} = semantic grounding)',
        fontsize=10.5, fontweight='bold', color=C['text']
    )
    ax_abl.set_ylim(0, max(max(weights), max(weights_no_beta)) * 100 * 1.35)
    ax_abl.grid(axis='y', color='#E0E0E0', linewidth=0.8, zorder=0)
    ax_abl.tick_params(axis='x', length=0)
    ax_abl.legend(fontsize=9.5, framealpha=0.85, edgecolor='#CCC')

    # 標注雜訊柱的差異
    delta = (weights_no_beta[noise_idx] - weights[noise_idx]) * 100
    ax_abl.annotate(
        f'Δ = −{delta:.1f}% pts\n(noise suppressed)',
        xy=(noise_idx + w_ab / 2, weights[noise_idx] * 100),
        xytext=(noise_idx - 1.0, max(max(weights), max(weights_no_beta)) * 100 * 1.1),
        fontsize=8.5, color=C['mismatch'],
        arrowprops=dict(arrowstyle='->', color=C['mismatch'],
                        connectionstyle='arc3,rad=0.2', lw=1.3),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=C['mismatch'], alpha=0.9)
    )

    # 在圖底部加上 music_id 資訊
    fig.text(0.5, 0.01,
             f"Target Music ID: {music_id}  |  "
             f"Explicit Profile: \"{explicit_text[:80]}...\"",
             ha='center', fontsize=8, color='#999999',
             style='italic', fontproperties=cjk_font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    logger.info(f"✅ 圖表儲存: {save_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用裝置: {device}")

    # 1. 選定 music_id
    if CFG.TARGET_MUSIC_ID:
        music_id = CFG.TARGET_MUSIC_ID
        logger.info(f"使用指定 music_id: {music_id}")
    else:
        music_id = find_best_sample_music_id()

    # 2. 載入資料
    logger.info(f"載入 Stage 4 profile: {music_id}")
    profile = load_profile(music_id)
    if profile is None:
        raise RuntimeError(f"Stage 4 profiles.jsonl 中找不到 music_id={music_id}")

    logger.info(f"載入 Stage 2 history: {music_id}")
    history = load_history(music_id)
    if history is None:
        raise RuntimeError(f"找不到 history: {CFG.HISTORY_DIR / (music_id + '__history.json')}")

    core_sbs = history.get('balanced_history', {}).get('core_sbs', [])
    if not core_sbs:
        raise RuntimeError(f"core_sbs 為空: {music_id}")

    logger.info(f"core_sbs 首數: {len(core_sbs)}")
    for i, item in enumerate(core_sbs):
        logger.info(f"  [{i}] {item.get('title','?')} / {item.get('genre','?')} — "
                    f"seed: {item.get('semantic_seed','')[:60]}...")

    # 3. 建立 explicit text
    explicit_text = build_explicit_text(profile)
    logger.info(f"Explicit text: {explicit_text[:100]}...")

    # 4. 載入 CLIP
    logger.info("載入 CLIP Text Encoder...")
    tokenizer  = CLIPTokenizer.from_pretrained(CFG.CLIP_MODEL)
    clip_model = CLIPTextModel.from_pretrained(CFG.CLIP_MODEL).to(device)

    # 5. 編碼
    logger.info("CLIP-T 編碼 explicit profile 與 semantic seeds...")
    explicit_emb = encode_single(explicit_text, tokenizer, clip_model, device)

    semantic_seeds = [item.get('semantic_seed', '') for item in core_sbs]
    seeds_emb = encode_batch(semantic_seeds, tokenizer, clip_model, device)  # (K, 512)

    # 6. 計算相似度與權重
    sims             = cosine_similarity(explicit_emb, seeds_emb)         # (K,)
    weights          = softmax_beta(sims, beta=CFG.BETA)                  # (K,)
    weights_no_beta  = softmax_beta(sims, beta=0.0)                       # (K,) = uniform

    # 7. 輸出統計
    logger.info("=== 統計摘要 ===")
    logger.info(f"β = {CFG.BETA}")
    logger.info(f"Uniform baseline: {1/len(core_sbs)*100:.2f}%")
    for i, item in enumerate(core_sbs):
        logger.info(f"  [{i}] {item.get('title','?')[:20]:20s} | "
                    f"sim={sims[i]:.4f} | w={weights[i]*100:.2f}%")

    noise_idx = int(np.argmin(weights))
    logger.info(f"\n最低權重 (雜訊候選): [{noise_idx}] {core_sbs[noise_idx].get('title','?')} "
                f"— w={weights[noise_idx]*100:.3f}% "
                f"({weights[noise_idx]/(1/len(core_sbs)):.2f}× baseline)")

    # 8. 繪圖
    plot_weights(
        core_sbs=core_sbs,
        sims=sims,
        weights=weights,
        weights_no_beta=weights_no_beta,
        explicit_text=explicit_text,
        music_id=music_id,
        save_path=CFG.OUTPUT_FILE,
    )

    print(f"\n✅ 輸出: {CFG.OUTPUT_FILE}")
    print(f"\n📊 關鍵數字 (可直接引用):")
    print(f"   β = {CFG.BETA}")
    print(f"   Uniform baseline: {1/len(core_sbs)*100:.2f}%")
    print(f"   最高權重音樂: {core_sbs[int(np.argmax(weights))].get('title','?')} "
          f"— {max(weights)*100:.2f}%")
    print(f"   最低權重音樂: {core_sbs[noise_idx].get('title','?')} "
          f"— {weights[noise_idx]*100:.2f}% "
          f"({weights[noise_idx]/(1/len(core_sbs)):.2f}× baseline)")


if __name__ == "__main__":
    main()