"""
用途：分析音訊特徵變化與偏好表示的關係。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from scipy.stats import mannwhitneyu
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
    BASE_DIR    = Path(r"data/user_profiling")
    HISTORY_DIR = BASE_DIR / "long_term_preference/stage2_history/personax"
    PROFILES    = BASE_DIR / "long_term_preference/stage4_recLLM/profiles.jsonl"
    HDF5_DIR    = Path(r"data/optimized_musechat_features_float16_v3")

    OUTPUT_DIR  = BASE_DIR / "experiments/visualization_outputs"
    OUTPUT_FILE = OUTPUT_DIR / "exp1_audio_gradient.png"

    # 取樣數量（設 None 表示全部）
    N_USERS     = 200

    # 每類音樂最多取幾首參與計算（避免某類數量太多主導結果）
    MAX_PER_CAT = 5

    AST_DIM     = 768

    COLORS = {
        'core':        '#27AE60',
        'exploratory': '#F39C12',
        'negative':    '#E74C3C',
        'target':      '#2D6BE4',
        'bg':          '#F8F9FA',
        'panel_bg':    '#FFFFFF',
        'grid':        '#E0E0E0',
        'title':       '#1A1A2E',
    }


# ============================================================
# 工具函數
# ============================================================

def get_cjk_font() -> FontProperties:
    from matplotlib import font_manager as fm
    candidates = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
                  'PingFang TC', 'Noto Sans CJK TC']
    available  = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return FontProperties(family=name)
    return FontProperties(family='DejaVu Sans')


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """兩個向量的 cosine similarity"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_cosine_sim(vecs: List[np.ndarray], target: np.ndarray) -> Optional[float]:
    """一組向量與 target 的平均 cosine similarity"""
    if not vecs:
        return None
    sims = [cosine_sim(v, target) for v in vecs]
    return float(np.mean(sims))


def sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ============================================================
# Step 1: 載入 Stage 2 history（三類 music_id）
# ============================================================

def load_history(music_id: str) -> Optional[Dict]:
    path = CFG.HISTORY_DIR / f"{music_id}__history.json"
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def collect_user_records(n_users: Optional[int]) -> Tuple[List[Dict], Set[str]]:
    """
    從 profiles.jsonl 取 music_id，再讀 Stage 2 history，
    收集每個用戶的 core / exploratory / negative / target music_id。
    回傳 records list 和所有需要 AST 向量的 music_id set。
    """
    import jsonlines
    if not CFG.PROFILES.exists():
        raise FileNotFoundError(f"找不到 profiles.jsonl: {CFG.PROFILES}")

    music_ids = []
    with jsonlines.open(CFG.PROFILES, 'r') as reader:
        for obj in reader:
            mid = obj.get('music_id') or obj.get('target_music', '')
            if mid:
                music_ids.append(mid)

    if n_users:
        music_ids = music_ids[:n_users * 3]   # 多取一些備用

    logger.info(f"profiles 候選：{len(music_ids)} 筆，開始載入 Stage 2 history...")

    records:     List[Dict] = []
    needed_ids:  Set[str]   = set()

    for mid in tqdm(music_ids, desc="載入 history"):
        if n_users and len(records) >= n_users:
            break
        hist = load_history(mid)
        if hist is None:
            continue
        bh            = hist.get('balanced_history', {})
        core_ids      = [x['music_id'] for x in bh.get('core_sbs', [])
                         if x.get('music_id')][:CFG.MAX_PER_CAT]
        explore_ids   = [x['music_id'] for x in bh.get('exploratory_sbs', [])
                         if x.get('music_id')][:CFG.MAX_PER_CAT]
        negative_ids  = [x['music_id'] for x in bh.get('negative_sbs', [])
                         if x.get('music_id')][:CFG.MAX_PER_CAT]

        # 三類都要有資料才有效
        if not core_ids or not explore_ids or not negative_ids:
            continue

        records.append({
            'target_id':    mid,
            'core_ids':     core_ids,
            'explore_ids':  explore_ids,
            'negative_ids': negative_ids,
        })
        needed_ids.update(core_ids + explore_ids + negative_ids)
        needed_ids.add(mid)   # target music 自己也需要

    logger.info(f"有效用戶：{len(records)}，需要 AST 向量：{len(needed_ids)} 筆")
    return records, needed_ids


# ============================================================
# Step 2: 建立 AST 索引（v3 HDF5 格式）
# ============================================================

def build_ast_index(needed_ids: Set[str]) -> Dict[str, np.ndarray]:
    """
    掃描 HDF5（v3 格式：pairs/{key}/target_music_all_cls [12,768]），
    以 prefix / suffix / full key 三種方式儲存，確保命中率最高。
    只儲存 needed_ids 中的 music_id 以節省記憶體。
    """
    h5_files = sorted(CFG.HDF5_DIR.glob("musechat_features_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"找不到 HDF5：{CFG.HDF5_DIR}")
    logger.info(f"掃描 {len(h5_files)} 個 HDF5...")

    index: Dict[str, np.ndarray] = {}

    for fpath in tqdm(h5_files, desc="建立 AST 索引"):
        if len(index) >= len(needed_ids) * 1.2:
            break
        try:
            with h5py.File(fpath, 'r') as f:
                grp = f.get('pairs', f)
                for key in grp.keys():
                    try:
                        sub = grp[key]
                        if 'target_music_all_cls' not in sub:
                            continue
                        raw = sub['target_music_all_cls'][()]  # [12, 768]
                        vec = raw.astype(np.float32).mean(axis=0)  # [768]

                        parts = key.rsplit('_', 1)
                        candidates = [key]
                        if len(parts) == 2:
                            candidates += [parts[0], parts[1]]
                        for c in candidates:
                            if c in needed_ids and c not in index:
                                index[c] = vec
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"無法讀取 {fpath.name}: {e}")

    logger.info(f"AST 索引：{len(index)} 筆 / 需要 {len(needed_ids)} 筆 "
                f"（命中率 {len(index)/max(len(needed_ids),1)*100:.1f}%）")
    return index


# ============================================================
# Step 3: 計算三類音樂的 cosine similarity vs Target
# ============================================================

def compute_similarities(
    records:   List[Dict],
    ast_index: Dict[str, np.ndarray],
) -> Dict[str, List[float]]:
    """
    對每個有效用戶計算：
      core_sims    : per-music cosine sim（core music vs target music）
      explore_sims : per-music cosine sim（exploratory music vs target music）
      negative_sims: per-music cosine sim（negative music vs target music）

    同時計算每個用戶的均值，供 Panel 2/3 使用。

    回傳：
      {
        'core_all':     [所有 core music 的 cosine sim，跨用戶]
        'explore_all':  [所有 exploratory music 的 cosine sim，跨用戶]
        'negative_all': [所有 negative music 的 cosine sim，跨用戶]
        'core_mean':    [每個用戶 core music 的平均 cosine sim]
        'explore_mean': [每個用戶 explore music 的平均 cosine sim]
        'negative_mean':[每個用戶 negative music 的平均 cosine sim]
        'gradient_hold':[True/False，每個用戶是否滿足 core > explore > negative]
        'n_valid':      有效用戶數（target + 三類都有 AST 向量）
      }
    """
    core_all,    explore_all,    negative_all    = [], [], []
    core_mean,   explore_mean,   negative_mean   = [], [], []
    gradient_hold = []
    n_valid = 0

    for rec in tqdm(records, desc="計算 cosine similarity"):
        tgt_id = rec['target_id']
        if tgt_id not in ast_index:
            continue
        tgt_vec = ast_index[tgt_id]

        core_vecs  = [ast_index[m] for m in rec['core_ids']    if m in ast_index]
        exp_vecs   = [ast_index[m] for m in rec['explore_ids'] if m in ast_index]
        neg_vecs   = [ast_index[m] for m in rec['negative_ids']if m in ast_index]

        if not core_vecs or not exp_vecs or not neg_vecs:
            continue

        # per-music sims（用於 Panel 1 violin）
        c_sims = [cosine_sim(v, tgt_vec) for v in core_vecs]
        e_sims = [cosine_sim(v, tgt_vec) for v in exp_vecs]
        n_sims = [cosine_sim(v, tgt_vec) for v in neg_vecs]

        core_all.extend(c_sims)
        explore_all.extend(e_sims)
        negative_all.extend(n_sims)

        # per-user means（用於 Panel 2/3）
        cm = float(np.mean(c_sims))
        em = float(np.mean(e_sims))
        nm = float(np.mean(n_sims))
        core_mean.append(cm)
        explore_mean.append(em)
        negative_mean.append(nm)

        # 梯度是否成立：core > explore > negative
        gradient_hold.append(cm > em > nm)
        n_valid += 1

    logger.info(f"有效用戶（三類都有 AST）：{n_valid}")
    return {
        'core_all':     core_all,
        'explore_all':  explore_all,
        'negative_all': negative_all,
        'core_mean':    core_mean,
        'explore_mean': explore_mean,
        'negative_mean': negative_mean,
        'gradient_hold': gradient_hold,
        'n_valid': n_valid,
    }


# ============================================================
# Step 4: 繪圖（1 × 3）
# ============================================================

def plot_all(sims: Dict):
    C = CFG.COLORS
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False})
    get_cjk_font()

    core_all    = np.array(sims['core_all'])
    exp_all     = np.array(sims['explore_all'])
    neg_all     = np.array(sims['negative_all'])
    core_mean   = np.array(sims['core_mean'])
    exp_mean    = np.array(sims['explore_mean'])
    neg_mean    = np.array(sims['negative_mean'])
    grad_hold   = sims['gradient_hold']
    N           = sims['n_valid']

    # 統計檢定
    _, p_ce = mannwhitneyu(core_all, exp_all,  alternative='greater')
    _, p_en = mannwhitneyu(exp_all,  neg_all,  alternative='greater')
    _, p_cn = mannwhitneyu(core_all, neg_all,  alternative='greater')

    grad_rate = sum(grad_hold) / len(grad_hold) if grad_hold else 0.0

    # 統計分類（per-user ordering）
    orders = {'core>exp>neg': 0, 'core>neg>exp': 0,
              'exp>core>neg': 0, 'exp>neg>core': 0,
              'neg>core>exp': 0, 'neg>exp>core': 0}
    for cm, em, nm in zip(core_mean, exp_mean, neg_mean):
        ranking = sorted([('Core', cm), ('Exp', em), ('Neg', nm)],
                          key=lambda x: -x[1])
        key = '>'.join(r[0] for r in ranking).lower().replace('exp', 'exp').replace('neg', 'neg')
        # 對應到標準欄位名稱
        simple = f"{ranking[0][0].lower()}>{ranking[1][0].lower()}>{ranking[2][0].lower()}"
        for k in orders:
            if k.replace('core','Core').replace('exp','Exp').replace('neg','Neg') \
                == f"{ranking[0][0]}>{ranking[1][0]}>{ranking[2][0]}":
                orders[k] += 1
                break

    # 重新計算 orders 更直接
    orders = {'core>exp>neg': 0, 'core>neg>exp': 0,
              'exp>core>neg': 0, 'exp>neg>core': 0,
              'neg>core>exp': 0, 'neg>exp>core': 0}
    for cm, em, nm in zip(core_mean, exp_mean, neg_mean):
        vals = [('core', cm), ('exp', em), ('neg', nm)]
        ranked = sorted(vals, key=lambda x: -x[1])
        key = f"{ranked[0][0]}>{ranked[1][0]}>{ranked[2][0]}"
        if key in orders:
            orders[key] += 1

    # ── 佈局 ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7), facecolor=C['bg'])
    gs  = fig.add_gridspec(1, 3, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle(
        f'Experiment 1  ·  Stage 1 Audio Semantic Gradient Validation\n'
        f'Cosine Similarity vs Target Music in AST 768D Space  (N={N} users)',
        fontsize=12, fontweight='bold', color=C['title'], y=1.03,
    )

    # ════════════════════════════════════════════════════════
    # Panel 1：Violin + Boxplot（三類分布）
    # ════════════════════════════════════════════════════════
    ax1.set_facecolor(C['panel_bg'])
    ax1.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, axis='y', zorder=0)

    data       = [core_all, exp_all, neg_all]
    labels     = ['Core Music', 'Exploratory Music', 'Negative Music']
    colors     = [C['core'], C['exploratory'], C['negative']]
    positions  = [1, 2, 3]

    # Violin
    vp = ax1.violinplot(data, positions=positions, widths=0.65,
                        showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(vp['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)

    # Box（重疊在 violin 上）
    bp = ax1.boxplot(data, positions=positions, widths=0.28,
                     patch_artist=True, notch=False,
                     medianprops=dict(color='white', linewidth=2.2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=2.5,
                                     alpha=0.3, linestyle='none'))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker, cap in zip(bp['whiskers'], bp['caps']):
        whisker.set_color('#666666')
        cap.set_color('#666666')

    # 均值標注
    for pos, arr, color in zip(positions, data, colors):
        mu = np.mean(arr)
        ax1.plot(pos, mu, 'D', color='white', markersize=7, zorder=6)
        ax1.plot(pos, mu, 'D', color=color,  markersize=5, zorder=7)
        ax1.text(pos, mu + 0.005, f'μ={mu:.4f}',
                 ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold', color=color)

    # 顯著性標注
    y_sig = max(np.max(core_all), np.max(exp_all), np.max(neg_all)) * 1.02
    step  = (np.max(core_all) - np.min(neg_all)) * 0.08

    def draw_sig_bar(ax, x1, x2, y, p):
        stars = sig_stars(p)
        color = '#27AE60' if p < 0.05 else '#999999'
        ax.plot([x1, x1, x2, x2], [y, y+step*0.3, y+step*0.3, y],
                color=color, linewidth=1.2)
        ax.text((x1+x2)/2, y+step*0.35,
                f'{stars}  p={p:.2e}', ha='center', va='bottom',
                fontsize=8, color=color, fontweight='bold')

    draw_sig_bar(ax1, 1, 2, y_sig,          p_ce)
    draw_sig_bar(ax1, 2, 3, y_sig,          p_en)
    draw_sig_bar(ax1, 1, 3, y_sig + step*1.1, p_cn)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel('Cosine Similarity vs Target Music', fontsize=9.5)
    ax1.set_title('Panel 1 · Cosine Similarity Distribution\n'
                  '(per-music, all users pooled)',
                  fontsize=10, fontweight='bold', color=C['title'], pad=6)

    n_strs = [f'n={len(d)}' for d in data]
    ax1.text(0.97, 0.03,
             '\n'.join(f'{l}: {n}' for l, n in zip(labels, n_strs)),
             transform=ax1.transAxes, fontsize=8, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=C['grid'], alpha=0.9))

    # ════════════════════════════════════════════════════════
    # Panel 2：Stacked Bar（per-user 排列比例）
    # ════════════════════════════════════════════════════════
    ax2.set_facecolor(C['panel_bg'])

    order_labels = ['core>exp>neg', 'core>neg>exp',
                    'exp>core>neg', 'exp>neg>core',
                    'neg>core>exp', 'neg>exp>core']
    order_colors = ['#1A7F4B', '#5DBD87',
                    '#D4860A', '#F5C06A',
                    '#A01010', '#E87070']
    order_display = ['Core>Exp>Neg\n(expected ✓)',
                     'Core>Neg>Exp',
                     'Exp>Core>Neg',
                     'Exp>Neg>Core',
                     'Neg>Core>Exp',
                     'Neg>Exp>Core']
    counts  = [orders[k] for k in order_labels]
    percents= [c / N * 100 for c in counts]

    bars = ax2.bar(range(len(order_labels)), percents,
                   color=order_colors, alpha=0.85, edgecolor='white',
                   linewidth=0.8, zorder=3)

    for bar, pct, cnt in zip(bars, percents, counts):
        if pct > 2:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{pct:.1f}%\n(n={cnt})',
                     ha='center', va='bottom', fontsize=8.5,
                     fontweight='bold' if order_labels[bars.index(bar)] == 'core>exp>neg' else 'normal',
                     color=order_colors[bars.index(bar)])

    # 預期排列特別標注
    bars[0].set_edgecolor('#000000')
    bars[0].set_linewidth(2.0)

    ax2.set_xticks(range(len(order_labels)))
    ax2.set_xticklabels(order_display, fontsize=8, rotation=20, ha='right')
    ax2.set_ylabel('Percentage of Users (%)', fontsize=9.5)
    ax2.set_title('Panel 2 · Per-User Gradient Ordering\n'
                  f'Expected order (Core>Exp>Neg): {grad_rate*100:.1f}% of users',
                  fontsize=10, fontweight='bold', color=C['title'], pad=6)
    ax2.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, axis='y', zorder=0)

    # 梯度一致率大字標注
    g_color = '#1A7F4B' if grad_rate >= 0.4 else '#D4860A'
    ax2.text(0.97, 0.97,
             f'Gradient Consistency\n{grad_rate*100:.1f}%  ({sum(grad_hold)}/{N})\n'
             f'users: Core > Exp > Neg',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             fontweight='bold', color=g_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor=g_color, alpha=0.95))

    # ════════════════════════════════════════════════════════
    # Panel 3：Per-User 連線散點圖（三類均值）
    # ════════════════════════════════════════════════════════
    ax3.set_facecolor(C['panel_bg'])
    ax3.grid(True, color=C['grid'], linewidth=0.5, alpha=0.6, zorder=0)

    # 只顯示前 N_SHOW 個用戶以免太擁擠
    N_SHOW = min(60, N)
    alpha_line = max(0.08, 0.5 - N_SHOW * 0.006)

    for i in range(N_SHOW):
        cm, em, nm = core_mean[i], exp_mean[i], neg_mean[i]
        color_line = '#1A7F4B' if cm > em > nm else '#BBBBBB'
        ax3.plot([1, 2, 3], [cm, em, nm],
                 color=color_line, linewidth=0.8,
                 alpha=alpha_line, zorder=2)

    # 均值 ± std 大點
    for pos, arr, color, label in [
        (1, core_mean,  C['core'],        'Core'),
        (2, exp_mean,   C['exploratory'], 'Exploratory'),
        (3, neg_mean,   C['negative'],    'Negative'),
    ]:
        mu  = np.mean(arr)
        std = np.std(arr)
        ax3.errorbar(pos, mu, yerr=std, fmt='o',
                     color=color, markersize=12, linewidth=2.2,
                     capsize=6, capthick=2, zorder=5,
                     label=f'{label}  μ={mu:.4f}±{std:.4f}')

    # 均值連線
    mu_vals = [np.mean(core_mean), np.mean(exp_mean), np.mean(neg_mean)]
    ax3.plot([1, 2, 3], mu_vals, 'k--', linewidth=1.8, alpha=0.7,
             zorder=6, label='Population mean')

    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['Core\nMusic', 'Exploratory\nMusic', 'Negative\nMusic'],
                        fontsize=9.5)
    ax3.set_ylabel('Mean Cosine Similarity vs Target Music', fontsize=9.5)
    ax3.set_title(f'Panel 3 · Per-User Mean Cosine Similarity\n'
                  f'(showing {N_SHOW}/{N} users, green lines = expected gradient)',
                  fontsize=10, fontweight='bold', color=C['title'], pad=6)
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9,
               edgecolor=C['grid'])

    # ── 儲存 ─────────────────────────────────────────────────
    plt.tight_layout(pad=1.5)
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CFG.OUTPUT_FILE, dpi=150, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()
    logger.info(f"✅ 圖表儲存：{CFG.OUTPUT_FILE}")

    # ── 終端機摘要 ───────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"📊 Experiment 1  關鍵數字（N={N} 個有效用戶）")
    print(f"{'='*68}")

    print(f"\n  [三類音樂 vs Target Music 的 cosine similarity]")
    print(f"  {'類別':<20} {'μ (per-music)':>14} {'σ':>8} {'樣本數':>8}")
    print(f"  {'-'*54}")
    for name, arr in [('Core Music', core_all),
                      ('Exploratory Music', exp_all),
                      ('Negative Music', neg_all)]:
        print(f"  {name:<20} {np.mean(arr):>14.4f} {np.std(arr):>8.4f} {len(arr):>8}")

    print(f"\n  [Mann-Whitney U 顯著性檢定（one-sided: A > B）]")
    print(f"  Core vs Exploratory : p = {p_ce:.2e}  {sig_stars(p_ce)}  "
          f"{'✓ 顯著' if p_ce < 0.05 else '✗ 不顯著'}")
    print(f"  Exploratory vs Neg  : p = {p_en:.2e}  {sig_stars(p_en)}  "
          f"{'✓ 顯著' if p_en < 0.05 else '✗ 不顯著'}")
    print(f"  Core vs Negative    : p = {p_cn:.2e}  {sig_stars(p_cn)}  "
          f"{'✓ 顯著' if p_cn < 0.05 else '✗ 不顯著'}")

    print(f"\n  [Per-User 梯度一致率]")
    print(f"  Core > Exp > Neg（完整預期梯度）: "
          f"{grad_rate*100:.1f}%  ({sum(grad_hold)}/{N})")
    for k, v in sorted(orders.items(), key=lambda x: -x[1]):
        pct = v / N * 100
        marker = ' ← expected' if k == 'core>exp>neg' else ''
        print(f"  {k:<20} : {pct:5.1f}%  (n={v}){marker}")

    print(f"\n  [Per-User 均值]")
    print(f"  Core       mean sim : {np.mean(core_mean):.4f} ± {np.std(core_mean):.4f}")
    print(f"  Exploratory mean sim: {np.mean(exp_mean):.4f} ± {np.std(exp_mean):.4f}")
    print(f"  Negative   mean sim : {np.mean(neg_mean):.4f} ± {np.std(neg_mean):.4f}")
    print(f"{'='*68}\n")


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=== 實驗二：Stage 2 音訊語義梯度驗證（v1）===")

    # 1. 載入 Stage 2 三類 music_id
    records, needed_ids = collect_user_records(CFG.N_USERS)
    if not records:
        raise RuntimeError(
            "找不到有效用戶記錄，請確認路徑：\n"
            f"  profiles.jsonl : {CFG.PROFILES}\n"
            f"  history_dir    : {CFG.HISTORY_DIR}"
        )

    # 2. 建立 AST 索引
    ast_index = build_ast_index(needed_ids)

    # 3. 計算 cosine similarity
    sims = compute_similarities(records, ast_index)
    if sims['n_valid'] == 0:
        raise RuntimeError(
            "計算完成但沒有有效用戶（target + 三類都有 AST 向量），\n"
            "請確認 HDF5 路徑和 music_id 格式是否匹配。"
        )

    # 4. 繪圖與輸出
    plot_all(sims)
    print(f"✅ 輸出：{CFG.OUTPUT_FILE}")


if __name__ == "__main__":
    main()