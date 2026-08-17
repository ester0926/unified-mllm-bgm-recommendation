"""
實驗二：LLM 偏好萃取的邏輯與一致性驗證
========================================================
目標：展示 RecLLM (Gemma) 能從 Positive/Exploratory/Negative 三類對話
      中正確識別衝突標籤並生成一致的用戶偏好畫像。

輸出：
  exp2_consistency_table.png — 1~2 位用戶的輸入/標籤/輸出對照表

流程：
1. 從 Stage 3 對話資料讀取三類對話節錄
2. 從 Stage 4 profiles.jsonl 讀取對應的 conflict_tags + summary_text
3. 自動選取最適合展示的用戶（三種標籤都有、summary 完整）
4. 繪製對照表圖片

使用方式：
  cd "<repo_root>"
  python exp2_consistency_table.py
"""

import json
import logging
import warnings
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# ★ CONFIG
# ============================================================
class CFG:
    BASE_DIR     = Path(r"data/user_profiling")

    # Stage 3：單一 JSONL 檔（每行一筆對話，含 music_id + dialogue_type）
    STAGE3_JSONL = BASE_DIR / "long_term_preference/stage3_dialogues/diverse_template/dialogues.jsonl"

    # Stage 4：RecLLM 輸出
    PROFILES     = BASE_DIR / "long_term_preference/stage4_recLLM/profiles.jsonl"

    OUTPUT_DIR   = BASE_DIR / "experiments/visualization_outputs"
    OUTPUT_FILE  = OUTPUT_DIR / "exp2_consistency_table.png"

    # 展示幾位用戶（建議 1~2）
    N_USERS      = 2

    # 掃描前多少個 profiles 來選樣
    MAX_SCAN     = 2000

    # 對話節錄最大字數（截斷用）
    MAX_QUOTE_CHARS = 120

    # Summary 最大字數（截斷用）
    MAX_SUMMARY_CHARS = 300

    COLORS = {
        'confirm':         '#27AE60',   # [CONFIRM]
        'new':             '#2D6BE4',   # [NEW]
        'confirm_dislike': '#E74C3C',   # [CONFIRM_DISLIKE]
        'modulate':        '#8E44AD',   # [MODULATE]
        'conflict':        '#F39C12',   # [CONFLICT]
        'neutral':         '#7F8C8D',   # 其他
        'header_bg':       '#2C3E50',
        'header_fg':       'white',
        'row_even':        '#F2F3F4',
        'row_odd':         'white',
        'border':          '#BDC3C7',
        'summary_bg':      '#EBF5FB',
        'bg':              '#FAFAFA',
        'text':            '#2C3E50',
        'title':           '#1A252F',
    }

    # Conflict tag 對應顏色與說明
    # 資料中 tag 格式為不帶括號的字串（如 CONFIRM），顯示時加 []
    TAG_META = {
        '[CONFIRM]':         ('confirm',         'Core preference confirmed'),
        '[CONFIRM_DISLIKE]': ('confirm_dislike', 'Dislike confirmed / Noise filtered'),
        '[MODULATE]':        ('modulate',        'Preference signal modulated'),
        '[NEW]':             ('new',             'New preference detected'),
        '[CONFLICT]':        ('conflict',        'Conflicting signal detected'),
    }


# ============================================================
# CJK 字型偵測（與 exp3 相同邏輯）
# ============================================================

def get_cjk_font() -> FontProperties:
    from matplotlib import font_manager as fm
    candidates = [
        'Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun',
        'PingFang TC', 'PingFang SC', 'Noto Sans CJK TC', 'Noto Sans CJK SC',
        'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'IPAGothic',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            logger.info(f"CJK 字型: {name}")
            return FontProperties(family=name)
    return FontProperties(family='DejaVu Sans')


# ============================================================
# Step 1: 載入資料
# ============================================================

def load_profiles(max_n: int) -> List[Dict]:
    import jsonlines
    profiles = []
    with jsonlines.open(CFG.PROFILES, 'r') as reader:
        for obj in reader:
            if 'music_id' not in obj and 'target_music' in obj:
                obj['music_id'] = obj['target_music']
            profiles.append(obj)
            if len(profiles) >= max_n:
                break
    return profiles


def load_stage3_index() -> Dict[str, Dict[str, Dict]]:
    """
    讀取 Stage 3 單一 JSONL 檔，建立索引：
      {music_id: {'positive': obj, 'exploratory': obj, 'negative': obj}}
    每個 obj 保留完整的那一行資料（含 dialogue_turns）。
    """
    index: Dict[str, Dict[str, Dict]] = {}
    path = CFG.STAGE3_JSONL
    if not path.exists():
        logger.warning(f"Stage 3 JSONL 不存在: {path}")
        return index

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid   = obj.get('music_id', '')
            dtype = obj.get('dialogue_type', '').lower()   # 'positive'/'exploratory'/'negative'
            if mid and dtype:
                if mid not in index:
                    index[mid] = {}
                index[mid][dtype] = obj

    logger.info(f"Stage 3 索引建立完成：{len(index)} 個 music_id")
    return index


def extract_dialogue_quote(dialogue_obj: Dict, dtype: str,
                            max_chars: int = CFG.MAX_QUOTE_CHARS) -> str:
    """
    取對話的最後 2～3 個回合（User + Recommender 交替）作為節錄，
    能清楚展示對話的走向與結論。
    """
    turns = dialogue_obj.get('dialogue_turns', [])
    if not turns:
        return '(no turns found)'

    # 取最後 N 個 turn（User + Recommender 各算一個）
    last_turns = turns[-4:]   # 最後 4 個 turn ≈ 2 個完整回合

    lines = []
    for t in last_turns:
        if not isinstance(t, dict):
            continue
        role    = t.get('role', '').strip()
        content = t.get('content', '').strip()
        if role and content:
            short = textwrap.shorten(content, width=120, placeholder='...')
            lines.append(f"{role}: {short}")

    if not lines:
        return '(no turns found)'

    return '\n'.join(lines)


def load_dialogue(music_id: str) -> Optional[Dict]:
    """保留此函數簽名供舊程式碼相容，實際不再使用"""
    return None


def extract_conflict_tags(profile: Dict) -> List[Tuple[str, str]]:
    """
    從 salient_facts 中提取 conflict_tag + fact 組合。
    資料格式：salient_facts = [{"fact": "...", "conflict_tag": "CONFIRM"}, ...]
    顯示時在 tag 前後加上 [] 以對應論文格式。
    """
    result = []
    facts = profile.get('salient_facts', [])
    for item in facts:
        if not isinstance(item, dict):
            continue
        raw_tag = item.get('conflict_tag', '').strip().upper()
        fact    = item.get('fact', '').strip()
        if not fact:
            continue
        # 加上括號，統一為 [TAG] 格式
        display_tag = f'[{raw_tag}]' if raw_tag else ''
        result.append((display_tag, fact))
    return result


def get_summary_text(profile: Dict, max_chars: int = CFG.MAX_SUMMARY_CHARS) -> str:
    """與其他實驗相同的 summary 取法"""
    facts = profile.get('salient_facts', [])
    if facts:
        if isinstance(facts[0], dict):
            texts = [f.get('fact', '') for f in facts]
        else:
            texts = list(facts)
        text = ' '.join(t for t in texts if t)
        if text.strip():
            return textwrap.shorten(text, width=max_chars, placeholder='...')
    return textwrap.shorten(
        profile.get('summary_text', '(no summary)'),
        width=max_chars, placeholder='...'
    )


# ============================================================
# Step 2: 選樣——找最適合展示的用戶
# ============================================================

def score_profile(profile: Dict, stage3_by_type: Dict[str, Dict]) -> float:
    """
    評分邏輯（數字越高越適合展示）：
    - salient_facts 中有至少 2 種不同 tag +2
    - 同時含 CONFIRM 和 CONFIRM_DISLIKE +2（最能展示衝突調節）
    - summary_text 夠長 +1
    - Stage 3 三種對話都有 +3，有 2 種 +1
    """
    score = 0.0
    tags = extract_conflict_tags(profile)
    tag_types = {t[0] for t in tags if t[0]}

    if len(tag_types) >= 2:
        score += 2
    if '[CONFIRM]' in tag_types and '[CONFIRM_DISLIKE]' in tag_types:
        score += 2

    summary = get_summary_text(profile)
    if len(summary) > 80:
        score += 1

    n_types = len(stage3_by_type)
    if n_types >= 3:
        score += 3
    elif n_types >= 2:
        score += 1

    return score


def select_best_profiles(n: int = CFG.N_USERS) -> List[Tuple[Dict, Dict]]:
    """
    掃描 profiles，選 score 最高的 n 位用戶。
    回傳 list of (profile, stage3_by_type)，
    其中 stage3_by_type = {'positive': obj, 'exploratory': obj, 'negative': obj}
    """
    logger.info("建立 Stage 3 索引...")
    stage3_index = load_stage3_index()

    profiles = load_profiles(CFG.MAX_SCAN)
    logger.info(f"掃描 {len(profiles)} 個 profiles 選樣...")

    scored = []
    for p in profiles:
        mid = p.get('music_id', '')
        s3  = stage3_index.get(mid, {})
        s   = score_profile(p, s3)
        if s > 0:
            scored.append((s, p, s3))

    scored.sort(key=lambda x: x[0], reverse=True)

    logger.info(f"有效候選: {len(scored)}")
    for i, (s, p, s3) in enumerate(scored[:5]):
        mid      = p.get('music_id')
        tags     = {t[0] for t in extract_conflict_tags(p) if t[0]}
        s3_types = list(s3.keys())
        logger.info(f"  [{i+1}] {mid}  score={s:.1f}  "
                    f"tags={tags}  dialogues={s3_types}")

    return [(p, s3) for _, p, s3 in scored[:n]]


# ============================================================
# Step 3: 繪製對照表
# ============================================================

def wrap_text(text: str, width: int = 45) -> str:
    return '\n'.join(textwrap.wrap(text, width=width))


def draw_user_panel(
    fig,
    gs_slot,
    profile: Dict,
    stage3_by_type: Dict[str, Dict],   # {'positive': obj, ...}
    user_index: int,
    cjk_font: FontProperties,
):
    """
    繪製單一用戶的三欄面板：
    左欄：輸入對話節錄 (3 行)
    中欄：Conflict Tags
    右欄：Summary Text
    """
    C = CFG.COLORS
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    mid = profile.get('music_id', 'unknown')

    # ── 用戶標題列 ──────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.90), 1.0, 0.09,
        boxstyle='round,pad=0.01',
        facecolor=C['header_bg'], edgecolor='none', zorder=2
    ))
    ax.text(0.5, 0.945,
            f'User #{user_index + 1}  |  music_id: {mid}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=C['header_fg'], zorder=3)

    # ── 三個區塊的 x 邊界 ────────────────────────────────────
    col_x = [0.0, 0.38, 0.64, 1.0]   # 左 / 中 / 右 邊界
    col_labels = ['Input Dialogues', 'Conflict Tags\n(RecLLM Output)', 'User Profile\n(80-100 words)']
    col_bg = ['#FDFEFE', '#F9EBEA', '#EBF5FB']

    for ci in range(3):
        ax.add_patch(mpatches.FancyBboxPatch(
            (col_x[ci] + 0.005, 0.01),
            col_x[ci+1] - col_x[ci] - 0.01, 0.87,
            boxstyle='round,pad=0.01',
            facecolor=col_bg[ci],
            edgecolor=C['border'], linewidth=0.8, zorder=1
        ))
        ax.text((col_x[ci] + col_x[ci+1]) / 2, 0.86,
                col_labels[ci],
                ha='center', va='center', fontsize=9,
                fontweight='bold', color=C['text'],
                fontproperties=cjk_font)

    # ── 左欄：輸入對話節錄 ──────────────────────────────────
    dialogue_types = [
        ('Positive',    '#27AE60', '🎵'),
        ('Exploratory', '#F39C12', '🔍'),
        ('Negative',    '#E74C3C', '🚫'),
    ]
    y_pos = 0.78
    for dtype, dcolor, icon in dialogue_types:
        dtype_key = dtype.lower()
        s3_obj = stage3_by_type.get(dtype_key)
        if s3_obj:
            quote = extract_dialogue_quote(s3_obj, dtype_key)
        else:
            quote = f'(no {dtype} dialogue found)'

        ax.text(col_x[0] + 0.015, y_pos,
                f'{dtype}:',
                ha='left', va='top', fontsize=8,
                fontweight='bold', color=dcolor)
        y_pos -= 0.055

        wrapped = wrap_text(f'"{quote}"', width=40)
        ax.text(col_x[0] + 0.015, y_pos,
                wrapped,
                ha='left', va='top', fontsize=7.5,
                color='#555555', style='italic',
                fontproperties=cjk_font,
                wrap=True)
        n_lines = wrapped.count('\n') + 1
        y_pos -= 0.055 + (n_lines - 1) * 0.04

    # ── 中欄：Conflict Tags ──────────────────────────────────
    tags = extract_conflict_tags(profile)

    if not tags:
        ax.text((col_x[1] + col_x[2]) / 2, 0.50,
                '(No conflict tags\nfound in profile)',
                ha='center', va='center', fontsize=8,
                color='#999999', style='italic')
    else:
        y_pos = 0.78
        for tag, text in tags[:8]:   # 最多顯示 8 筆
            color_key = CFG.TAG_META.get(tag, ('neutral', ''))[0]
            tag_color  = C.get(color_key, C['neutral'])

            # Tag 標籤背景
            ax.add_patch(mpatches.FancyBboxPatch(
                (col_x[1] + 0.01, y_pos - 0.022), 0.22, 0.025,
                boxstyle='round,pad=0.005',
                facecolor=tag_color, edgecolor='none', alpha=0.85, zorder=3
            ))
            ax.text(col_x[1] + 0.12, y_pos - 0.008,
                    tag if tag else '—',
                    ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color='white', zorder=4)

            # Tag 對應文字
            short_text = textwrap.shorten(text, width=35, placeholder='...')
            ax.text(col_x[1] + 0.01, y_pos - 0.035,
                    short_text,
                    ha='left', va='top', fontsize=7,
                    color='#444444', fontproperties=cjk_font)
            y_pos -= 0.10

    # ── 右欄：Summary Text ───────────────────────────────────
    summary = get_summary_text(profile)
    wrapped_summary = '\n'.join(textwrap.wrap(summary, width=32))

    ax.text(col_x[2] + 0.01, 0.78,
            wrapped_summary,
            ha='left', va='top', fontsize=7.8,
            color='#2C3E50', linespacing=1.5,
            fontproperties=cjk_font)

    # 字數統計
    word_count = len(summary.split())
    ax.text(col_x[2] + 0.01, 0.04,
            f'{word_count} words',
            ha='left', va='bottom', fontsize=7.5,
            color='#888888', style='italic')


def print_panel_content(selected: List[Tuple[Dict, Dict]]):
    """
    將圖表中每個用戶面板的完整文字內容印到 terminal，
    方便手動製作表格。
    """
    SEP = "=" * 80

    for i, (profile, stage3_by_type) in enumerate(selected):
        mid = profile.get('music_id', 'unknown')
        print(f"\n{SEP}")
        print(f"User #{i+1}  |  music_id: {mid}")
        print(SEP)

        # ── 左欄：輸入對話節錄 ──────────────────────────────────
        print("\n【左欄】Input Dialogues")
        print("-" * 40)
        for dtype in ['positive', 'exploratory', 'negative']:
            s3_obj = stage3_by_type.get(dtype)
            if s3_obj:
                quote = extract_dialogue_quote(s3_obj, dtype)
            else:
                quote = f'(no {dtype} dialogue found)'
            print(f"  [{dtype.capitalize()}]")
            print(f"  {quote}")
            print()

        # ── 中欄：Conflict Tags ──────────────────────────────────
        print("【中欄】Conflict Tags (RecLLM Output)")
        print("-" * 40)
        tags = extract_conflict_tags(profile)
        if not tags:
            print("  (no tags found)")
        else:
            for tag, fact in tags:
                print(f"  {tag:<20}  {fact}")
        print()

        # ── 右欄：User Profile ───────────────────────────────────
        print("【右欄】User Profile (Summary)")
        print("-" * 40)
        summary = get_summary_text(profile, max_chars=99999)  # 不截斷
        print(f"  {summary}")
        word_count = len(summary.split())
        print(f"\n  ({word_count} words)")
        print()

    print(SEP)

    """主繪圖函數"""
    plt.rcParams.update({
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.unicode_minus': False,
    })
    cjk_font = get_cjk_font()
    C = CFG.COLORS
    n = len(selected)

    fig = plt.figure(figsize=(16, 6.5 * n + 2))
    fig.patch.set_facecolor(C['bg'])

    gs = GridSpec(n + 1, 1, figure=fig,
                  height_ratios=[0.4] + [1] * n,
                  hspace=0.12)

    # ── 全圖標題 ──────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.set_facecolor(C['bg'])
    ax_title.text(
        0.5, 0.85,
        'Experiment 2: LLM Preference Extraction — Consistency & Conflict Resolution',
        ha='center', va='top', fontsize=14, fontweight='bold', color=C['title']
    )
    ax_title.text(
        0.5, 0.40,
        'Input: Positive / Exploratory / Negative dialogues  →  '
        'RecLLM (Gemma) conflict tagging  →  Output: 80-100 word user profile',
        ha='center', va='top', fontsize=10, color='#555555'
    )

    # 圖例
    legend_items = []
    for tag, (color_key, desc) in CFG.TAG_META.items():
        patch = mpatches.Patch(color=C[color_key], label=f'{tag}  {desc}')
        legend_items.append(patch)
    ax_title.legend(
        handles=legend_items, loc='lower center',
        ncol=4, fontsize=8.5, framealpha=0.9,
        edgecolor=C['border'], facecolor=C['bg'],
        bbox_to_anchor=(0.5, -0.05)
    )

    # ── 各用戶面板 ────────────────────────────────────────────
    for i, (profile, stage3_by_type) in enumerate(selected):
        draw_user_panel(fig, gs[i + 1], profile, stage3_by_type, i, cjk_font)

    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CFG.OUTPUT_FILE, dpi=160, bbox_inches='tight',
                facecolor=C['bg'])
    plt.close()
    logger.info(f"✅ 圖表儲存: {CFG.OUTPUT_FILE}")


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=== 實驗二：LLM 一致性驗證 ===")

    selected = select_best_profiles(n=CFG.N_USERS)

    if not selected:
        logger.error("找不到適合的 profile，請確認 PROFILES 路徑是否正確")
        return

    logger.info(f"選定 {len(selected)} 位用戶進行視覺化")
    for i, (p, s3) in enumerate(selected):
        mid  = p.get('music_id')
        tags = extract_conflict_tags(p)
        summary = get_summary_text(p)
        logger.info(f"\n用戶 #{i+1}: {mid}")
        logger.info(f"  Tags ({len(tags)} 筆): {[t[0] for t in tags[:6]]}")
        logger.info(f"  Summary: {summary[:80]}...")
        found_types = list(s3.keys())
        if found_types:
            logger.info(f"  Stage 3 對話: ✅ {found_types}")
        else:
            logger.warning(f"  Stage 3 對話: ❌ 未找到")

    # ── 印出所有文字內容（方便手動製表）──────────────────────
    print_panel_content(selected)

    print(f"\n✅ 輸出: {CFG.OUTPUT_FILE}")
    print(f"\n📂 Stage 3 路徑: {CFG.STAGE3_JSONL}")
    print(f"📂 Stage 4 路徑: {CFG.PROFILES}")


if __name__ == "__main__":
    main()