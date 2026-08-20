"""
用途：啟動 Streamlit 展示介面，用來呼叫後端服務並查看推薦結果與解釋。
輸入：依照程式內設定讀取模型 checkpoint、metadata、cache 與測試樣本。
輸出：提供本機介面或 API 回傳推薦分數、候選清單與文字解釋。
執行：請先依 README 與 ZENODO.md 放好資料，再從 repo 根目錄啟動。
"""

import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Music XAI · 推薦決策說明介面",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f7f5; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e3db; }
.xai-card {
    background:#ffffff; border:1px solid #e5e3db;
    border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:14px;
}
.section-title { font-size:13px; font-weight:600; color:#3d3d3a; margin-bottom:10px; }
.badge { display:inline-block; font-size:11px; font-weight:500;
         padding:2px 10px; border-radius:99px; margin-right:4px; }
.badge-ok     { background:#EAF3DE; color:#3B6D11; }
.badge-warn   { background:#FAEEDA; color:#854F0B; }
.badge-info   { background:#E6F1FB; color:#185FA5; }
.badge-danger { background:#FCEBEB; color:#A32D2D; }
.badge-gray   { background:#F1EFE8; color:#5F5E5A; }
.nl-box {
    background:#f7f7f5; border-left:3px solid #378ADD;
    border-radius:0 8px 8px 0; padding:10px 14px;
    font-size:13.5px; line-height:1.7; color:#3d3d3a;
}
.fallback-box {
    background:#f7f7f5; border-left:3px solid #888780;
    border-radius:0 8px 8px 0; padding:10px 14px;
    font-size:13.5px; line-height:1.7; color:#3d3d3a;
}
.caveat { font-size:11.5px; color:#888780; line-height:1.55; margin-top:8px; }
.warn-block  { background:#FAEEDA; border-radius:8px; padding:10px 14px;
               font-size:12.5px; color:#854F0B; margin-top:10px; }
.ok-block    { background:#EAF3DE; border-radius:8px; padding:10px 14px;
               font-size:12.5px; color:#3B6D11; margin-top:10px; }
.info-block  { background:#E6F1FB; border-radius:8px; padding:10px 14px;
               font-size:12.5px; color:#185FA5; margin-top:10px; }
.danger-block { background:#FCEBEB; border-radius:8px; padding:10px 14px;
                font-size:12.5px; color:#A32D2D; margin-top:10px; }
.insight-block { background:#EAF3DE; border-left:4px solid #1D9E75;
                 border-radius:0 8px 8px 0; padding:10px 14px;
                 font-size:12.5px; color:#085041; margin-top:10px; font-weight:500; }
.tab-header  { font-size:13px; font-weight:600; color:#3d3d3a; margin:0 0 12px;
               padding-bottom:6px; border-bottom:2px solid #378ADD; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# API 呼叫相關函式
# ══════════════════════════════════════════════════════════════════════════════

def call_infer(pair_key: str, user_text: str, pool_seed_idx: int) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/infer",
            json={"pair_key": pair_key, "user_text": user_text,
                  "pool_seed_idx": pool_seed_idx},
            timeout=420,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ 無法連線 Backend。請先執行：uvicorn model_service:app --port 8000")
    except requests.exceptions.Timeout:
        st.error("⏱ 推論超時（500-pool 約需 1~3 分鐘），請確認 GPU 正常。")
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend 錯誤：{e.response.text}")
    return None


def get_health() -> dict | None:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=4).json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_sample_ids() -> list[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/sample_ids?n=200", timeout=10)
        return r.json().get("pair_keys", [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 圖表繪製相關函式
# ══════════════════════════════════════════════════════════════════════════════

def pool_chart(scores: list, bpr: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores, nbinsx=35,
                               marker_color="#B5D4F4", marker_line_width=0, name="候選曲"))
    fig.add_vline(x=bpr, line_color="#378ADD", line_width=2, line_dash="dash",
                  annotation_text=f"GT {bpr:.3f}",
                  annotation_font_size=11, annotation_font_color="#185FA5")
    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="BPR Score", title_font_size=11,
                   tickfont_size=10, gridcolor="#e5e3db"),
        yaxis=dict(title="Count", title_font_size=11,
                   tickfont_size=10, gridcolor="#e5e3db"),
        bargap=0.05,
    )
    return fig


def signed_contrib_fig(signed: list) -> go.Figure:
    """
    帶正負號的 delta 橫條圖。
    正值（藍）= 移除後分數下降 → 該模態支持 GT
    負值（紅）= 移除後分數上升 → 該模態干擾 GT（失敗案例的關鍵信號）
    """
    names  = [x["name"] for x in signed]
    deltas = [x["delta"] for x in signed]
    colors = ["#378ADD" if d > 0.005 else "#E24B4A" if d < -0.005 else "#888780"
              for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=names, orientation="h",
        marker_color=colors,
        text=[f"{d:+.4f}" for d in deltas],
        textposition="outside", textfont_size=11,
    ))
    fig.add_vline(x=0, line_color="#888780", line_width=1)
    fig.update_layout(
        height=240, margin=dict(l=0, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Δ score = 完整 - 移除後（正值 = 支持 GT，負值 = 干擾 GT）",
                   title_font_size=10, tickfont_size=10, gridcolor="#e5e3db"),
        yaxis=dict(tickfont_size=12), showlegend=False,
    )
    return fig


def counterfactual_fig(cfs: list) -> go.Figure:
    labels = [x["label"] for x in cfs]
    deltas = [x["delta_score"] for x in cfs]
    colors = ["#888780" if abs(d) < 0.001 else "#1D9E75" if d > 0 else "#E24B4A"
              for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{d:+.4f}" for d in deltas],
        textposition="outside", textfont_size=11,
    ))
    fig.add_vline(x=0, line_color="#888780", line_width=1)
    fig.update_layout(
        height=220, margin=dict(l=0, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Δ score vs 原始（正值 = GT 分數上升，負值 = 下降）",
                   title_font_size=10, tickfont_size=10, gridcolor="#e5e3db"),
        yaxis=dict(tickfont_size=11), showlegend=False,
    )
    return fig


def conf_badge(gap: float, rank: int) -> tuple[str, str]:
    """
    信心等級判斷。
    修正：rank > 50 無論 gap 多大都應為低信心。
    Gap 大但 rank 爛 = 「穩定地推錯」，不是高信心。
    """
    if rank <= 10 and gap > 0.08:  return "高信心",              "badge-ok"
    if rank <= 20 and gap > 0.04:  return "中等信心",            "badge-warn"
    if rank <= 50:                  return "中等信心（rank 偏後）", "badge-warn"
    # rank > 50：無論 gap 多大都是低信心
    if gap > 0.04:
        return "低信心 — 穩定誤推薦", "badge-danger"
    return "低信心 — 不穩定",         "badge-danger"


def clean_nl(text: str) -> tuple[str, bool]:
    """
    去除 generate() fallback 前綴，回傳 (clean_text, is_fallback)。
    is_fallback=True 表示這是標註 t4 而非真正 model.generate() 的輸出。
    """
    prefixes = [
        "[generate 發生例外，顯示標註 t4]\n",
        "[generate 失敗",
        "[generate 發生例外",
    ]
    for p in prefixes:
        if text.startswith(p):
            clean = text[len(p):].strip() if "\n" in p else text
            return clean, True
    return text, False


# ══════════════════════════════════════════════════════════════════════════════
# 側邊欄
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎵 Music XAI")
    st.caption("Unified MLLM Pointwise v3 · mc Hard Negative · 新增 [RANK] token")
    st.divider()

    h = get_health()
    if h and h.get("status") == "ok":
        st.markdown(
            f"<div style='font-size:12px;color:#3B6D11;background:#EAF3DE;"
            f"border-radius:8px;padding:6px 10px;'>"
            f"✅ Backend 已連線<br>"
            f"songs: {h['songs_loaded']} · users: {h['users_loaded']}<br>"
            f"test pairs: {h['test_pairs']} · device: {h['device']}</div>",
            unsafe_allow_html=True,
        )
    elif h and h.get("status") == "loading":
        st.warning("⏳ 模型載入中，請稍後…")
    else:
        st.markdown(
            "<div style='font-size:12px;color:#A32D2D;background:#FCEBEB;"
            "border-radius:8px;padding:6px 10px;'>"
            "❌ Backend 離線<br>"
            "<code>uvicorn model_service:app --port 8000</code></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**選擇推論樣本**")

    pair_keys = get_sample_ids()
    if not pair_keys:
        pair_keys = ["（Backend 未連線）"]

    sel_pair_key = st.selectbox(
        "Pair Key（= GT Music ID）", pair_keys,
        help="23碼 pair_key，同時是 gt_music_id（dataset.py line 385）",
    )
    st.caption(f"video_id: `{sel_pair_key[:11]}`")

    user_text = st.text_area(
        "使用者輸入文字（可留空，自動使用 conv_map t3）", value="", height=70,
    )
    pool_seed_idx = st.number_input(
        "Pool seed idx", min_value=0, max_value=9999, value=0, step=1,
        help="seed = 20260315 + idx，影響 499 首負例組成",
    )
    run_btn = st.button("🚀 執行推論", type="primary", use_container_width=True)

    st.divider()
    st.caption("BACKEND_URL 環境變數可覆蓋 localhost:8000")


# ══════════════════════════════════════════════════════════════════════════════
# 主畫面
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🎵 Music XAI：推薦決策說明介面")
st.caption("區分模型忠實解釋（faithful explanation）與輔助說明（auxiliary explanation）")

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn and sel_pair_key and "Backend" not in sel_pair_key:
    with st.spinner("推論中… 500-pool 排名 + 消融 + 反事實分析，約需 1~3 分鐘"):
        r = call_infer(sel_pair_key, user_text.strip(), int(pool_seed_idx))
        if r:
            st.session_state.result = r
            st.success("✅ 推論完成")

result = st.session_state.result
if result is None:
    st.info("👈 左側選擇 pair_key 後點擊「執行推論」，介面將呈現真實模型輸出。")
    st.stop()

# ── 取出後端回傳欄位 ─────────────────────────────────────────────────────────
pair_key      = result["pair_key"]
video_id      = result["video_id"]
gt_mid        = result["gt_music_id"]
prompt_txt    = result["prompt_text"]
prompt_src    = result.get("prompt_source", "unknown")
bpr           = result["bpr_score"]
rank          = result["pool_rank"]
gap           = result["gap_to_2nd"]
gate          = result["gate_value"]
gate_disp     = result.get("gate_display", f"{gate:.3f}")
case_status   = result.get("case_status", "borderline")
case_summary  = result.get("case_summary", "")
nl_raw        = result["nl_explanation"]
contrib       = result["modality_contrib"]
contrib_signed = result.get("contrib_signed", [])
abl           = result["ablation"]
ltp_b         = result["ltp_branch"]
top5          = result["top5"]
scores_500    = result["all_pool_scores"]
contrastive   = result.get("contrastive", {})
cfs           = result.get("counterfactuals", [])
warns         = result["warnings"]
todos         = result["todos"]

nl_text, nl_is_fallback = clean_nl(nl_raw)
cl, cc = conf_badge(gap, rank)

# ── Case status 推薦品質橫幅 ──────────────────────────────────────────────
case_colors = {
    "success":    ("ok-block",      "✅"),
    "borderline": ("warn-block",    "⚠️"),
    "failure":    ("danger-block",  "❌"),
}
case_cls, case_icon = case_colors.get(case_status, ("info-block", "ℹ️"))
st.markdown(f'<div class="{case_cls}">{case_icon} {case_summary}</div>',
            unsafe_allow_html=True)

# ── 4 個關鍵指標（2×2 layout 避免窄螢幕截斷）──────────────────────────────
# 以 2+2 排列指標，避免窄視窗截斷
row1_c1, row1_c2 = st.columns(2)
row2_c1, row2_c2 = st.columns(2)

with row1_c1:
    st.metric("BPR Score", f"{bpr:.4f}",
              help="GT 音樂的 Pointwise ranking score（全域可比）。負值代表模型認為 GT 比基準差，是訓練失敗的信號。")

with row1_c2:
    # rank 大於 50 時以低信心狀態顯示
    rank_delta = f"前 {rank} 名" if rank <= 50 else f"後 {500 - rank} 名"
    rank_color = "normal" if rank <= 50 else "inverse"
    st.metric("500-pool Rank", f"{rank}",
              delta=rank_delta, delta_color=rank_color,
              help="GT 在 500-pool 中的排名，越小越好。")

with row2_c1:
    # gap 雖大但排名差時仍視為低信心
    if rank > 50 and gap > 0.04:
        gap_label = f"穩定誤推薦（差距 {gap:.4f}）"
        gap_color = "inverse"   # 紅色——穩定是壞事
    elif gap < 0.01:
        gap_label = "極不穩定（完全並列）"
        gap_color = "inverse"
    elif gap < 0.04:
        gap_label = "不穩定"
        gap_color = "inverse"
    elif gap < 0.08:
        gap_label = "中等穩定"
        gap_color = "off"
    else:
        gap_label = "穩定"
        gap_color = "normal"
    st.metric("Gap to 2nd", f"{gap:.4f}", delta=gap_label, delta_color=gap_color,
              help="第 1 名與第 2 名的分差。Gap 大但 Rank 差 = 模型穩定地把 GT 排在後面。")

with row2_c2:
    st.metric("Gate（P_ltp）", gate_disp,
              help="ltp_gate_scalar 的 sigmoid 值。缺 gate.pt 時顯示 0.500（初始值，不可解讀）。")

# 推薦狀態標籤
st.markdown(
    f'<span class="badge {cc}">{cl}</span>'
    f'<span class="badge badge-gray">pair_key {pair_key}</span>'
    f'<span class="badge badge-gray">video_id {video_id}</span>'
    f'<span class="badge badge-gray">prompt: {prompt_src}</span>',
    unsafe_allow_html=True,
)

# 警告與待確認事項
for w in warns:
    st.markdown(f'<div class="warn-block">{w}</div>', unsafe_allow_html=True)

if todos:
    with st.expander("ℹ️ 系統提示（展開查看）", expanded=False):
        for t in todos:
            st.markdown(f'<div class="info-block">{t}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs（5 個）
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 輔助說明",
    "🔬 模態消融",
    "↔️ 對比與反事實",
    "⚠️ 不確定性",
    "📊 候選池比較",
])

# ─────────────────────────────────────────────────────────────────────────────
# 頁籤 1：推薦文字說明
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("**輔助敘述（non-faithful summary）**")
    st.markdown(
        '<div class="info-block" style="margin-bottom:10px;">'
        '這段文字屬於 generation/fallback 輔助敘述，<strong>不應取代 faithful explanation</strong>，'
        '也不應視為 ranking branch 的直接證據。'
        '真實的決策依據請看「模態消融」頁籤的 Δ score。'
        '</div>',
        unsafe_allow_html=True,
    )

    # 備用文字只顯示內容，不顯示內部錯誤前綴
    box_cls = "fallback-box" if nl_is_fallback else "nl-box"
    st.markdown(
        f'<div class="xai-card"><div class="{box_cls}">{nl_text}</div>'
        f'<div class="caveat" style="margin-top:8px;">'
        f'{"⚠️ model.generate() 呼叫失敗，此為 conv_map 標註 t4（人工標注的 GT response）。" if nl_is_fallback else "由 LLaMA 2-7B generation branch 生成。"}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**實際使用的 prompt text：**")
    st.code(prompt_txt, language=None)


# ─────────────────────────────────────────────────────────────────────────────
# 頁籤 2：模態消融與支持來源說明
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Faithful explanation：模態消融**")
    st.caption("Δ score = 完整模型分數 - 移除後分數。正值代表該模態支持 GT；**負值代表該模態主動干擾 GT**（失敗案例的核心信號）。")

    # 所有 delta 為負時，代表此案例可作為失敗分析
    all_negative = contrib_signed and all(x["delta"] <= 0 for x in contrib_signed)

    if all_negative:
        st.markdown(
            '<div class="danger-block">🔴 <strong>所有模態 delta 均為負值</strong>：'
            '移除任何模態後 GT 分數反而上升。這是此次推薦失敗的根本原因——'
            '模型的多模態表徵共同指向了錯誤的音樂，而非 GT。'
            '這不是程式錯誤，而是模型在此案例中的真實失敗模式。</div>',
            unsafe_allow_html=True,
        )

    if contrib_signed:
        st.plotly_chart(signed_contrib_fig(contrib_signed),
                        use_container_width=True, key="signed_contrib")
    else:
        # 備用處理：改用 ablation dict 建圖
        base = abl.get("完整模型", 0)
        fallback_signed = [
            {"name": k, "delta": round(base - v, 4)}
            for k, v in abl.items() if k != "完整模型"
        ]
        if fallback_signed:
            st.plotly_chart(signed_contrib_fig(fallback_signed),
                            use_container_width=True, key="signed_contrib_fallback")

    # Contrastive 百分比全為 0 時，改用文字說明原因
    if all_negative:
        st.markdown(
            '<div class="caveat">'
            '注意：所有 delta 為負時，正向貢獻百分比（contrib_pct）無意義（全為 0%）。'
            '請以上方的 Δ score 絕對值判斷各模態的干擾程度。'
            '影片內容 Δ 最大負值代表影片特徵是最主要的誤導來源。'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("正向貢獻百分比（僅統計正值 delta，代表支持 GT 的模態）：")
        cols = st.columns(4)
        color_map = {"影片內容": "#378ADD", "長期偏好": "#EF9F27",
                     "文字描述": "#1D9E75", "音樂特徵": "#888780"}
        for i, (name, pct) in enumerate(contrib.items()):
            with cols[i % 4]:
                st.metric(name, f"{pct}%")

    with st.expander("詳細消融數值", expanded=False):
        st.dataframe(
            [{"模態": k, "BPR Score": v, "Δ score": round(abl.get("完整模型", 0) - v, 4)}
             for k, v in abl.items()],
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 頁籤 3：對比與反事實分析
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("**Contrastive：GT vs 模型選中的第 1 名候選**")
        if contrastive:
            # 偵測 GT 是否在所有模態下都受到干擾
            gt_pct  = contrastive.get("gt_contrib_pct", {})
            top1_pct = contrastive.get("top1_contrib_pct", {})
            gt_all_zero = all(v == 0 for v in gt_pct.values())

            st.markdown(
                f'<div class="xai-card">'
                f'<div style="font-size:12px;color:#888780;margin-bottom:8px;">'
                f'GT 候選：<code>{contrastive.get("gt_pair_key","")}</code><br>'
                f'模型 top1：<code>{contrastive.get("model_top1_pair_key","")}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if gt_all_zero:
                # GT 所有模態均為負 delta → 貢獻百分比無意義，直接說明
                st.markdown(
                    '<div class="danger-block" style="margin:0 0 8px;">'
                    '⚠️ GT 所有模態 delta 均為負值（干擾）：百分比在此無意義。'
                    '請看「模態消融」頁籤的 Δ score 絕對值，那才是真實的干擾程度。'
                    '</div>',
                    unsafe_allow_html=True,
                )
                # 改顯示消融的絕對 delta 值（負值才有意義）
                gt_abl = contrastive.get("gt_ablation", {})
                if gt_abl:
                    base = gt_abl.get("完整模型", 0)
                    st.markdown("**GT 各模態干擾程度（Δ score，越負越嚴重）：**")
                    st.dataframe(
                        [{"模態": k, "Δ score": round(base - v, 4),
                          "方向": "🔴 干擾 GT" if (base - v) < -0.01 else "🟡 中性" if abs(base-v) < 0.01 else "🟢 支持 GT"}
                         for k, v in gt_abl.items() if k != "完整模型"],
                        use_container_width=True, hide_index=True,
                    )
            else:
                summary = contrastive.get("summary", "")
                st.markdown(f'<div class="caveat">{summary}</div>', unsafe_allow_html=True)
                col_gt, col_top1 = st.columns(2)
                with col_gt:
                    st.markdown("**GT 模態貢獻（%）**")
                    for name, pct in gt_pct.items():
                        st.metric(name, f"{pct}%")
                with col_top1:
                    st.markdown("**Top1 模態貢獻（%）**")
                    for name, pct in top1_pct.items():
                        st.metric(name, f"{pct}%")

            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Top1 詳細消融數值", expanded=False):
                top1_abl = contrastive.get("top1_ablation", {})
                if top1_abl:
                    base1 = top1_abl.get("完整模型", 0)
                    st.dataframe(
                        [{"模態": k, "BPR Score": v,
                          "Δ score": round(base1 - v, 4)}
                         for k, v in top1_abl.items()],
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.info("Contrastive 資料尚未取得")

    with c2:
        st.markdown("**Counterfactual：改變輸入條件後 GT 分數的變化**")
        st.caption("只計算 GT BPR score 的變化，不重跑 500-pool，無 timeout 風險。")

        if cfs:
            st.plotly_chart(counterfactual_fig(cfs),
                            use_container_width=True, key="cf_fig")

            # 自動突顯 delta 較大的反事實差異
            big_insights = [x for x in cfs if abs(x["delta_score"]) > 0.5 and x["label"] != "原始設定"]
            for ins in big_insights:
                direction = "上升" if ins["delta_score"] > 0 else "下降"
                color_cls = "insight-block" if ins["delta_score"] > 0 else "danger-block"
                st.markdown(
                    f'<div class="{color_cls}">'
                    f'🔍 <strong>關鍵洞察：{ins["label"]}</strong> 使 GT 分數{direction} {ins["delta_score"]:+.4f}。'
                    + (f'代表此模態在本案例中主動誤導了模型，移除後模型反而更容易找到 GT。'
                       if ins["delta_score"] > 0 else
                       f'代表此模態對 GT 的推薦有正向貢獻，移除後推薦品質下降。')
                    + '</div>',
                    unsafe_allow_html=True,
                )

            st.dataframe(cfs, use_container_width=True, hide_index=True)
        else:
            st.info("Counterfactual 資料尚未取得")


# ─────────────────────────────────────────────────────────────────────────────
# 頁籤 4：不確定性
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("**不確定性指標**")
        # 使用 st.metric 顯示不確定性指標
        i1, i2 = st.columns(2)
        with i1:
            st.metric("BPR Score", f"{bpr:.4f}")
            st.metric("Gap to 2nd", f"{gap:.4f}",
                      delta="極不穩定" if gap < 0.01 else "不穩定" if gap < 0.04 else "中等",
                      delta_color="normal" if gap >= 0.04 else "inverse")
        with i2:
            st.metric("Pool Rank", f"{rank} / 500")
            st.metric("Gate（P_ltp）", gate_disp)

        # gap = 0 特別說明
        if gap < 0.001:
            st.markdown(
                '<div class="danger-block">🔴 Gap to 2nd = 0.0000：'
                '第 1 名與第 2 名分數完全相同，代表模型對所有候選的評分幾乎無差別，'
                '推薦結果隨機性極高，此案例的 XAI 解釋參考價值低。</div>',
                unsafe_allow_html=True,
            )

        # 不確定性量尺
        np_ = max(5, min(95, int(90 - min(gap, 0.15) / 0.15 * 85)))
        st.markdown(
            f'<div class="xai-card" style="margin-top:10px;">'
            f'<div class="section-title">不確定性量尺（Score 差距）</div>'
            f'<div style="height:12px;border-radius:6px;'
            f'background:linear-gradient(90deg,#1D9E75,#EF9F27 50%,#E24B4A);'
            f'position:relative;margin:8px 0;">'
            f'<div style="position:absolute;top:-4px;left:{np_}%;width:4px;height:20px;'
            f'background:#3d3d3a;border-radius:2px;transform:translateX(-50%);"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#888780;">'
            f'<span>高信心（差距大）</span><span>低信心（差距小）</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("**指標解讀說明**")
        st.markdown("""
| 指標 | 閾值 | 意義 |
|---|---|---|
| BPR Score | 無絕對意義 | 只可跨樣本比較大小 |
| Gap to 2nd | >0.08 穩定 / <0.02 不穩定 | 分差越小越隨機 |
| Pool Rank | ≤10 好 / >50 差 | 相對學術指標 |
| Gate | <0.05 偏好貢獻低 | 缺 gate.pt 顯示 0.5 |
""")
        st.markdown(
            '<div class="xai-card"><div class="section-title">為什麼 Rank 不等於品質？</div>'
            '<ul style="font-size:12px;color:#3d3d3a;margin:0;padding-left:1.2rem;">'
            '<li>分數集中時 Rank 1 和 Rank 50 無實質差別</li>'
            '<li>499 首負例由 seed=20260315+idx 決定，不同 seed 結果不同</li>'
            '<li>500-pool 是學術評估指標，≠ 使用者滿意度</li>'
            '<li>NL 說明（generation）與排序（ranking）是不同 branch</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 頁籤 5：候選池比較
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    c1, c2 = st.columns([1.3, 1], gap="medium")

    with c1:
        st.markdown("**500-pool 分數分佈（真實）**")
        st.plotly_chart(pool_chart(scores_500, bpr),
                        use_container_width=True, key="pool_dist")

        std_val = float(np.std(scores_500)) if scores_500 else 0.0
        if std_val < 0.05:
            st.markdown(
                f'<div class="warn-block">⚠️ 分佈集中（std={std_val:.3f}）：'
                '排名差異不穩定，此次結果可信度低。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ok-block">✅ 分佈分散（std={std_val:.3f}）：'
                '推薦曲有明顯分差。</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Pool seed = 20260315 + {pool_seed_idx}，負例固定可重現。"
            f"GT pair_key 固定在 pool index 0，排除同 video_id（{video_id}）的所有 pairs。"
        )

    with c2:
        st.markdown("**前 5 名候選**")
        # 縮短 pair_key 顯示，避免表格欄位被截斷
        if top5:
            rows = []
            for i, item in enumerate(top5):
                pk = item[0] if isinstance(item, (list, tuple)) else item.get("pair_key", "")
                sc = item[1] if isinstance(item, (list, tuple)) else item.get("score", 0)
                rows.append({
                    "#": i + 1,
                    "pair_key（前 11 碼）": pk[:11],  # 截短，避免表格過寬
                    "BPR": round(float(sc), 3),
                    "GT": "✅" if pk == gt_mid else "",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # 若 GT 不在前 5 名，顯示 GT 的實際排名
            gt_in_top5 = any(
                (item[0] if isinstance(item, (list, tuple)) else item.get("pair_key")) == gt_mid
                for item in top5
            )
            if not gt_in_top5:
                st.markdown(
                    f'<div class="warn-block">GT 未出現在前 5 名，實際排名第 {rank}。'
                    f'GT BPR = {bpr:.4f}。</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("前 5 名候選資料尚未取得")

        st.caption(
            "pair_key 格式 = 23碼（即 gt_music_id）。\n"
            "排名依真實 BPR Score 降冪排列（pointwise scoring，全域可比）。"
        )