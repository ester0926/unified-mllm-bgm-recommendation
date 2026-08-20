"""
用途：建立或分析 persona 條件下的 LTP 與評估結果。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import csv
import datetime as _dt
import json
import os
import random
from collections import defaultdict

import numpy as np

from scripts.analysis import b5_build_persona_specs as SPEC


OUT_DIR = Path(os.environ.get(
    "B5_PERSONA_OUT_DIR",
    PROJECT_ROOT / "results" / "analysis" / "b5_personas_v21",
))
CACHE_DIR = PROJECT_ROOT / "cache"
LTP_NPY   = CACHE_DIR / "ltp_hybrid.npy"
LTP_IDS   = CACHE_DIR / "ltp_hybrid_ids.json"
BANK_NPY  = CACHE_DIR / "song_bank.npy"
BANK_IDS  = CACHE_DIR / "song_bank_ids.json"

HISTORY_LEN = 20
MIX_COUNTS = {"core": 14, "adjacent": 4, "off": 2}
SEED = 20260726

# consistency → core 層取樣半徑（取最近質心的前 R 倍候選再抽樣）
RADIUS = {"very_high": 1.5, "high": 2.0, "medium": 4.0, "low": 1e9}

CF_FLIPS = ["cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"]


def log(msg):
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_video_ast():
    """video_id → AST 向量（沿用 stage5 的先到先存雙索引語義）。"""
    bank = np.load(BANK_NPY).astype(np.float32)
    ids = json.loads(BANK_IDS.read_text(encoding="utf-8"))
    out = {}
    for i, pk in enumerate(ids):
        parts = pk.rsplit("_", 1)
        if len(parts) == 2:
            out.setdefault(parts[0], bank[i])
            out.setdefault(parts[1], bank[i])
        out.setdefault(pk, bank[i])
    return out


def flip_spec(spec: dict, flip: str) -> dict:
    """回傳翻轉單一屬性後的規格副本。"""
    s = json.loads(json.dumps(spec))
    if flip == "cf_tempo":
        s["tempo"] = {"fast": "slow", "slow": "fast"}.get(s["tempo"], "slow")
    elif flip == "cf_energy":
        s["energy"] = {"high": "low", "low": "high"}.get(s["energy"], "low")
    elif flip == "cf_vocal":
        s["vocal"] = {"vocal_required": "instrumental_leaning",
                      "instrumental_leaning": "vocal_required"}.get(
                          s["vocal"], "instrumental_leaning")
    elif flip == "cf_popularity":
        s["popularity"] = {"mainstream": "niche", "niche": "mainstream"}.get(
            s["popularity"], "niche")
    elif flip == "cf_consistency":
        s["consistency"] = "low" if s["consistency"] in ("very_high", "high") else "very_high"
        s["novelty"] = "high" if s["novelty"] in ("very_low", "low") else "low"
    return s


def select_history(spec, md, views, p25, p75, ast, ltp_ids_set, rng, variant="matched"):
    """
    依 70/20/10 抽 20 首。回傳 [(music_id, layer)]。
    反事實嚴格核心池不足時，以符合翻轉目標屬性為必要條件，
    依其餘屬性符合數排序補足，避免歷史長度成為混淆變項。
    只挑選**有既有 LTP 向量**的曲目（Plan B 的前提）。
    """
    def core_pool():
        return [m for m, e in md.items()
                if m in ltp_ids_set and m in ast
                and SPEC.matches_core(SPEC.track_tags(e), views.get(m), spec, p25, p75)]

    def adjacent_pool(core_set):
        """符合偏好曲風，但放寬節奏／能量其中一項。"""
        relaxed = json.loads(json.dumps(spec))
        relaxed["tempo"] = "any"
        relaxed["energy"] = "any"
        out = []
        for m, e in md.items():
            if m in core_set or m not in ltp_ids_set or m not in ast:
                continue
            if SPEC.matches_core(SPEC.track_tags(e), views.get(m), relaxed, p25, p75):
                out.append(m)
        return out

    def off_pool():
        """違反核心偏好：命中排斥曲風，或完全不含偏好曲風。"""
        pref, rej = set(spec["preferred_genres"]), set(spec["rejected_genres"])
        out = []
        for m, e in md.items():
            if m not in ltp_ids_set or m not in ast:
                continue
            tags = SPEC.track_tags(e)
            if (rej and (tags & rej)) or (pref and not (tags & pref)):
                out.append(m)
        return out

    n_core = MIX_COUNTS["core"]
    n_adj = MIX_COUNTS["adjacent"]
    n_off = MIX_COUNTS["off"]

    core = core_pool()
    if not core:
        return []

    # consistency：先取一個種子，再依 AST 距離決定取樣半徑
    seed_id = rng.choice(core)
    seed_vec = ast[seed_id]
    seed_n = seed_vec / (np.linalg.norm(seed_vec) + 1e-8)
    sims = np.array([float(np.dot(ast[m] / (np.linalg.norm(ast[m]) + 1e-8), seed_n))
                     for m in core])
    order = np.argsort(-sims)
    radius = RADIUS.get(spec["consistency"], 4.0)
    take = min(len(core), max(n_core, int(n_core * radius)))
    core_cand = [core[i] for i in order[:take]]
    core_sel = rng.sample(core_cand, min(n_core, len(core_cand)))

    def target_attr_ok(m):
        if variant in ("matched", "cf_consistency"):
            return True
        tags = SPEC.track_tags(md[m])
        v = views.get(m)
        if variant == "cf_tempo":
            return spec["tempo"] == "any" or spec["tempo"] in tags
        if variant == "cf_energy":
            return ("loud" in tags) if spec["energy"] == "high" else ("loud" not in tags)
        if variant == "cf_vocal":
            has_vocal = bool(tags & SPEC.VOCAL_TAGS)
            return has_vocal if spec["vocal"] == "vocal_required" else not has_vocal
        if variant == "cf_popularity":
            if v is None:
                return False
            return v >= p75 if spec["popularity"] == "mainstream" else v <= p25
        return True

    def soft_score(m):
        tags = SPEC.track_tags(md[m])
        v = views.get(m)
        score = 3 * int(bool(tags & set(spec["preferred_genres"])))
        score -= 3 * int(bool(tags & set(spec["rejected_genres"])))
        if spec["tempo"] != "any":
            score += int(spec["tempo"] in tags)
        if spec["energy"] != "any":
            score += int(("loud" in tags) == (spec["energy"] == "high"))
        if spec["vocal"] != "any":
            score += int(bool(tags & SPEC.VOCAL_TAGS) ==
                         (spec["vocal"] == "vocal_required"))
        if spec["popularity"] == "mainstream":
            score += int(v is not None and v >= p75)
        elif spec["popularity"] == "niche":
            score += int(v is not None and v <= p25)
        return score

    n_core_strict = len(core_sel)
    if len(core_sel) < n_core:
        eligible = [m for m in ltp_ids_set
                    if m in md and m in ast and m not in core_sel and target_attr_ok(m)]
        eligible.sort(key=lambda m: (soft_score(m), m), reverse=True)
        need = n_core - len(core_sel)
        window = eligible[:max(need * 8, need)]
        if len(window) < need:
            raise RuntimeError(f"{variant} 核心池不足：需要 {need}，僅有 {len(window)}")
        core_sel.extend(rng.sample(window, need))

    core_set = set(core_sel)
    adj = adjacent_pool(core_set)
    if len(adj) < n_adj:
        raise RuntimeError(f"{variant} adjacent 池不足：需要 {n_adj}，僅有 {len(adj)}")
    adj_sel = rng.sample(adj, n_adj)

    used = core_set | set(adj_sel)
    off = [m for m in off_pool() if m not in used]
    if len(off) < n_off:
        raise RuntimeError(f"{variant} off 池不足：需要 {n_off}，僅有 {len(off)}")
    off_sel = rng.sample(off, n_off)

    return ([(m, "core" if idx < n_core_strict else "core_relaxed")
             for idx, m in enumerate(core_sel)]
            + [(m, "adjacent") for m in adj_sel]
            + [(m, "off") for m in off_sel])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    specs_doc = json.loads((OUT_DIR / "persona_specs.json").read_text(encoding="utf-8"))
    personas = specs_doc["personas"]
    prototypes = specs_doc["prototypes"]
    p25 = specs_doc["view_count_quantiles"]["p25"]
    p75 = specs_doc["view_count_quantiles"]["p75"]

    log("載入 metadata 與向量…")
    md, views = SPEC.load_metadata()
    ltp = np.load(LTP_NPY).astype(np.float32)
    ltp_ids = json.loads(LTP_IDS.read_text(encoding="utf-8"))
    ltp_map = {m: i for i, m in enumerate(ltp_ids)}
    ltp_ids_set = set(ltp_map)
    ast = build_video_ast()
    log(f"LTP={ltp.shape}　AST 索引={len(ast)}　metadata={len(md)}")

    real_norm = np.linalg.norm(ltp, axis=1)
    lo_n, hi_n = np.percentile(real_norm, [5, 95])
    ltp_unit = ltp / (np.linalg.norm(ltp, axis=1, keepdims=True) + 1e-8)

    # 偏差尺度校正所需的兩個統計量（見檔頭說明）
    mu = ltp.mean(axis=0)
    target_dev = float(np.median(np.linalg.norm(ltp - mu, axis=1)))
    log(f"全域平均 norm={np.linalg.norm(mu):.4f}　真實偏差 norm 中位數 d*={target_dev:.4f}")

    histories, vectors, validation = {}, {}, []

    for p in personas:
        pid = p["persona_id"]
        base_spec = prototypes[p["prototype_id"]]

        variants = {"matched": base_spec}
        for flip in CF_FLIPS:
            variants[flip] = flip_spec(base_spec, flip)

        for vname, spec in variants.items():
            sel = select_history(spec, md, views, p25, p75, ast, ltp_ids_set, rng,
                                 variant=vname)
            if len(sel) != HISTORY_LEN:
                raise RuntimeError(f"{pid}/{vname} 歷史長度 {len(sel)} != {HISTORY_LEN}")
            vecs = np.stack([ltp[ltp_map[m]] for m, _ in sel])
            raw = vecs.mean(axis=0)
            # 偏差尺度校正：方向由曲目決定，幅度還原到真實分布尺度
            dev = raw - mu
            dev_norm = float(np.linalg.norm(dev))
            pv = mu + dev * (target_dev / dev_norm) if dev_norm > 1e-8 else raw.copy()
            key = pid if vname == "matched" else f"{pid}::{vname}"
            vectors[key] = pv
            if vname == "matched":
                histories[pid] = [{"music_id": m, "layer": lay} for m, lay in sel]

            n = float(np.linalg.norm(pv))
            pu = pv / (n + 1e-8)
            nn_sim = float((ltp_unit @ pu).max())
            validation.append({
                "persona_id": pid, "variant": vname,
                "n_tracks": len(sel),
                "n_core": sum(1 for _, l in sel if l in ("core", "core_relaxed")),
                "n_core_strict": sum(1 for _, l in sel if l == "core"),
                "n_core_relaxed": sum(1 for _, l in sel if l == "core_relaxed"),
                "n_adjacent": sum(1 for _, l in sel if l == "adjacent"),
                "n_off": sum(1 for _, l in sel if l == "off"),
                "norm": n,
                "norm_in_real_p5_p95": bool(lo_n <= n <= hi_n),
                "cos_to_nearest_real_ltp": nn_sim,
                "raw_dev_norm": dev_norm,
                "scale_applied": target_dev / dev_norm if dev_norm > 1e-8 else 1.0,
            })

    log(f"已產生 {len(vectors)} 條向量（{len(personas)} Persona × {1 + len(CF_FLIPS)} 變體）")

    # ---- 分布驗證 -----------------------------------------------------------
    keys = list(vectors)
    V = np.stack([vectors[k] for k in keys])
    Vu = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    matched_keys = [k for k in keys if "::" not in k]
    Mu = np.stack([vectors[k] for k in matched_keys])
    Mu = Mu / (np.linalg.norm(Mu, axis=1, keepdims=True) + 1e-8)
    pair = Mu @ Mu.T
    np.fill_diagonal(pair, np.nan)

    cf_shift = defaultdict(list)
    for k in keys:
        if "::" not in k:
            continue
        base, flip = k.split("::")
        if base in vectors:
            a = vectors[base] / (np.linalg.norm(vectors[base]) + 1e-8)
            b = vectors[k] / (np.linalg.norm(vectors[k]) + 1e-8)
            cf_shift[flip].append(float(np.dot(a, b)))

    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_personas": len(personas), "n_vectors": len(vectors),
        "deviation_scaling": {
            "global_mean_norm": float(np.linalg.norm(mu)),
            "target_deviation_norm": target_dev,
            "note": "Persona = μ + dev × (d*/‖dev‖)；方向由曲目決定，"
                    "僅將偏差幅度還原至真實分布尺度（前二階動差對齊）",
        },
        "real_pairwise_cos_median": float(np.median(
            (lambda U: (U @ U.T)[np.triu_indices(U.shape[0], 1)])(
                ltp_unit[np.random.default_rng(7).choice(ltp.shape[0], 2000, replace=False)]))),
        "real_ltp_norm_p5_p95": [float(lo_n), float(hi_n)],
        "real_ltp_norm_median": float(np.median(real_norm)),
        "persona_norm_median": float(np.median([v["norm"] for v in validation])),
        "share_norm_in_range": float(np.mean([v["norm_in_real_p5_p95"] for v in validation])),
        "cos_to_nearest_real_min": float(min(v["cos_to_nearest_real_ltp"] for v in validation)),
        "cos_to_nearest_real_median": float(np.median(
            [v["cos_to_nearest_real_ltp"] for v in validation])),
        "pairwise_cos_between_personas": {
            "median": float(np.nanmedian(pair)), "max": float(np.nanmax(pair)),
            "min": float(np.nanmin(pair)),
        },
        "counterfactual_cos_to_matched": {
            f: {"median": float(np.median(v)), "min": float(np.min(v)), "n": len(v)}
            for f, v in cf_shift.items()
        },
    }

    log("-" * 70)
    log(f"真實 LTP norm 中位數 {summary['real_ltp_norm_median']:.4f}"
        f"（p5–p95 {lo_n:.2f}–{hi_n:.2f}）")
    log(f"Persona norm 中位數 {summary['persona_norm_median']:.4f}"
        f"，落在 p5–p95 內 {summary['share_norm_in_range']*100:.1f}%")
    log(f"與最近真實 LTP 的餘弦：中位數 {summary['cos_to_nearest_real_median']:.4f}"
        f"，最低 {summary['cos_to_nearest_real_min']:.4f}")
    pc = summary["pairwise_cos_between_personas"]
    log(f"Persona 兩兩餘弦：中位數 {pc['median']:.4f}，最高 {pc['max']:.4f}"
        f"（真實 LTP 為 {summary['real_pairwise_cos_median']:.4f}，愈接近愈好）")
    for f, s in summary["counterfactual_cos_to_matched"].items():
        log(f"  反事實 {f:16s} 與原向量餘弦 中位數 {s['median']:.4f}（最低 {s['min']:.4f}）")

    # ---- 輸出 ---------------------------------------------------------------
    np.savez_compressed(OUT_DIR / "persona_ltp.npz",
                        keys=np.array(keys, dtype=object), vectors=V.astype(np.float32))
    (OUT_DIR / "persona_histories.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT_DIR / "persona_ltp_validation.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(validation[0].keys()))
        w.writeheader()
        w.writerows(validation)
    (OUT_DIR / "persona_ltp_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# B5 步驟 2–4：Persona LTP 建構與分布驗證\n",
        f"- 產生時間：{summary['generated_at']}",
        f"- {summary['n_personas']} 個 Persona × 6 變體（matched + 5 種向量層級反事實）"
        f" = {summary['n_vectors']} 條向量\n",
        "## 一、合成歷史\n",
        "每個 Persona 20 首，分層為 14 core / 4 adjacent / 2 off；"
        "刻意保留不一致樣本以免測試過易。core 層的取樣半徑由 consistency 欄位控制。\n",
        "## 二、分布驗證（Plan B 的關鍵前提）\n",
        "| 檢查項 | 結果 |", "|---|---|",
        f"| 真實 LTP norm 中位數（p5–p95）| {summary['real_ltp_norm_median']:.4f} "
        f"（{lo_n:.2f}–{hi_n:.2f}）|",
        f"| Persona LTP norm 中位數 | {summary['persona_norm_median']:.4f} |",
        f"| Persona norm 落在真實 p5–p95 內 | **{summary['share_norm_in_range']*100:.1f}%** |",
        f"| 與最近真實 LTP 的餘弦（中位數／最低）| "
        f"{summary['cos_to_nearest_real_median']:.4f} ／ "
        f"{summary['cos_to_nearest_real_min']:.4f} |",
        f"| Persona 之間兩兩餘弦（中位數／最高）| {pc['median']:.4f} ／ {pc['max']:.4f} |",
        "\n## 三、反事實位移\n",
        "| 翻轉項 | 與原向量餘弦（中位數）| 最低 | n |", "|---|---|---|---|",
    ]
    for f, s in summary["counterfactual_cos_to_matched"].items():
        md_lines.append(f"| {f} | {s['median']:.4f} | {s['min']:.4f} | {s['n']} |")
    md_lines.append("\n> 餘弦愈低代表翻轉造成的位移愈大。若某翻轉項的餘弦接近 1，"
                    "代表該屬性在本曲庫中無法有效區分，對應的反事實條件不應納入結論。\n")
    (OUT_DIR / "persona_ltp_validation.md").write_text("\n".join(md_lines) + "\n",
                                                       encoding="utf-8")
    log(f"已輸出至 {OUT_DIR}")


if __name__ == "__main__":
    main()
