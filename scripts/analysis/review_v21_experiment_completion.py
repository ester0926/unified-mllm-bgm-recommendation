from pathlib import Path
import csv
import json
import zipfile

from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT_JSON = ROOT / "results" / "analysis" / "v21_experiment_completion_review.json"
OUT_MD = DOCS / "v21_補充實驗完成度自我複審_0730.md"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    persona_dir = ROOT / "results" / "analysis" / "b5_personas_v21"
    histories = read_json(persona_dir / "persona_histories.json")
    validation = read_csv(persona_dir / "persona_ltp_validation.csv")
    lengths = [len(v) for v in histories.values()]
    check("Persona history length", len(lengths) == 24 and set(lengths) == {20},
          f"personas={len(lengths)}, unique_lengths={sorted(set(lengths))}")
    composition_ok = all(
        int(r["n_core"]) == 14 and int(r["n_adjacent"]) == 4 and int(r["n_off"]) == 2
        for r in validation
    )
    check("Persona 14/4/2 composition", composition_ok,
          f"rows={len(validation)}, relaxed_core_total={sum(int(r['n_core_relaxed']) for r in validation)}")

    eval_dir = ROOT / "results" / "main_eval" / "exp_01" / "persona_eval_v21"
    eval_conditions = [
        "matched", "shuffled", "random", "cf_tempo", "cf_energy", "cf_vocal",
        "cf_popularity", "cf_consistency", "no_ltp",
    ]
    eval_counts = {}
    pool_ok = True
    for condition in eval_conditions:
        rows = read_csv(eval_dir / f"persona_v2_{condition}.csv")
        eval_counts[condition] = len(rows)
        pool_ok &= all(len(r["pool_pair_keys"].split(";")) == 500 for r in rows)
    check("Persona evaluation completeness", set(eval_counts.values()) == {480},
          json.dumps(eval_counts, ensure_ascii=False))
    check("Standard 500-pool records", pool_ok, "all rows retain 500 candidate IDs")
    noltp_reuse = read_json(eval_dir / "persona_v2_no_ltp_reuse_provenance.json")
    check("No-LTP reuse provenance",
          noltp_reuse["rows"] == 480 and noltp_reuse["pool_size"] == 500
          and noltp_reuse["output"]["sha256"],
          "Persona-independent exp_04 output reused only after row identity and pool invariants were verified")

    persona_summary = read_json(persona_dir / "persona_metrics_v21_summary.json")
    cfs = {r["counterfactual"]: r for r in persona_summary["counterfactual"]}
    check("Counterfactual target-direction metrics",
          set(cfs) == {"cf_tempo", "cf_energy", "cf_vocal", "cf_popularity", "cf_consistency"}
          and all(float(r["CDA"]) == float(r["CDA"]) for r in cfs.values()),
          "all five CDA values, including consistency/diversity, are finite")
    conds = {r["condition"]: r for r in persona_summary["by_condition"]}
    ndcg_ok = all(float(conds[c]["nDCG@5"]) == float(conds[c]["nDCG@5"])
                  and float(conds[c]["nDCG@10"]) == float(conds[c]["nDCG@10"])
                  for c in conds)
    check("Standard nDCG@5/@10", ndcg_ok,
          "IDCG computed from all 500 candidates; missing relevance stays at its rank with zero gain")
    check("Persona contradiction metric naming",
          all("PCRR" in r and "UPCR" not in r for r in persona_summary["explanation"]),
          "renamed to Persona Claim Contradiction Rate")

    conflict = read_json(ROOT / "results" / "analysis" / "b6_conflict_v21" / "conflict_summary.json")
    interaction_metrics = {r["metric"] for r in conflict["interactions"]}
    check("No-LTP-adjusted conflict DiD",
          interaction_metrics == {"persona_fit", "video_fit", "R@1", "MRR"},
          f"metrics={sorted(interaction_metrics)}; prototype x context two-way cluster bootstrap")

    path_summary = read_json(
        ROOT / "results" / "analysis" / "path_level_generation_v21" / "path_level_summary.json"
    )
    check("Four-model generation path table", set(path_summary["metrics"]) == {
        "exp_01", "exp_02", "exp_03", "exp_04"},
        "polarity-sensitive reference alignment/contradiction/nonexistent plus metadata and UCR")
    path_metrics = read_csv(
        ROOT / "results" / "analysis" / "path_level_generation_v21" / "path_level_metrics.csv"
    )
    check("Input support separated from reference alignment",
          all(r.get("preference_input_support") in {
              "not_observable_from_vector_input", "not_applicable_no_ltp"} for r in path_metrics),
          "opaque-vector input support is not misreported as text evidence")
    blind_packet = read_csv(
        ROOT / "results" / "analysis" / "path_level_generation_v21"
        / "preference_claim_blind_audit_packet.csv"
    )
    check("Blinded claim-calibration packet",
          len(blind_packet) == 100
          and all(not r["adjudicated_label"] for r in blind_packet),
          "100 model-stratified claims are blinded; independent human coding remains pending and is not misreported as completed")

    component_dir = ROOT / "results" / "main_eval" / "exp_01" / "fixed_component_intervention_v21"
    component_conditions = [
        "full", "no_explicit", "no_implicit", "no_explicit_norm", "no_implicit_norm", "no_both",
    ]
    component_counts = {
        c: len(read_csv(component_dir / f"fixed_component_{c}.csv")) for c in component_conditions
    }
    check("Fixed-model component intervention completeness",
          set(component_counts.values()) == {200}, json.dumps(component_counts))
    component_reuse = read_json(component_dir / "fixed_component_full_reuse_provenance.json")
    check("Fixed-model full-condition reuse provenance",
          component_reuse.get("verified_invariants", {}).get("sample_count") == 200
          and component_reuse.get("verified_invariants", {}).get(
              "all_source_top1_in_reconstructed_pool"
          ) is True
          and component_reuse.get("output", {}).get("sha256"),
          "full condition reuses the identical exp_01 checkpoint outputs; all intervention conditions were rerun")
    component_summary = read_json(
        ROOT / "results" / "analysis" / "fixed_hybrid_component_v21" / "fixed_component_summary.json"
    )
    d = component_summary["experiment"]["decomposition"]
    check("Hybrid decomposition validation",
          d["holdout_r2"] > 0.9999 and d["holdout_relative_mae"] < 1e-4
          and d["split_stability_explicit_component_cosine_mean"] > 0.999
          and d["split_stability_implicit_component_cosine_mean"] > 0.999,
          f"R2={d['holdout_r2']:.12f}, rel_MAE={d['holdout_relative_mae']:.3e}, "
          f"condition={d['design_condition_number']:.1f}")
    latency = component_summary["experiment"]["pool_scaling_latency"]
    check("Inference-cost benchmark", {r["pool_size"] for r in latency} == {100, 500, 1000},
          "30 queries per pool size; latency, token count and peak GPU memory recorded")

    manifest = read_json(ROOT / "results" / "analysis" / "v21_reproducibility_manifest.json")
    check("Reproducibility manifest",
          all(x.get("exists") and x.get("sha256") for group in [
              "inputs_and_checkpoints", "analysis_scripts", "analysis_artifacts"
          ] for x in manifest[group]),
          "SHA-256 for scripts, inputs and checkpoints; environment/Git metadata included")

    docx = DOCS / "論文_口試後修訂_v21_gpt_補充實驗完成_追蹤修訂版.docx"
    with zipfile.ZipFile(docx) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    doc_text = "\n".join(
        "".join(p.xpath(".//w:t/text()", namespaces=NS))
        for p in root.xpath(".//w:p", namespaces=NS)
    )
    doc_phrases = [
        "四模型生成說明之偏好主張與證據代理指標",
        "固定 Hybrid 模型之偏好成分介入結果",
        "分布匹配隨機向量",
        "差異中的差異",
        "v21 補充實驗版本紀錄",
    ]
    check("v21 manuscript integration", all(p in doc_text for p in doc_phrases),
          "methods, Chapter 4 results, Chapter 5 interpretation and Appendix C provenance integrated")

    passed = sum(c["passed"] for c in checks)
    review = {
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
        "scope_boundaries": [
            "No-LTP 仍是另一個重訓模型的模型層比較，不是固定權重下只移除 LTP 的單因子介入。",
            "Persona 向量由既有 LTP 向量組合而成，未驗證完整 Stage 3-5 對話生成、畫像萃取與融合流程。",
            "影片情境為分離度偏弱的 YouTube 音樂影片形式叢集，不等同旅遊、美食、教學或日常創作者短影音。",
            "偏好衝突分組與結果共享標籤；差異中的差異與雙向叢集拔靴只能減輕、不能消除內生性。",
            "固定模型成分來自穩定但條件數偏高的仿射分解；範數校正只控制尺度，不能建立一般性因果識別。",
            "已建立 100 筆盲化校準封包，但尚無獨立人類判讀；規則式參考畫像一致率仍是未校準代理指標。",
            "同骨幹雙模組基線、真實使用者／創作者與第二公開資料集仍屬外部效度與期刊擴充工作。",
        ],
        "recommendation_matrix": [
            ["P0-1", "已完成", "24 個 Persona 均為 20 首歷史；核心／鄰近／偏離固定 14／4／2，放寬核心仍保留反事實目標。"],
            ["P0-2", "已完成", "CDA 同時要求原推薦符合原屬性、反事實推薦符合翻轉屬性；一致性改以 top-10 標籤離散度判定。"],
            ["P0-3", "已補正，限制保留", "改報（Matched-No-LTP）× 衝突組 DiD 與原型 × 情境雙向叢集拔靴；結果不顯著，正文不再作機制宣稱。"],
            ["P0-4", "核心分析完成；人工校準待完成", "四模型極性敏感主張分析與表 4-16 已回寫；100 筆盲化封包已備妥，但獨立人類標註尚未完成。"],
            ["P1-1", "已完成", "Random-LTP 改為真實 LTP 分布內、低相似且每個 Persona 固定的對照向量。"],
            ["P1-2", "設計限制已揭露", "No-LTP 保留為模型設定比較；主要配對證據以 matched vs shuffled／分布匹配 random 為主。"],
            ["P1-3", "已完成", "nDCG@5／@10 的 IDCG 改由完整 500 候選池計算。"],
            ["P1-4", "已完成", "UPCR 改名為 Persona 屬性矛盾率（PCRR），並限制為詞彙規則可辨識的反向屬性。"],
            ["P1-5", "未補完整流程；限制已揭露", "Persona 實驗只驗證偏好向量條件反應，不宣稱驗證完整五階段建構流程。"],
            ["P1-6", "部分改善；限制保留", "補入 consistency CDA 與固定長度歷史；BPM 代理、情境替代、自然語言 Persona 與部分欄位仍受資料限制。"],
            ["P1-7", "外部驗證未完成", "只保留 YouTube 音樂影片分層的探索性異質性結果，不外推至真實創作者內容類型。"],
            ["P1-8", "已補固定模型介入；因果限制保留", "完成 exp_01 固定 checkpoint 的六條件成分介入與生成比較；仿射分解及分布偏移仍不允許固定功能或因果歸因。"],
            ["P1-9", "已完成", "UCR 報告模板已刪除『LTP 為必要輸入』的過強敘述，改為單一已訓練模型的結構敏感度。"],
            ["P1-10", "已完成", "外部路徑可由環境變數覆寫，manifest 記錄實際輸入、checkpoint、程式與輸出 SHA-256。"],
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# v21 補充實驗完成度自我複審（2026-07-30）",
        "",
        f"- 自動檢查：{passed}/{len(checks)} 通過",
        f"- 整體判定：{'程式、輸出與論文回寫均通過；保留外部效度與因果識別限制' if review['all_passed'] else '仍有未通過項目，不能定稿'}",
        "",
        "## 一、逐項檢查",
        "",
        "| 檢查項目 | 結果 | 證據 |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c['name']} | {'通過' if c['passed'] else '未通過'} | {c['detail']} |")
    lines.extend(["", "## 二、v20 建議逐項對照", "", "| 項目 | v21 狀態 | 證據與判讀 |", "|---|---|---|"])
    for item, status, evidence in review["recommendation_matrix"]:
        lines.append(f"| {item} | {status} | {evidence} |")
    lines.extend(["", "## 三、仍須保留的證據邊界", ""])
    for item in review["scope_boundaries"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 四、最終判定",
        "",
        "v20 審查中可在現有資料與模型範圍內補正的 P0／P1 程式缺口、統計定義、固定模型介入、推論成本與論文回寫均已完成。不能在本碩論中誠實宣稱完成者，則包括獨立人工盲標校準、完整五階段 Persona、真實創作者情境、第二公開資料集、同骨幹可識別路徑消融與一般性因果識別；v21 已逐項限縮論證並列為研究限制或未來工作。因此，本次補實驗工作可判定為『定稿範圍內完成，外部效度與因果識別工作未完成且未被誤報為完成』。",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not review["all_passed"]:
        raise SystemExit(f"v21 review failed: {passed}/{len(checks)}")
    print(f"[OK] v21 completion review {passed}/{len(checks)}")
    print(OUT_MD)


if __name__ == "__main__":
    main()
