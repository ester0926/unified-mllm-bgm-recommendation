"""
用途：檢查資料切分是否有潛在資料洩漏。
輸入：既有實驗輸出、metadata、評估 CSV 或分析用中間檔。
輸出：論文分析用表格、圖表、摘要 JSON/CSV 或檢查清單。
執行：請先確認前一階段輸出檔已存在，再從 repo 根目錄執行。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import logging
import random
from typing import Dict, List, Set

import h5py
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_train_music_ids(train_h5_files: List[str]) -> Set[str]:
    """
    遍歷訓練集 HDF5 檔案，收集所有出現過的 music_id。

    Args:
        train_h5_files : 訓練集的 .h5 檔案路徑列表

    Returns:
        train_music_ids : 所有訓練集音樂 ID 的集合
    """
    train_music_ids: Set[str] = set()

    for h5_path in tqdm(train_h5_files, desc="Building train music ID set"):
        try:
            with h5py.File(h5_path, "r") as f:
                if "music_ids" in f:
                    ids = f["music_ids"][:]
                    for mid in ids:
                        if isinstance(mid, bytes):
                            train_music_ids.add(mid.decode("utf-8"))
                        else:
                            train_music_ids.add(str(mid))
                if "gt_music_id" in f:
                    gt_id = f["gt_music_id"][()]
                    if isinstance(gt_id, bytes):
                        train_music_ids.add(gt_id.decode("utf-8"))
                    else:
                        train_music_ids.add(str(gt_id))
        except Exception as e:
            logger.warning(f"無法讀取 {h5_path}: {e}")

    logger.info(f"訓練集唯一音樂 ID 數量: {len(train_music_ids)}")
    return train_music_ids


@torch.no_grad()
def evaluate_cold_start_music(
    model,
    test_dataset,
    all_music_features: torch.Tensor,   # (M, music_dim)
    all_music_ids: List[str],
    train_music_ids: Set[str],
    device: torch.device,
    train_config,
    model_config,
) -> Dict[str, float]:
    """
    Cold-start Music Subset Evaluation（Pointwise v2）。

    分別回報：
      - 整體測試集的 R@10（含訓練集已見音樂）
      - 冷啟動子集的 R@10（GT 為訓練集從未見過的音樂）
      - memorization_gap = 兩者之差（越小代表泛化能力越好）

    使用 pointwise_pool_scoring（與 evaluate.py 的主評估流程一致）。
    每首候選獨立 forward，score 全域可比，不依賴 batch 上下文。
    """
    # 延遲匯入以避免循環依賴
    from evaluate import pointwise_pool_scoring, compute_ranking_metrics

    model.eval()
    pool_size      = train_config.music_pool_size
    micro_batch_sz = getattr(train_config, "pointwise_eval_batch_size", 32)
    M = all_music_features.size(0)

    overall_r10: List[float] = []
    cold_r10:    List[float] = []

    for idx in tqdm(range(len(test_dataset)), desc="Cold-start Music Analysis"):
        sample      = test_dataset[idx]
        gt_music_id = sample.get("gt_music_id", "")
        video_id    = sample.get("video_id", "")
        is_cold     = (gt_music_id not in train_music_ids)

        video_feat     = sample["video_feat"].unsqueeze(0).to(device)
        ltp_feat       = sample["ltp_feat"].unsqueeze(0).to(device)
        text_feat      = sample["text_feat"].unsqueeze(0).to(device)
        input_ids      = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        # 找 GT 的 global index
        gt_global_idx = next(
            (i for i, sid in enumerate(all_music_ids) if sid == gt_music_id), 0
        )

        # 建構 500 首音樂池（GT 固定在 index 0）
        excl = {i for i, sid in enumerate(all_music_ids) if sid[:11] == video_id}
        candidates = [i for i in range(M) if i not in excl and i != gt_global_idx]
        negatives  = random.sample(candidates, min(pool_size - 1, len(candidates)))
        pool_idx   = [gt_global_idx] + negatives
        pool_feats = all_music_features[pool_idx].to(device)

        # Pointwise scoring：每首獨立計算，全域可比
        scores = pointwise_pool_scoring(
            model=model,
            video_feat=video_feat,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_music_features=pool_feats,
            micro_batch_size=micro_batch_sz,
            device=device,
        )

        r10 = compute_ranking_metrics(scores, gt_idx=0)["recall@10"]
        overall_r10.append(r10)
        if is_cold:
            cold_r10.append(r10)

    n_cold  = len(cold_r10)
    n_total = len(test_dataset)
    overall = float(np.mean(overall_r10)) if overall_r10 else 0.0
    cold    = float(np.mean(cold_r10))    if cold_r10    else 0.0
    gap     = overall - cold
    ratio   = n_cold / max(n_total, 1)

    results = {
        "overall_recall@10":    overall,
        "cold_start_recall@10": cold,
        "cold_start_count":     n_cold,
        "total_count":          n_total,
        "cold_start_ratio":     ratio,
        "memorization_gap":     gap,
    }

    # 結果輸出
    logger.info("=" * 60)
    logger.info("  Music-Level Leakage Analysis Results")
    logger.info("=" * 60)
    logger.info(f"  Overall R@10     : {overall:.4f}  (n={n_total})")
    logger.info(f"  Cold-start R@10  : {cold:.4f}  (n={n_cold}, {ratio:.1%} of test)")
    logger.info(f"  Memorization gap : {gap:+.4f}")

    if n_cold == 0:
        logger.warning(
            "  ⚠️  No cold-start samples found. All test GT music appeared in training."
            "  Consider checking if music-level leakage is a concern."
        )
    elif gap > 0.05:
        logger.warning(
            f"  ⚠️  Gap > 5% ({gap:.1%}). Possible music memorization effect."
            "  Recommend discussing as a limitation in the paper."
        )
    else:
        logger.info(
            f"  ✅  Gap ≤ 5% ({gap:.1%}). Model generalizes beyond seen music."
        )
    logger.info("=" * 60)

    return results