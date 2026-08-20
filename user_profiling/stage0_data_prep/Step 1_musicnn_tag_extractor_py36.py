"""
用途：從音訊檔抽取 MusicNN 標籤與音樂特徵。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

# 原始檔名：musicnn_tag_extractor_py36.py
# 環境：Python 3.6 + musicnn
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import jsonlines
import gc
import traceback
from datetime import datetime

# ========== 設定 ==========
AUDIO_FOLDER = Path("data/extracted_audio/audio")
OUTPUT_FILE = Path("data/user_profiling/music_metadata_simple/musicnn_tags_raw.jsonl")
VIDEO_IDS_JSON = "data/video_ids_from_hdf5_106files.json"

TAGS_PER_MODEL = 10
MIN_CONFIDENCE = 0.10
NUM_WORKERS = 6  # Python 3.6 建議少一點

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 載入 HDF5 中出現的 music_id ==========
def load_hdf5_music_ids(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ids = set(data.get('video_ids', []))
        for d in data.get('details', []):
            ids.update([d.get('target_music'), d.get('candidate_music')])
        return {i for i in ids if i}
    except Exception as e:
        logger.error(f"載入失敗: {e}")
        return set()

HDF5_IDS = load_hdf5_music_ids(VIDEO_IDS_JSON)
logger.info(f"載入 HDF5 音樂 ID: {len(HDF5_IDS)} 個")

# ========== 載入已處理 ==========
existing = set()
if OUTPUT_FILE.exists():
    try:
        with jsonlines.open(OUTPUT_FILE) as reader:
            for obj in reader:
                existing.add(obj["music_id"])
        logger.info(f"已跳過 {len(existing)} 個已處理檔案")
    except:
        pass

# ========== 單檔處理 ==========
def process_audio(music_id, audio_path):
    try:
        import musicnn.extractor
        result = {}

        # MSD
        try:
            r = musicnn.extractor.extractor(str(audio_path), model='MSD_musicnn',
                                         extract_features=False, input_length=3.0)
            if len(r) == 3:
                taggram, tags, _ = r
            else:
                taggram, tags = r
            preds = np.mean(taggram, axis=0)
            msd = [(tags[i], float(preds[i])) for i in range(len(tags)) if preds[i] >= MIN_CONFIDENCE]
            msd.sort(key=lambda x: x[1], reverse=True)
            result["tags_msd"] = [{"tag": t, "confidence": c} for t, c in msd[:TAGS_PER_MODEL]]
        except:
            result["tags_msd"] = []

        # MTT
        try:
            r = musicnn.extractor.extractor(str(audio_path), model='MTT_musicnn',
                                         extract_features=False, input_length=3.0)
            if len(r) == 3:
                taggram, tags, _ = r
            else:
                taggram, tags = r
            preds = np.mean(taggram, axis=0)
            mtt = [(tags[i], float(preds[i])) for i in range(len(tags)) if preds[i] >= MIN_CONFIDENCE]
            mtt.sort(key=lambda x: x[1], reverse=True)
            result["tags_mtt"] = [{"tag": t, "confidence": c} for t, c in mtt[:TAGS_PER_MODEL]]
        except:
            result["tags_mtt"] = []

        result.update({
            "music_id": music_id,
            "audio_path": str(audio_path),
            "processed_at": datetime.now().isoformat()
        })
        return result
    except Exception as e:
        logger.error(f"處理失敗 {music_id}: {e}")
        return None

# ========== 主流程 ==========
def main():
    audio_files = [p for p in AUDIO_FOLDER.glob("*.wav") if p.stem in HDF5_IDS and p.stem not in existing]
    logger.info(f"本次需處理: {len(audio_files)} 首")

    if not audio_files:
        logger.info("無新檔案，結束。")
        return

    args = [(p.stem, p) for p in audio_files]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_audio, mid, path) for mid, path in args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="musicnn"):
            result = future.result()
            if result:
                with jsonlines.open(OUTPUT_FILE, 'a') as f:
                    f.write(result)

    logger.info("musicnn 標籤提取完成！")

if __name__ == "__main__":
    main()
