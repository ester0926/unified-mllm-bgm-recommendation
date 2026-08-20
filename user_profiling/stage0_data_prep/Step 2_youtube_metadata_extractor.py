"""
用途：擷取 YouTube 影片 metadata，供後續偏好建模使用。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

# 原始檔名：youtube_metadata_extractor.py
# 環境：Python 3.9+ + yt-dlp
import yt_dlp
import jsonlines
from pathlib import Path
from tqdm import tqdm
import logging
import time
import random

INPUT_FILE = Path("data/user_profiling/music_metadata_simple/musicnn_tags_raw.jsonl")
OUTPUT_FILE = Path("data/user_profiling/music_metadata_simple/youtube_metadata.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_metadata(music_id):
    url = f"https://www.youtube.com/watch?v={music_id}"
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info:
                return {
                    "music_id": music_id,
                    "title": info.get('title'),
                    "artist": info.get('uploader'),
                    "album": info.get('album'),
                    "release_date": info.get('release_date') or info.get('upload_date'),
                    "duration": info.get('duration'),
                    "view_count": info.get('view_count'),
                    "youtube_url": url,
                }
    except:
        pass
    return {"music_id": music_id}

# 主流程
existing = {obj["music_id"] for obj in jsonlines.open(OUTPUT_FILE) if OUTPUT_FILE.exists()}
music_ids = [obj["music_id"] for obj in jsonlines.open(INPUT_FILE) if obj["music_id"] not in existing]

for mid in tqdm(music_ids, desc="yt-dlp"):
    result = get_metadata(mid)
    with jsonlines.open(OUTPUT_FILE, 'a') as f:
        f.write(result)
    time.sleep(random.uniform(0.1, 0.3))  # 防封鎖
