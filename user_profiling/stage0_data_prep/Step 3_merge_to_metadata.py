"""
用途：合併 MusicNN、YouTube 與人工整理欄位，產生統一 metadata。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

# 原始檔名：merge_to_metadata.py
import json
import jsonlines
from collections import defaultdict

tags_file = "data/user_profiling/music_metadata_simple/musicnn_tags_raw.jsonl"
yt_file = "data/user_profiling/music_metadata_simple/youtube_metadata.jsonl"
output = "data/user_profiling/music_metadata_simple/music_metadata.json"

yt_data = {obj["music_id"]: obj for obj in jsonlines.open(yt_file)}

metadata = {}
genres = {"pop", "rock", "hip-hop", "electronic", "jazz", "classical", "metal", "r&b", "country"}

for obj in jsonlines.open(tags_file):
    mid = obj["music_id"]
    tags = [t["tag"] for t in obj["tags_msd"] + obj["tags_mtt"]]
    genre = next((t for t in tags if t in genres), "unknown")
    
    yt = yt_data.get(mid, {})
    metadata[mid] = {
        "genre": genre,
        "artist": yt.get("artist", "unknown"),
        "title": yt.get("title", "unknown"),
        "tags": tags
    }

with open(output, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"生成完成！共 {len(metadata)} 首 → {output}")
