# file: merge_to_metadata.py
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
