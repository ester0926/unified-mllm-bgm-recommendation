# file: Step_4j_reextract_two_stage.py
# 兩階段提取：先逐個提取音訊，再合併（避免 concat demuxer 問題）
import json
import jsonlines
import subprocess
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
import sys

# ========== 設定 ==========
CORRUPTED_IDS = [
    "oQ0Zu9agnsY",
    "oEkJ1ixdcdU", 
    "l2RQ4RAPP7k",
    "nYc4kfn0g_s",
    "mwNk65qHHmo"
]

# 路徑設定
AUDIO_FOLDER = Path("data/extracted_audio/audio")
VIDEO_SEGMENTS_DIR = Path("data/video_segments_10s")
BACKUP_FOLDER = Path("data/extracted_audio/audio_backup_corrupted")
TEMP_AUDIO_DIR = Path("data/extracted_audio/temp_audio_segments")

# JSON 檔案路徑
MUSICNN_TAGS_FILE = Path("data/user_profiling/music_metadata_simple/musicnn_tags_raw.jsonl")
YOUTUBE_METADATA_FILE = Path("data/user_profiling/music_metadata_simple/youtube_metadata.jsonl")
METADATA_FILE = Path("data/user_profiling/music_metadata_simple/music_metadata_fixed.json")

# 確保目錄存在
BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 尋找影片段 ==========
def find_segment_files(music_id):
    """尋找所有影片段檔案"""
    segment_files = []
    missing_segments = []
    
    for i in range(12):
        found = False
        possible_paths = [
            VIDEO_SEGMENTS_DIR / music_id / f"{music_id}_{i+1:02d}.mp4",
            VIDEO_SEGMENTS_DIR / music_id / f"{music_id}_{i}.mp4",
            VIDEO_SEGMENTS_DIR / music_id / f"{music_id}_{i+1}.mp4",
        ]
        
        for segment_path in possible_paths:
            if segment_path.exists():
                segment_files.append(segment_path)
                found = True
                break
        
        if not found:
            missing_segments.append(i)
    
    return segment_files, missing_segments

# ========== 兩階段音檔提取 ==========
def extract_audio_two_stage(music_id):
    """兩階段提取：先逐個提取音訊，再合併"""
    output_path = AUDIO_FOLDER / f"{music_id}.wav"
    
    # 尋找影片段
    segment_files, missing_segments = find_segment_files(music_id)
    
    if len(missing_segments) > 3:
        logger.warning(f"{music_id}: 缺失 {len(missing_segments)} 個段落，跳過")
        return False
    
    if not segment_files:
        logger.error(f"{music_id}: 沒有任何影片段可用")
        return False
    
    logger.info(f"{music_id}: 找到 {len(segment_files)}/12 個影片段")
    
    # 備份損壞的檔案
    if output_path.exists():
        backup_path = BACKUP_FOLDER / f"{music_id}.wav"
        try:
            shutil.move(str(output_path), str(backup_path))
            logger.info(f"已備份: {backup_path.name}")
        except Exception as e:
            logger.warning(f"備份失敗: {e}")
    
    # 建立臨時目錄
    temp_dir = TEMP_AUDIO_DIR / music_id
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # ========== 階段 1: 逐個提取音訊 ==========
        logger.info(f"{music_id}: [階段 1/2] 逐個提取音訊...")
        temp_audio_files = []
        
        for idx, video_segment in enumerate(segment_files):
            temp_audio = temp_dir / f"segment_{idx:02d}.wav"
            
            command = [
                'ffmpeg',
                '-i', str(video_segment),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '22050',
                '-ac', '1',
                '-y',
                str(temp_audio)
            ]
            
            result = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result == 0 and temp_audio.exists() and temp_audio.stat().st_size > 10000:
                temp_audio_files.append(temp_audio)
            else:
                logger.warning(f"  段落 {idx} 提取失敗")
        
        if not temp_audio_files:
            logger.error(f"{music_id}: 所有段落提取失敗")
            return False
        
        logger.info(f"  成功提取: {len(temp_audio_files)}/{len(segment_files)} 個段落")
        
        # ========== 階段 2: 合併音訊檔案 ==========
        logger.info(f"{music_id}: [階段 2/2] 合併音訊檔案...")
        
        # 建立合併清單
        concat_list = temp_dir / "concat_list.txt"
        with open(concat_list, 'w', encoding='utf-8') as f:
            for temp_audio in temp_audio_files:
                abs_path = temp_audio.absolute()
                path_str = str(abs_path).replace('\\', '/')
                f.write(f"file '{path_str}'\n")
        
        # 使用 concat demuxer 合併（這次是 WAV 檔案，不會有編碼問題）
        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',  # 直接複製，不重新編碼
            '-y',
            str(output_path)
        ]
        
        result = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result == 0 and output_path.exists():
            file_size = output_path.stat().st_size
            
            # 驗證檔案大小
            if file_size > 100000:
                logger.info(f"✓ {music_id}: 提取成功 ({file_size / 1024 / 1024:.2f} MB)")
                
                # 驗證音檔時長
                try:
                    import wave
                    with wave.open(str(output_path), 'rb') as wf:
                        duration = wf.getnframes() / wf.getframerate()
                        logger.info(f"  時長: {duration:.2f} 秒")
                        
                        if duration >= 10:
                            return True
                        else:
                            logger.warning(f"  音檔太短")
                            return False
                except Exception as e:
                    logger.warning(f"  無法驗證音檔: {e}")
                    return True  # 大小正常就算成功
            else:
                logger.warning(f"{music_id}: 檔案太小 ({file_size / 1024:.2f} KB)")
                return False
        else:
            logger.error(f"{music_id}: 合併失敗")
            return False
    
    except Exception as e:
        logger.error(f"{music_id}: 提取失敗 - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # 清理臨時檔案
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass

# ========== 重新生成 musicnn tags ==========
def regenerate_musicnn_tags(music_id):
    """重新生成 musicnn tags"""
    audio_path = AUDIO_FOLDER / f"{music_id}.wav"
    
    if not audio_path.exists():
        logger.error(f"{music_id}: 音檔不存在，無法生成 tags")
        return None
    
    try:
        import musicnn.extractor
        import numpy as np
        
        result = {
            "music_id": music_id,
            "audio_path": str(audio_path),
            "processed_at": datetime.now().isoformat()
        }
        
        # MSD model
        try:
            r = musicnn.extractor.extractor(str(audio_path), model='MSD_musicnn',
                                         extract_features=False, input_length=3.0)
            if len(r) == 3:
                taggram, tags, _ = r
            else:
                taggram, tags = r
            preds = np.mean(taggram, axis=0)
            msd = [(tags[i], float(preds[i])) for i in range(len(tags)) if preds[i] >= 0.10]
            msd.sort(key=lambda x: x[1], reverse=True)
            result["tags_msd"] = [{"tag": t, "confidence": c} for t, c in msd[:10]]
        except Exception as e:
            logger.warning(f"{music_id}: MSD 提取失敗 - {e}")
            result["tags_msd"] = []
        
        # MTT model
        try:
            r = musicnn.extractor.extractor(str(audio_path), model='MTT_musicnn',
                                         extract_features=False, input_length=3.0)
            if len(r) == 3:
                taggram, tags, _ = r
            else:
                taggram, tags = r
            preds = np.mean(taggram, axis=0)
            mtt = [(tags[i], float(preds[i])) for i in range(len(tags)) if preds[i] >= 0.10]
            mtt.sort(key=lambda x: x[1], reverse=True)
            result["tags_mtt"] = [{"tag": t, "confidence": c} for t, c in mtt[:10]]
        except Exception as e:
            logger.warning(f"{music_id}: MTT 提取失敗 - {e}")
            result["tags_mtt"] = []
        
        logger.info(f"✓ {music_id}: musicnn tags 生成成功")
        return result
    
    except Exception as e:
        logger.error(f"{music_id}: musicnn tags 生成失敗 - {e}")
        return None

# ========== 更新 JSONL 檔案 ==========
def update_jsonl_file(file_path, music_id, new_data):
    """更新 JSONL 檔案中的特定記錄"""
    if not file_path.exists():
        logger.error(f"檔案不存在: {file_path}")
        return False
    
    try:
        records = []
        updated = False
        
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                if obj.get("music_id") == music_id:
                    records.append(new_data)
                    updated = True
                else:
                    records.append(obj)
        
        if not updated:
            records.append(new_data)
        
        # 備份
        backup_path = file_path.parent / f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.copy2(file_path, backup_path)
        
        # 寫回
        with jsonlines.open(file_path, 'w') as writer:
            for record in records:
                writer.write(record)
        
        return True
    
    except Exception as e:
        logger.error(f"更新 JSONL 失敗: {e}")
        return False

# ========== 更新 metadata JSON ==========
def update_metadata_json(music_id, tags):
    """更新 metadata JSON 中的記錄"""
    if not METADATA_FILE.exists():
        logger.error(f"檔案不存在: {METADATA_FILE}")
        return False
    
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if music_id not in metadata:
            logger.warning(f"{music_id} 不在 metadata 中")
            return False
        
        # 更新 tags 和 genre
        metadata[music_id]["tags"] = tags
        
        genres = {"pop", "rock", "hip-hop", "electronic", "jazz", "classical", "metal", "r&b", "country"}
        genre = "unknown"
        
        for tag in tags:
            if tag.lower() in genres:
                genre = tag.lower()
                break
        
        if genre == "unknown" and tags:
            genre = tags[0].lower()
        
        metadata[music_id]["genre"] = genre
        
        logger.info(f"更新 {music_id}: genre={genre}, tags={len(tags)}個")
        
        # 備份
        backup_path = METADATA_FILE.parent / f"{METADATA_FILE.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(METADATA_FILE, backup_path)
        
        # 寫回
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        logger.error(f"更新 metadata 失敗: {e}")
        return False

# ========== 主流程 ==========
def main():
    logger.info("="*70)
    logger.info("兩階段音檔提取（修正版）")
    logger.info("="*70)
    
    logger.info(f"\n需要處理的音樂 ID: {len(CORRUPTED_IDS)} 個")
    
    stats = {
        "total": len(CORRUPTED_IDS),
        "audio_extracted": 0,
        "audio_failed": 0,
        "musicnn_updated": 0,
        "metadata_updated": 0,
    }
    
    for idx, music_id in enumerate(CORRUPTED_IDS, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"處理 [{idx}/{len(CORRUPTED_IDS)}]: {music_id}")
        logger.info(f"{'='*70}")
        
        # Step 1: 提取音檔
        if extract_audio_two_stage(music_id):
            stats["audio_extracted"] += 1
            
            # Step 2: 生成 tags
            logger.info(f"[步驟 2/3] 生成 musicnn tags...")
            musicnn_data = regenerate_musicnn_tags(music_id)
            
            if musicnn_data:
                stats["musicnn_updated"] += 1
                
                # Step 3: 更新檔案
                logger.info(f"[步驟 3/3] 更新 JSON 檔案...")
                update_jsonl_file(MUSICNN_TAGS_FILE, music_id, musicnn_data)
                
                tags = []
                for tag_obj in musicnn_data.get("tags_msd", []):
                    tags.append(tag_obj["tag"])
                for tag_obj in musicnn_data.get("tags_mtt", []):
                    tags.append(tag_obj["tag"])
                tags = list(dict.fromkeys(tags))
                
                if update_metadata_json(music_id, tags):
                    stats["metadata_updated"] += 1
        else:
            stats["audio_failed"] += 1
    
    # 顯示統計
    logger.info("\n" + "="*70)
    logger.info("處理完成！")
    logger.info("="*70)
    logger.info(f"\n【音檔提取】")
    logger.info(f"  成功: {stats['audio_extracted']}/{stats['total']}")
    logger.info(f"  失敗: {stats['audio_failed']}/{stats['total']}")
    logger.info(f"\n【Tags 更新】 {stats['musicnn_updated']}/{stats['total']}")
    logger.info(f"【Metadata 更新】 {stats['metadata_updated']}/{stats['total']}")
    
    if stats['audio_extracted'] == stats['total']:
        logger.info("\n✅ 全部處理成功！")
    else:
        logger.warning(f"\n⚠️  {stats['audio_failed']} 個失敗")
    
    logger.info("="*70)

if __name__ == "__main__":
    main()
