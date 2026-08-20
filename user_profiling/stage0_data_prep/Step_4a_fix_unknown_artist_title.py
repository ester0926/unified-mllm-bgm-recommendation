"""
用途：修正 metadata 或音訊抽取過程中的缺漏與錯誤。
輸入：原始 metadata、音訊特徵、合成對話或前一階段輸出。
輸出：偏好 profile、LTP 向量、品質檢查結果或修補後資料。
執行：依 stage 編號順序執行，缺資料時請先看 DATA.md 與 LTP_PIPELINE.md。
"""

# 原始檔名：Step_4a_fix_unknown_artist_title.py
# 修正 metadata 中 unknown 的 artist 和 title
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
import time
import random

# ========== 設定 ==========
METADATA_FILE = Path("data/user_profiling/music_metadata_simple/music_metadata_fixed.json")
YOUTUBE_METADATA_FILE = Path("data/user_profiling/music_metadata_simple/youtube_metadata.jsonl")

# 是否實際執行 yt-dlp 重新抓取（False = 僅報告，True = 實際抓取）
ENABLE_YTDLP_FETCH = True

# 抓取延遲（避免被封鎖）
FETCH_DELAY_MIN = 0.2
FETCH_DELAY_MAX = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 載入 YouTube 原始資料 ==========
def load_youtube_data():
    """載入原始 YouTube metadata"""
    data = {}
    if YOUTUBE_METADATA_FILE.exists():
        try:
            with jsonlines.open(YOUTUBE_METADATA_FILE) as reader:
                for obj in reader:
                    data[obj["music_id"]] = obj
            logger.info(f"載入 {len(data)} 筆 YouTube metadata")
        except Exception as e:
            logger.error(f"載入 YouTube 資料失敗: {e}")
    else:
        logger.warning(f"YouTube metadata 檔案不存在: {YOUTUBE_METADATA_FILE}")
    return data

# ========== 使用 yt-dlp 重新抓取 ==========
def fetch_youtube_metadata(music_id):
    """使用 yt-dlp 重新抓取 YouTube metadata"""
    try:
        import yt_dlp
        
        url = f"https://www.youtube.com/watch?v={music_id}"
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if info:
                return {
                    "music_id": music_id,
                    "title": info.get('title'),
                    "artist": info.get('uploader') or info.get('channel'),
                    "album": info.get('album'),
                    "release_date": info.get('release_date') or info.get('upload_date'),
                    "duration": info.get('duration'),
                    "view_count": info.get('view_count'),
                    "youtube_url": url,
                    "fetched_at": datetime.now().isoformat()
                }
    except Exception as e:
        logger.warning(f"抓取 {music_id} 失敗: {e}")
    
    return None

# ========== 分析需要修正的項目 ==========
def analyze_metadata(metadata):
    """分析哪些項目需要修正"""
    needs_fix = {
        "artist_unknown": [],
        "title_unknown": [],
        "both_unknown": [],
        "total_unknown": 0
    }
    
    for music_id, entry in metadata.items():
        artist_unknown = entry.get("artist") == "unknown"
        title_unknown = entry.get("title") == "unknown"
        
        if artist_unknown and title_unknown:
            needs_fix["both_unknown"].append(music_id)
        elif artist_unknown:
            needs_fix["artist_unknown"].append(music_id)
        elif title_unknown:
            needs_fix["title_unknown"].append(music_id)
        
        if artist_unknown or title_unknown:
            needs_fix["total_unknown"] += 1
    
    return needs_fix

# ========== 從原始資料修正 ==========
def fix_from_youtube_data(metadata, youtube_data, needs_fix):
    """從原始 YouTube 資料修正"""
    logger.info("\n[步驟 1/2] 從原始 YouTube metadata 修正...")
    
    fixed_count = 0
    still_need_fix = []
    
    for music_id in tqdm(needs_fix["both_unknown"] + needs_fix["artist_unknown"] + needs_fix["title_unknown"], 
                         desc="檢查原始資料"):
        if music_id not in youtube_data:
            still_need_fix.append(music_id)
            continue
        
        yt_info = youtube_data[music_id]
        entry = metadata[music_id]
        
        fixed = False
        
        # 修正 artist
        if entry.get("artist") == "unknown" and yt_info.get("artist"):
            entry["artist"] = yt_info["artist"]
            fixed = True
        
        # 修正 title
        if entry.get("title") == "unknown" and yt_info.get("title"):
            entry["title"] = yt_info["title"]
            fixed = True
        
        if fixed:
            fixed_count += 1
        
        # 檢查是否還有 unknown
        if entry.get("artist") == "unknown" or entry.get("title") == "unknown":
            still_need_fix.append(music_id)
    
    logger.info(f"✓ 從原始資料修正: {fixed_count} 個")
    logger.info(f"⚠ 仍需修正: {len(still_need_fix)} 個")
    
    return still_need_fix

# ========== 使用 yt-dlp 重新抓取 ==========
def fix_with_ytdlp(metadata, youtube_data, music_ids):
    """使用 yt-dlp 重新抓取並修正"""
    if not ENABLE_YTDLP_FETCH:
        logger.info("\n[步驟 2/2] 跳過 yt-dlp 抓取（ENABLE_YTDLP_FETCH = False）")
        return 0
    
    logger.info(f"\n[步驟 2/2] 使用 yt-dlp 重新抓取 {len(music_ids)} 個...")
    
    fixed_count = 0
    failed_count = 0
    
    for music_id in tqdm(music_ids, desc="yt-dlp 抓取"):
        # 抓取新資料
        yt_info = fetch_youtube_metadata(music_id)
        
        if yt_info:
            entry = metadata[music_id]
            
            # 修正 artist
            if entry.get("artist") == "unknown" and yt_info.get("artist"):
                entry["artist"] = yt_info["artist"]
                fixed_count += 1
            
            # 修正 title
            if entry.get("title") == "unknown" and yt_info.get("title"):
                entry["title"] = yt_info["title"]
                fixed_count += 1
            
            # 更新 YouTube metadata 檔案
            update_youtube_metadata_file(music_id, yt_info, youtube_data)
        else:
            failed_count += 1
        
        # 延遲避免被封鎖
        time.sleep(random.uniform(FETCH_DELAY_MIN, FETCH_DELAY_MAX))
    
    logger.info(f"✓ yt-dlp 修正: {fixed_count} 個欄位")
    logger.info(f"✗ 抓取失敗: {failed_count} 個")
    
    return fixed_count

# ========== 更新 YouTube metadata JSONL ==========
def update_youtube_metadata_file(music_id, new_data, youtube_data):
    """更新 YouTube metadata JSONL 檔案"""
    try:
        # 更新記憶體中的資料
        youtube_data[music_id] = new_data
    except Exception as e:
        logger.warning(f"更新記憶體資料失敗: {e}")

def save_youtube_metadata(youtube_data):
    """儲存 YouTube metadata 到 JSONL"""
    try:
        # 備份原檔案
        if YOUTUBE_METADATA_FILE.exists():
            backup_path = YOUTUBE_METADATA_FILE.parent / f"{YOUTUBE_METADATA_FILE.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            shutil.copy2(YOUTUBE_METADATA_FILE, backup_path)
            logger.info(f"備份 YouTube metadata: {backup_path.name}")
        
        # 寫入新資料
        with jsonlines.open(YOUTUBE_METADATA_FILE, 'w') as writer:
            for music_id in sorted(youtube_data.keys()):
                writer.write(youtube_data[music_id])
        
        logger.info(f"✓ 已更新 YouTube metadata 檔案")
    except Exception as e:
        logger.error(f"儲存 YouTube metadata 失敗: {e}")

# ========== 顯示修正前後對比 ==========
def show_comparison(metadata, sample_ids, title="修正範例"):
    """顯示修正前後的對比"""
    if not sample_ids:
        return
    
    logger.info(f"\n{title} (顯示前 5 個):")
    for music_id in sample_ids[:5]:
        if music_id in metadata:
            entry = metadata[music_id]
            logger.info(f"  {music_id}:")
            logger.info(f"    Artist: {entry.get('artist', 'N/A')}")
            logger.info(f"    Title: {entry.get('title', 'N/A')}")

# ========== 主流程 ==========
def main():
    logger.info("="*70)
    logger.info("修正 Unknown Artist/Title")
    logger.info("="*70)
    
    # 載入 metadata
    logger.info(f"\n載入 metadata...")
    if not METADATA_FILE.exists():
        logger.error(f"檔案不存在: {METADATA_FILE}")
        return
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"✓ 載入 {len(metadata)} 筆 metadata")
    
    # 載入 YouTube 原始資料
    youtube_data = load_youtube_data()
    
    # 分析需要修正的項目
    logger.info(f"\n分析需要修正的項目...")
    needs_fix = analyze_metadata(metadata)
    
    logger.info(f"\n統計:")
    logger.info(f"  Artist unknown: {len(needs_fix['artist_unknown'])} 個")
    logger.info(f"  Title unknown: {len(needs_fix['title_unknown'])} 個")
    logger.info(f"  Both unknown: {len(needs_fix['both_unknown'])} 個")
    logger.info(f"  總計需修正: {needs_fix['total_unknown']} 個")
    
    if needs_fix['total_unknown'] == 0:
        logger.info("\n✅ 所有項目都正常，無需修正！")
        return
    
    # 顯示部分範例
    show_comparison(metadata, needs_fix['both_unknown'], "Both Unknown 範例")
    
    # 確認是否繼續
    logger.info(f"\n設定:")
    logger.info(f"  ENABLE_YTDLP_FETCH = {ENABLE_YTDLP_FETCH}")
    if ENABLE_YTDLP_FETCH:
        logger.info(f"  延遲範圍: {FETCH_DELAY_MIN}-{FETCH_DELAY_MAX} 秒")
    
    confirm = input("\n是否繼續修正? (yes/no): ").strip().lower()
    if confirm != "yes":
        logger.info("已取消")
        return
    
    # 備份原檔案
    logger.info(f"\n備份原始檔案...")
    backup_path = METADATA_FILE.parent / f"{METADATA_FILE.stem}_before_artist_title_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(METADATA_FILE, backup_path)
    logger.info(f"✓ 備份: {backup_path.name}")
    
    # 修正流程
    # 步驟 1: 從原始 YouTube 資料修正
    all_need_fix = needs_fix["both_unknown"] + needs_fix["artist_unknown"] + needs_fix["title_unknown"]
    still_need_fix = fix_from_youtube_data(metadata, youtube_data, needs_fix)
    
    # 步驟 2: 使用 yt-dlp 重新抓取
    if still_need_fix:
        fix_with_ytdlp(metadata, youtube_data, still_need_fix)
    
    # 儲存更新後的 metadata
    logger.info(f"\n儲存更新後的 metadata...")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 已儲存: {METADATA_FILE}")
    
    # 儲存更新後的 YouTube metadata
    if ENABLE_YTDLP_FETCH and still_need_fix:
        save_youtube_metadata(youtube_data)
    
    # 重新統計
    logger.info(f"\n重新分析...")
    needs_fix_after = analyze_metadata(metadata)
    
    logger.info(f"\n" + "="*70)
    logger.info("修正完成！")
    logger.info("="*70)
    
    logger.info(f"\n【修正前】")
    logger.info(f"  Artist unknown: {len(needs_fix['artist_unknown'])}")
    logger.info(f"  Title unknown: {len(needs_fix['title_unknown'])}")
    logger.info(f"  Both unknown: {len(needs_fix['both_unknown'])}")
    logger.info(f"  總計: {needs_fix['total_unknown']}")
    
    logger.info(f"\n【修正後】")
    logger.info(f"  Artist unknown: {len(needs_fix_after['artist_unknown'])}")
    logger.info(f"  Title unknown: {len(needs_fix_after['title_unknown'])}")
    logger.info(f"  Both unknown: {len(needs_fix_after['both_unknown'])}")
    logger.info(f"  總計: {needs_fix_after['total_unknown']}")
    
    logger.info(f"\n【修正數量】")
    logger.info(f"  成功修正: {needs_fix['total_unknown'] - needs_fix_after['total_unknown']} 個")
    
    if needs_fix_after['total_unknown'] > 0:
        logger.info(f"\n⚠ 仍有 {needs_fix_after['total_unknown']} 個項目無法修正")
        logger.info("可能原因:")
        logger.info("  1. YouTube 影片已被刪除或設為私人")
        logger.info("  2. 影片資訊本身就沒有 artist/title")
        logger.info("  3. yt-dlp 抓取失敗")
        
        # 顯示無法修正的範例
        show_comparison(metadata, needs_fix_after['both_unknown'][:5], "\n無法修正的範例")
    else:
        logger.info(f"\n✅ 全部修正完成！")
    
    logger.info(f"\n備份檔案位置:")
    logger.info(f"  Metadata: {backup_path}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
