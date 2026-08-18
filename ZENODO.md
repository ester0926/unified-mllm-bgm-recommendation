# Zenodo 下載包說明

本 repo 的程式碼與小型結果由 GitHub 管理；大型重現檔案另以 Zenodo record 發布。Zenodo record 發布後，請在本文件補上 DOI 與下載連結。

```text
Zenodo DOI: 10.5281/zenodo.21980446
Zenodo record URL: https://doi.org/10.5281/zenodo.21980446
```

## 檔案清單

| 檔案 | 內容 | 解壓位置 |
|---|---|---|
| `bgm_recommender_model_checkpoints_v1.tar.zst` | `exp_01` 至 `exp_07` 的 `best/` checkpoint，包括 LoRA adapter、projector、ranking head 與相關設定 | repo 根目錄 |
| `bgm_recommender_ltp_cache_v1.tar.zst` | runtime cache、final `preference_vectors*.h5`、Stage 5 generation logs 與 projection weights | repo 根目錄 |
| `bgm_recommender_metadata_and_ids_v1.tar.zst` | music metadata、HDF5 ID 清單、split cache、feature-alignment whitelist | repo 根目錄 |
| `CHECKSUMS.txt` | Zenodo 壓縮包的 SHA256 checksum | 不需解壓 |

## 解壓方式

下載 Zenodo 檔案後，在本 repo 根目錄執行：

```powershell
tar -xf path\to\bgm_recommender_model_checkpoints_v1.tar.zst
tar -xf path\to\bgm_recommender_ltp_cache_v1.tar.zst
tar -xf path\to\bgm_recommender_metadata_and_ids_v1.tar.zst
```

解壓後應出現：

```text
checkpoints/exp_01/best/
checkpoints/exp_02/best/
...
checkpoints/exp_07/best/
cache/
data/user_profiling/stage5_output/
data/user_profiling/music_metadata_simple/
data/video_ids_from_hdf5_*.json
data/musechat_split_cache.json
data_preparation/feature_alignment/
```

## Checksum 驗證

PowerShell 可使用：

```powershell
Get-FileHash .\bgm_recommender_model_checkpoints_v1.tar.zst -Algorithm SHA256
Get-FileHash .\bgm_recommender_ltp_cache_v1.tar.zst -Algorithm SHA256
Get-FileHash .\bgm_recommender_metadata_and_ids_v1.tar.zst -Algorithm SHA256
```

將輸出與 Zenodo 上的 `CHECKSUMS.txt` 比對即可。

## 未包含資料

Zenodo v1 重現包不包含完整 MuseChat HDF5 features 與原始影音檔。完整 HDF5 features 體積約 TB 等級，不適合放入 GitHub 或一般 Zenodo record；原始影音檔也可能涉及授權限制。若需完整重建 features，請依 [DATA.md](DATA.md) 與 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) 中的資料準備說明處理。
