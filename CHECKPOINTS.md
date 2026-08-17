# Checkpoint 與大型檔案說明

模型權重、LTP cache 與 MuseChat features 不放入 GitHub。這個 repo 只保存程式、說明文件與小型摘要結果。

## 不要直接 commit 的檔案

以下檔案通常很大，請不要直接上傳到 GitHub：

```text
adapter_model.safetensors
projectors.pt
ranking_head.pt
*.h5
*.hdf5
*.npy
*.npz
*.pt
*.pth
*.safetensors
```

完整 checkpoint 可改用 Git LFS、GitHub Release、Zenodo、Google Drive 或實驗室 NAS 保存。

## 預期 checkpoint 放置方式

若要重現 `exp_01` 至 `exp_07`，請把權重放在下列位置：

```text
checkpoints/exp_01/best/
checkpoints/exp_02/best/
checkpoints/exp_03/best/
checkpoints/exp_04/best/
checkpoints/exp_05/best/
checkpoints/exp_06/best/
checkpoints/exp_07/best/
```

每個 `best/` 資料夾通常需要：

```text
adapter_model.safetensors
projectors.pt
ranking_head.pt
config or README file
```

## Checksum 與 provenance

本 repo 保留：

```text
results/analysis/v21_reproducibility_manifest.json
```

這個 manifest 可用來追蹤 LTP cache、User Profiling 輸出、checkpoint 與分析結果的來源。若未來要公開大型檔案，建議另外產生 `CHECKSUMS.txt`，並把 checksum 與下載連結一起放在 GitHub Release 或 README 中。
