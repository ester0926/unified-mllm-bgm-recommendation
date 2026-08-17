# Checkpoint 與大型檔案管理

模型權重、LTP cache 與 MuseChat features 未納入 Git 版本控制。本 repository 僅保存程式、設定、流程文件與小型摘要結果。

## 版本控制範圍

以下類型屬於大型或可再生檔案，預設由 `.gitignore` 排除：

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

若需公開大型檔案，可使用 GitHub Release、Zenodo、Google Drive 或實驗室 NAS，並提供 checksum 方便後續研究者確認檔案一致性。

## 預期 checkpoint 結構

重現 `exp_01` 至 `exp_07` 時，預期 checkpoint 放置於：

```text
checkpoints/exp_01/best/
checkpoints/exp_02/best/
checkpoints/exp_03/best/
checkpoints/exp_04/best/
checkpoints/exp_05/best/
checkpoints/exp_06/best/
checkpoints/exp_07/best/
```

每個 `best/` 資料夾通常包含：

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

該 manifest 可用來追蹤 LTP cache、User Profiling 輸出、checkpoint 與分析結果的來源。若公開大型檔案，建議同時提供 `CHECKSUMS.txt`，記錄檔名、大小、checksum 與下載位置。
