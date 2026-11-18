# Lab 6 改善紀錄

## 2026-11-18（Asia/Taipei）

修改檔案：

- `Lab6_Unet_studentID.ipynb`
- `Lab6_DiT_studentID.ipynb`

本次未改動模型家族或結構設定，仍為 64×64 RGB pixel-space、無條件 DDPM、雜訊預測與 MSE 目標；資料仍只讀取既有 `resized_flowers102`，輸出路徑、檔名、ZIP 規格與既有 EMA 流程不變。

### U-Net 設定變更

- 訓練 epoch：200 → 400。
- optimizer：Adam（`lr=1e-4`）→ AdamW（`lr=2e-4`、`betas=(0.9, 0.99)`、矩陣/卷積權重 `weight_decay=1e-4`，bias 與 normalization 等一維參數不衰減）。
- learning-rate schedule：固定 LR → 5 epochs 線性 warm-up，之後按 optimizer step 做 cosine decay，最低為初始 LR 的 5%。
- 最終 DDIM 採樣步數：100 → 250；`ddim_eta=0.0`、seed、批次大小與輸出介面不變。

### DiT 設定變更

- 訓練 epoch：200 → 400。
- optimizer：Adam（`lr=1e-4`）→ AdamW（`lr=1e-4`、`betas=(0.9, 0.95)`、矩陣權重 `weight_decay=1e-2`，bias 與 normalization 等一維參數不衰減）。
- learning-rate schedule：固定 LR → 10 epochs 線性 warm-up，之後按 optimizer step 做 cosine decay，最低為初始 LR 的 5%。
- 最終 DDIM 採樣步數：100 → 250；`ddim_eta=0.0`、seed、批次大小與輸出介面不變。

### 修改依據與預期

- warm-up 可降低 U-Net/DiT 從零初始化開始時的大幅參數更新；cosine decay 讓後期更新更細緻。
- AdamW 將權重衰減與梯度更新解耦；DiT 使用較常見的 `(0.9, 0.95)` 動量設定，且不對 bias/normalization 一維參數做衰減。
- 400 epochs 提供較多資料覆蓋與最佳化步數；250-step DDIM 以較高生成成本換取較小的離散化誤差。
- 以上是有依據的改善方向，但實際生成品質與 FID 必須經重新訓練及正式評估確認，不能保證一定改善。

### Checkpoint 相容性

- scheduler 狀態已寫入原有 `last_checkpoint.pt` 與週期 checkpoint，不新增 checkpoint 檔名或路徑；成功的 optimizer step 才會推進 scheduler，與 AMP skip 保持一致。
- 新設定的 optimizer、參數群組、總訓練步數及 scheduler 與舊訓練 checkpoint 不相容。兩份 notebook 都應以 `resume_checkpoint=None` 從頭訓練；程式會以 `training_config` 差異拒絕舊 checkpoint 的訓練續跑，不會靜默或強制載入。
- 舊 checkpoint 的模型架構與 diffusion 設定未變，因此仍可在 `train_model=False` 時載入做純生成；但那不會得到本次新訓練設定的效果。
- 新 checkpoint 續訓時必須保留相同的 `num_epochs`，因 cosine 排程的總步數由它決定。

### 實際驗證結果

- 僅進行 notebook JSON、Python 語法、關鍵設定、checkpoint/scheduler 接線與輸出契約的靜態/記憶體內檢查。
- 未啟動或中斷 notebook kernel，未執行訓練、採樣、圖片生成、FID 評估或 ZIP 匯出，也未修改既有 notebook 輸出。
- **尚未執行修改後的訓練與評估，沒有可宣稱的品質或 FID 改善結果。**
