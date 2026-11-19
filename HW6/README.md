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

## 2026-11-18（Asia/Taipei）— U-Net 800 epochs 與分層時間步實驗

本次只修改 `Lab6_Unet_studentID.ipynb` 與本 README；DiT、資料集、既有訓練產物、輸出路徑及檔名均未更動。

### 修改前後設定

- 從隨機初始化重新訓練：維持 `resume_checkpoint=None`。
- epoch：400 → 800。
- AdamW：維持最高 LR `2e-4`、`betas=(0.9, 0.99)`、矩陣/卷積權重 `weight_decay=1e-4`，bias 與 normalization 等一維參數不衰減。
- LR schedule：5 epochs warm-up、最低 LR `1e-5` → 10 epochs warm-up、最低 LR `2e-6`；warm-up 後仍按成功的 optimizer step 做 cosine decay。
- EMA decay 上限：0.9999 → 0.9995；原有依 `global_step` 漸增的 EMA warm-up 保留。
- timestep：每張圖片獨立均勻抽樣 → 循環式分層均勻抽樣。每個 batch 使用一個 0–999 均勻循環起點與近似等距 offset，再隨機排列；因此每張圖片的邊際分布仍是 0–999 的精確離散均勻分布，且最後不足完整 batch 時會依實際 batch 大小重新分層。
- 訓練目標仍為普通雜訊預測 MSE；1,000-step linear noise schedule、U-Net 架構、channels、bottleneck attention、batch size 與水平翻轉資料增強不變。

### 採樣選項

- 預設仍為 DDIM 250 steps、`eta=0.3`。
- 同一 checkpoint 可手動改成 500 steps、`eta=0.0`，或 250 steps、`eta=0.0`；notebook 不會自動連跑多組設定，也不會建立額外 ZIP。
- 新增預設關閉的 `flip_tta=False`。開啟後，每個採樣時間步會分別預測原圖與水平翻轉圖的雜訊，將後者翻回後平均，再交給原有 DDPM/DDIM reverse step。這只影響採樣，不改訓練、模型架構或 checkpoint 權重。
- `generation_config.json` 會沿用原路徑並額外記錄 `flip_tta`；ZIP 仍覆寫原本的 `student_outputs/DDPM/images.zip`，規格仍是 8,189 張位於根目錄的 64×64 RGB PNG。

### 修改原因與 checkpoint 相容性

- 延長訓練並降低 cosine 尾端 LR，目的是提供更多後期小幅更新；較短 EMA 記憶（上限 0.9995）能讓 EMA 在此訓練長度下更快跟隨線上模型。這些選擇仍需實驗驗證，不保證 FID 改善。
- 分層時間步可降低單一 batch 偶然集中在少數噪聲區段的情況，同時不改變每張圖片的均勻時間步邊際分布或 MSE 目標。
- `training_config` 已納入 800 epochs、最低 LR、EMA 上限及 timestep sampling 名稱；optimizer、scheduler、AMP scaler、RNG 與 DataLoader RNG 仍儲存在原 checkpoint。舊訓練 checkpoint 因設定不同會被相容性檢查拒絕，不會靜默續訓。
- 本實驗必須由 `resume_checkpoint=None` 從頭訓練。完成後的新 checkpoint 可在完全相同訓練設定下正確續訓；採樣步數、eta 與 flip TTA 不屬於訓練相容性設定，因此可使用同一新 checkpoint 自由調整。

### 驗證結果

- 僅進行 notebook JSON、所有 code cell 語法、LR 端點、分層抽樣邊界/間距/邊際分布推導、flip TTA 接線、checkpoint 狀態與輸出契約的靜態或記憶體內檢查。
- 未啟動或中斷 notebook kernel，未執行訓練、採樣、圖片生成、ZIP 匯出或 FID 評估，既有 notebook 輸出保留原樣。
- **尚未執行新訓練與評估，沒有新的生成品質或 FID 結果。**
