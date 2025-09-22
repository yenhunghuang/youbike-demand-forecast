# YouBike Demand Forecasting Feature Specification

## 目標與 KPI (Objectives & KPIs)
- 主要指標：尖峰 (07:00-10:00、17:00-20:00) 與離峰時段的 MAE、RMSE。
- 預測視窗：T+15、T+30、T+60 分鐘。
- 驗收門檻：
  - 尖峰 T+15 MAE ≤ 2.0、RMSE ≤ 3.5。
  - 離峰 T+30 MAE ≤ 2.5、RMSE ≤ 3.8。
  - 全日 T+60 MAE ≤ 3.0、RMSE ≤ 4.5。
- 使用者影響：預測準確度需支援補車／回收決策，減少缺車、滿車率各 5% 以上。

## 資料來源 (Data Sources)
- **即時資料**：台北市 YouBike 2.0 JSON Feed，欄位包含站點編號 (`sno`)、站點名稱 (`sna`)、經緯度 (`lat`, `lng`)、可借 (`sbi`)、可還 (`bemp`)、總柱數 (`tot`)、最後更新時間 (`mday`)、服務狀態 (`act`) 等。
- **更新頻率**：官方資料每分鐘更新一次，需透過排程抓取以建立歷史序列。
- **資料品質注意事項**：
  - `act` = 0 代表停用站點，需標記或過濾。
  - 站點異動（新增/移除）需追蹤，維護節點主檔。
  - `mday` 以台北時間提供，轉換為 UTC 時需留意時區。
- **引用**：<https://data.gov.tw/en/datasets/137993>

## 資料處理與特徵工程
1. **資料蒐集**：`etl/fetch_youbike.py` 每分鐘抓取 JSON，儲存於 `data/raw/`。
2. **資料清理**：將 `sbi`, `bemp` 等欄位轉換為整數；缺失值以鄰近時間點或站點平均填補。
3. **時間特徵**：加入星期幾、是否假日、時間區段 (peak/offpeak) 等衍生欄位。
4. **圖結構**：透過 `etl/build_graph.py` 以地理座標計算距離、kNN 建邊，存於 `data/processed/youbike_graph.json`。
5. **序列切片**：生成長度 L 的歷史視窗 (例如 12 個 5 分鐘步長) 與多視窗預測目標。

## 模型候選 (Candidate Models)
- **Baseline LSTM**：快速驗證時間序列預測的下限表現。
- **Diffusion Convolutional RNN (DCRNN)**：適用於時空圖資料，捕捉流量擴散。
- **Graph WaveNet**：透過門控圖卷積與小波變換建模長距離依賴。
- **實作框架**：PyTorch 與 PyTorch Geometric Temporal。

## 評估策略 (Evaluation Strategy)
- 將資料依時間切分為訓練 / 驗證 / 測試集，採 rolling window 驗證。
- 在尖峰與離峰時段分別計算 MAE、RMSE。
- 產生 `reports/metrics.json`，欄位對應驗收 YAML：
  ```json
  {
    "t15_peak": {"mae": 1.95, "rmse": 3.30},
    "t30_offpeak": {"mae": 2.40, "rmse": 3.60},
    "t60_all": {"mae": 2.85, "rmse": 4.40}
  }
  ```
- 日後可擴充殘差分析、站點等級比較、AB 測試等報告。

## 視覺化需求 (Visualization Requirements)
- 使用 deck.gl / pydeck 結合地理資訊呈現預測值與誤差。
- 地圖要提供時間滑桿切換 T+15、T+30、T+60，並能顯示實際值與預測的差異。
- 支援以顏色或點大小呈現可借車剩餘量、或殘差熱度圖。
- 可附加 Tooltip 顯示站點名稱、預測趨勢、信心區間。

## 風險與替代方案 (Risks & Mitigations)
- **資料缺失**：設計回填策略、資料品質監控儀表板。
- **站點維護或欄位異常**：建立站點主檔與資料驗證，必要時回退至鄰近站點的估計。
- **突發事件**：記錄特殊事件 (路跑、暴雨)，可透過外部特徵或情境模擬降低風險。
- **模型失準**：提供 ARIMA、XGBoost 等傳統模型作為備援比較。

## 參考資源 (References)
- YouBike 2.0 即時資料來源：<https://data.gov.tw/en/datasets/137993>
- PyTorch Geometric Temporal 官方文檔與範例：<https://github.com/benedekrozemberczki/pytorch_geometric_temporal>
- DCRNN 與 Graph WaveNet 論文與實作：ICLR 2018、IJCAI 2019 論文與開源實作。
