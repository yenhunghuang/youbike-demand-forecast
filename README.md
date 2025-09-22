# YouBike Demand Forecast

## Overview / 專案簡介
YouBike Demand Forecast 為一個以規格導向 (spec-driven) 為核心的專案腳手架，協助團隊快速啟動台北市 YouBike 2.0 站點需求預測工作。專案聚焦在建立即時資料擷取流程、圖神經網路模型訓練框架，以及 deck.gl 互動式地圖，讓營運與產品團隊能夠檢視未來 15、30、60 分鐘的可借車數量預測。

這份腳手架同時提供中英雙語的文件與 README，並透過 PRD、ADR、spec、Acceptance 等文檔將專案規格清楚落地，方便跨職能協作。CI 也預設會執行 baseline 評估與測試，以確保程式碼品質。

## Feature Highlights
- 📄 **Documentation-first**：`docs/` 與 `spec/` 目錄包含 PRD、架構設計、ADR 與驗收案例。
- 🧭 **完整資料管線**：`etl/` 內含即時資料擷取與基於地理距離的圖構建腳本。
- 🧠 **模型腳手架**：提供 LSTM baseline、DCRNN、Graph WaveNet 的訓練與評估模板。
- 🗺️ **視覺化展示**：`web/deck/` 提供 deck.gl demo，支援多時間視窗與殘差檢視。
- ✅ **CI 守門人**：GitHub Actions 會安裝依賴、檢查 spec 檔案、執行 `model/evaluate.py` 與 `pytest`。

## Repository Structure
```
├── docs/                      # PRD、架構、ADR 文件
├── etl/                       # 抓取即時資料與圖構建腳本
├── model/                     # Baseline 與圖神經網路訓練、評估程式
├── spec/                      # 功能規格與驗收 YAML
├── web/deck/                  # deck.gl demo 靜態頁面
├── tests/                     # Pytest 單元測試
├── requirements.txt           # Python 依賴清單
└── .github/workflows/ci.yml   # GitHub Actions CI 設定
```

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip**：若在 Apple Silicon 或沒有 GPU 的環境安裝 `torch` / `torch-geometric` 遇到問題，可依照官方指引改用對應的 wheel，或暫時註解掉高階依賴僅驗證腳手架。

## Running the ETL Pipeline
1. 下載即時資料：
   ```bash
   python etl/fetch_youbike.py
   ```
   指令會在 `data/raw/` 生成帶時間戳的 JSON。
2. 建立站點圖：
   ```bash
   python etl/build_graph.py --k 8
   ```
   會輸出 `data/processed/youbike_graph.json`，包含節點資訊與加權邊列表。

## Baseline & Advanced Models
- `python model/baseline_lstm.py`：利用合成資料執行示範訓練步驟並輸出損失，驗證模型骨架可運作。
- `python model/dcrnn_train.py` / `python model/gwnet_train.py`：預留訓練流程模板，後續可補上資料載入與訓練邏輯。

## Evaluation Workflow
執行下列指令會產生 `reports/metrics.json`，格式符合驗收規格：
```bash
python model/evaluate.py
```
檔案會輸出 MAE 與 RMSE 指標，分別對齊 T+15 (尖峰)、T+30 (離峰)、T+60 (整日) 的視窗，可供 CI 上傳 artifact 或人工檢視。

## Visualization Demo
```bash
python -m http.server --directory web/deck 8000
```
開啟瀏覽器至 <http://localhost:8000>，即可查看 deck.gl `ScatterplotLayer` 範例與時間滑桿的佈局 placeholder。

## Testing
```bash
pytest
```
Pytest 目前會檢查評估報告能正確序列化與讀取，後續可持續擴增資料與模型邏輯的測試覆蓋。

## References / 參考資料
- YouBike 2.0 即時資料來源：<https://data.gov.tw/en/datasets/137993>
- PyTorch Geometric Temporal 官方文檔與範例：<https://github.com/benedekrozemberczki/pytorch_geometric_temporal>
- DCRNN 論文與實作：Li, Yaguang, et al. "Diffusion convolutional recurrent neural network." ICLR 2018.
- Graph WaveNet 論文與實作：Wu, Zonghan, et al. "Graph WaveNet for deep spatial-temporal graph modeling." IJCAI 2019.
