# 系統架構概覽 (Architecture Overview)

## 系統流程圖
```
┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│  Data Ingest │──▶│ Feature Store │──▶│ Graph Builder   │
│ (fetch_youbike)│ │ (ETL 清理)    │    │ (kNN, 邊權重)   │
└──────────────┘    └──────────────┘    └──────┬─────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │ Model Train  │
                                        │ (LSTM/DCRNN) │
                                        └──────┬───────┘
                                               │
                  ┌──────────────┐             │             ┌──────────────┐
                  │ Batch Scoring│◀────────────┘────────────▶│ Evaluation   │
                  │ / Serving    │                           │ (metrics)    │
                  └──────────────┘                           └──────┬───────┘
                                                                    │
                                                                    ▼
                                                             ┌──────────────┐
                                                             │ deck.gl UI   │
                                                             │ (Visualization)
                                                             └──────────────┘
```

## 技術棧 (Tech Stack)
- **資料處理**：Python 3.11、Pandas、NumPy。
- **圖構建**：scikit-learn NearestNeighbors、Geo 計算。
- **模型訓練**：PyTorch、PyTorch Geometric Temporal、DCRNN、Graph WaveNet。
- **服務層**：FastAPI (預留)、可能搭配 Uvicorn 或 Batch Job。
- **視覺化**：deck.gl / pydeck、可部署於靜態網站或整合至內部儀表板。
- **CI/CD**：GitHub Actions、自動化測試與 artifact 上傳。

## 模組邊界 (Module Boundaries)
- `etl/`：資料擷取、清理、特徵工程、圖構建。
- `model/`：模型訓練腳本、參數設定、評估報告輸出。
- `spec/`：規格、驗收、ADR，提供產品與技術對齊依據。
- `web/`：視覺化原型，後續可接 API 或批次輸出的檔案。

## 部署與監控考量
- 預測結果可寫入資料庫或物件儲存 (S3/GCS)，供 deck.gl 前端查詢。
- 評估指標與資料品質可透過 GitHub Actions 或 Airflow 產出報告，再進行 Slack 告警。
- 模型版本需追蹤 (例如 MLflow)，以支援回溯與 A/B 測試。
