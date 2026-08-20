# 智慧貨櫃動態風險巡檢決策系統

**阿對對隊** · 高雄港

把「港區壅塞程度」與「貨櫃破損辨識結果」合成單一風險分數（RPS），
讓巡檢人力隨時間動態分配到最該去的貨櫃中心。

```
RPS = PPI × ( 1 + β · D_score )

PPI     = 小船等量 / 泊靠上限          （壅塞程度）
D_score = Σ 在泊貨櫃破損信心            （破損風險）
```

## 線上展示

| 頁面 | 網址 |
|---|---|
| [入口頁](https://arnold9487.github.io/2026-imarine-6th/) |https://arnold9487.github.io/2026-imarine-6th/|
| [動態風險巡檢決策儀表盤](https://arnold9487.github.io/2026-imarine-6th/dashboard/) |https://arnold9487.github.io/2026-imarine-6th/dashboard/|
| [高雄港壓力指數](https://arnold9487.github.io/2026-imarine-6th/pressure/) 記得刷新頁面一下|https://arnold9487.github.io/2026-imarine-6th/pressure/|

## 目錄結構

```
imarine/
├── index.html          入口頁
├── dashboard/          動態風險巡檢決策儀表盤（網頁）
├── pressure/           高雄港壓力指數（網頁，含 build/ 進出港原始數據）
├── pressure-offline/   壓力指數離線版（雙擊 啟動.bat 即可離線展示）
├── model/              貨櫃破損辨識模型（Python 訓練流程）
└── docs/               簡報與文件
```

三個子專案各自有 README，細節請看各資料夾。

## 辨識模型

以 SeaFront 合成資料集訓練 YOLO 破損偵測模型，走 sim-to-real 路線。

- 資料集說明：[model/SeaFront資料集說明.md](model/SeaFront資料集說明.md)
- 訓練流程規劃：[model/訓練流程規劃.md](model/訓練流程規劃.md)
- 實驗紀錄：[model/訓練/實驗紀錄.md](model/訓練/實驗紀錄.md)

### 權重檔案

模型權重不納入版本控制（單檔約 19MB）：

- **訓練成果 `best.pt`** — 請至本專案 [Releases](../../releases) 下載，放到 `model/訓練/runs/damage_yolo11s-2/weights/`
- **預訓練權重 `yolo11s.pt` / `yolo26n.pt`** — 首次執行訓練時由 ultralytics 自動下載，不需手動準備

## 本機執行

網頁部分因為會 `fetch()` 讀取 JSON，不能直接用 `file://` 開啟，需要起一個本機伺服器：

```bash
python -m http.server 8000
```

然後開 http://localhost:8000/ 。

`pressure-offline/` 是為了離線展示做的版本，資料改成 `.js` 內嵌，可以直接雙擊 `index.html` 開啟。
