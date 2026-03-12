# 📊 Company Bankruptcy Prediction (公司破產預測)

本專案旨在利用機器學習模型預測公司是否面臨破產風險。透過深入的資料前處理、特徵工程以及不平衡資料（Imbalanced Data）處理技術，建構出穩定且具預測力的分類模型。

## 📂 資料來源
本專案使用 Kaggle 提供的開源資料集：
[Company Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
資料源自台灣經濟新報（Taiwan Economic Journal, TEJ），包含了 1999 年至 2009 年間的公司財務指標。

## 🛠️ 技術棧 (Tech Stack)
* **程式語言:** Python
* **資料處理與視覺化:** Pandas, NumPy, Matplotlib, Seaborn, klib
* **機器學習:** Scikit-learn (Random Forest, SVM, PCA, GridSearchCV)
* **資料不平衡處理:** Imbalanced-learn (BorderlineSMOTE, SMOTEENN, RandomUnderSampler)

## 📁 專案結構與檔案說明

專案由三個主要腳本組成，具備資料傳遞的相依性：

1. **`bankruptcy_final.py`** (主程式與資料前處理)
   - 自動透過 `kagglehub` 下載資料集。
   - 執行 EDA、常數與重複值清理。
   - 偏態轉換（Log1p, Square root, Yeo-Johnson）。
   - 基於標準差（Standard Deviation Threshold）的離群值處理與 Z-score 標準化。
   - 使用 BorderlineSMOTE 進行向上採樣，以及 SMOTEENN 進行向下採樣。
   - 運用 PCA 進行特徵降維與篩選，並建立 SVM 基準模型。

2. **`rd_up_pca.py`** (向上採樣 - 隨機森林模型)
   - 匯入 `bankruptcy_final.py` 處理好的向上採樣資料（SMOTE）。
   - 使用 `GridSearchCV` 進行隨機森林模型的超參數調優（包含 `n_estimators`, `max_depth`, `min_samples_split`）。
   - 繪製混淆矩陣（Confusion Matrix）與 Top 15 特徵重要性長條圖。

3. **`rd_down_pca.py`** (向下採樣 - 隨機森林模型)
   - 匯入 `bankruptcy_final.py` 的標準化特徵。
   - 撰寫迴圈測試不同 `random_state` 下的 `RandomUnderSampler` 效果，選出最佳抽樣結果。
   - 結合 PCA 特徵貢獻度篩選變數。
   - 訓練隨機森林模型，並視覺化預測結果與特徵重要性。

## 🚀 如何執行專案

1. **安裝必要的套件:**
   確保你的環境中安裝了所需套件（建議使用虛擬環境）：
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn klib feature_engine kagglehub