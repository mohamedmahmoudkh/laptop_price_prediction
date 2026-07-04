# 💻 Laptop Price Prediction

A machine learning project that predicts laptop prices (in Euros) from hardware specifications, built as part of a Kaggle-style regression assignment. The final model — a tuned **Gradient Boosting Regressor** — achieves an **R² of 0.90**, beating the target benchmark of 0.73 by **+17 percentage points**.

## 📊 Dataset

| | |
|---|---|
| Training samples | 881 |
| Test samples | 394 |
| Features | 16 (15 excl. target) |
| Target | `Price (Euro)` |
| Missing values | 0 |

The dataset contains laptop specs such as company, type, screen resolution, CPU, RAM, storage (SSD/HDD), GPU, operating system, weight, and price.

## 🔍 Project Workflow

### 1. Exploratory Data Analysis
- Price distribution is strongly right-skewed (skew = 1.73); a `log1p` transform normalizes it (skew = -0.07).
- Price varies clearly by brand (Razer, Apple, LG = premium; Vero, Chuwi, Mediacom = budget) and by laptop type (Workstation/Gaming = highest, Netbook/Notebook = lowest).
- RAM shows a strong positive relationship with price — 32GB laptops cost ~3x more than 4GB ones on average.
- Correlation analysis shows `RAM (GB)` (~0.73) and the engineered `RAM_x_SSD` interaction (~0.78) as the strongest predictors.

### 2. Data Preparation & Feature Engineering
- Parsed raw text columns (`ScreenResolution`, `Memory`) with regex into structured numeric fields (touchscreen flag, IPS flag, resolution, SSD/HDD/Flash storage in GB).
- Engineered **13 new features**, including `PPI` (pixel density), `CPU_Score`, `RAM_x_SSD`, `RAM_x_CPU`, `Total_Storage`, `SSD_ratio`, `Is_Premium`, and log-transformed variants of skewed columns.
- Label-encoded all 7 categorical columns (fit jointly on train+test to avoid unseen-label errors) — chosen over one-hot encoding since tree-based models handle integer categories natively.
- Applied `StandardScaler` for linear models only; tree-based models trained on raw values.
- Final feature set: **28 engineered features**.

### 3. Modelling
Six models were trained and compared with 5-fold cross-validation:

| Model | CV R² | CV RMSE (€) |
|---|---|---|
| Linear Regression | 0.7728 | 336.12 |
| Ridge (α=10) | 0.7740 | 335.32 |
| Lasso (α=1) | 0.7732 | 335.81 |
| Decision Tree (depth=8) | 0.7314 | 359.35 |
| Polynomial Regression (deg=2) | 0.7426 | 356.59 |
| **Gradient Boosting ✅ (primary)** | **0.9001** | **224.75** |

The Gradient Boosting model is trained on `log1p(Price)` and predictions are inverse-transformed with `expm1()`.

### 4. Learning Curve
A learning curve on Linear Regression shows a classic high-bias (underfitting) pattern — training and validation error converge and plateau well above zero, confirming that a more expressive, non-linear model (Gradient Boosting) is needed rather than simply gathering more data.

### 5. Hyperparameter Tuning
`GridSearchCV` (5-fold, 36 combinations × 5 folds) was used to tune the Gradient Boosting Regressor:

| Hyperparameter | Values Tried | Best Value |
|---|---|---|
| `n_estimators` | 300, 500 | 500 |
| `learning_rate` | 0.03, 0.05, 0.08 | 0.05 |
| `max_depth` | 4, 5, 6 | 4 |
| `min_samples_leaf` | 3, 5 | 3 |
| `subsample` | 0.8 (fixed) | 0.8 |

### 6. Final Model Evaluation

| Metric | Value |
|---|---|
| **R² Score** | 0.9001 |
| **RMSE** | €224.75 |
| **MAE** | €145.18 |
| Kaggle Target R² | 0.730 (exceeded by +17.0 pp) |

The most important features are `RAM_x_SSD`, `RAM (GB)`, `CPU_Score`, and `log_RAM`, followed by brand-level signals (`Is_Premium`, `Company_enc`). `HDD_GB` and `Flash_GB` contribute negligible predictive power.

## 🏆 Final Model Summary

| Aspect | Detail |
|---|---|
| Algorithm | Gradient Boosting Regressor (scikit-learn) |
| Training target | `log1p(Price)`, inverse-transformed with `expm1()` |
| Best parameters | `n_estimators=500, learning_rate=0.05, max_depth=4, min_samples_leaf=3, subsample=0.8` |
| Feature count | 28 |
| Validation strategy | 5-fold `KFold` cross-validation (shuffled, `random_state=42`) |

## 📁 Repository Structure

```
.
├── notebook.ipynb                              # Full analysis: EDA, feature engineering, modelling, tuning
├── submission_v2.csv                           # Final test-set price predictions
├── laptop_price_prediction_report_FINAL.docx   # Full written report with charts & discussion
└── README.md
```

## 🛠️ Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — modelling (`LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `GradientBoostingRegressor`, `GridSearchCV`, `Pipeline`, `PolynomialFeatures`)
- `matplotlib`, `seaborn` — visualization
- `scipy` — statistics (skewness)

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
   ```
3. Launch the notebook
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. Run all cells to reproduce the EDA, feature engineering, model training, tuning, and final predictions (`submission_v2.csv`).

## 📄 Report

See [`laptop_price_prediction_report_FINAL.docx`](./laptop_price_prediction_report_FINAL.docx) for the full write-up with all charts, tables, and detailed analysis.
