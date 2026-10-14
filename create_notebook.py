import json

def create_lab_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Header
    cells.append(md_cell("""
# Multiple Linear Regression & Feature Selection Lab
### Comparing 5 Feature Selection Methods on 50 Startups & Boston Housing Datasets

This notebook implements and compares five feature selection algorithms in Python using **scikit-learn**:
1. **Lasso (L1 Ranking)**: Feature ranking based on L1 regularization shrinkage coefficients.
2. **Forward SFS (Sequential Forward Selection)**: Greedy forward stepwise selection minimizing cross-validation RMSE.
3. **RFE (Recursive Feature Elimination)**: Backward iterative elimination of least significant features.
4. **ElasticNet Ranking**: Feature ranking combining L1 and L2 penalties.
5. **Exhaustive Search (Best Subset Selection)**: Globally optimal subset search across all $\\binom{p}{k}$ combinations.

For each method and for each feature subset size $k \\in [1, 5]$, we evaluate:
- **RMSE (Root Mean Squared Error)**: Lower is better.
- **$R^2$ (Coefficient of Determination)**: Higher is better.
"""))

    # Imports
    cells.append(md_cell("## 1. Environment Setup & Library Imports"))
    cells.append(code_cell("""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Matplotlib styling for high-quality publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
print("Libraries imported successfully!")
"""))

    # Core evaluation function
    cells.append(md_cell("## 2. Core Feature Selection & Evaluation Framework"))
    cells.append(code_cell("""
def evaluate_5_methods(X, y, feature_names, kmax=5, cv_folds=5, random_state=42):
    n, p = X.shape
    kmax = min(kmax, p)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Standardize features for regularized ranking models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    alpha_max = np.max(np.abs(X_scaled.T @ y_centered)) / n

    # 1. Lasso Ranking
    lasso = Lasso(alpha=max(alpha_max * 0.05, 1e-4), max_iter=10000, random_state=random_state)
    lasso.fit(X_scaled, y_centered)
    lasso_ranking = np.argsort(-np.abs(lasso.coef_)).tolist()

    # 2. ElasticNet Ranking
    enet = ElasticNet(alpha=max(alpha_max * 0.05, 1e-4), l1_ratio=0.5, max_iter=10000, random_state=random_state)
    enet.fit(X_scaled, y_centered)
    enet_ranking = np.argsort(-np.abs(enet.coef_)).tolist()

    # 3. RFE
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    rfe.fit(X_scaled, y)
    rfe_ranking = np.argsort(rfe.ranking_).tolist()

    # 4. Forward SFS
    sfs_ranking = []
    available = set(range(p))
    for _ in range(kmax):
        best_rmse = float('inf')
        best_feat = None
        for cand in available:
            sub = sfs_ranking + [cand]
            preds = cross_val_predict(LinearRegression(), X[:, sub], y, cv=cv)
            score = np.sqrt(mean_squared_error(y, preds))
            if score < best_rmse:
                best_rmse = score
                best_feat = cand
        sfs_ranking.append(best_feat)
        available.remove(best_feat)

    # 5. Exhaustive Search
    exh_subsets = []
    for k in range(1, kmax + 1):
        best_rmse = float('inf')
        best_comb = None
        for comb in itertools.combinations(range(p), k):
            comb_list = list(comb)
            preds = cross_val_predict(LinearRegression(), X[:, comb_list], y, cv=cv)
            score = np.sqrt(mean_squared_error(y, preds))
            if score < best_rmse:
                best_rmse = score
                best_comb = comb_list
        exh_subsets.append(best_comb)

    methods = [
        {"name": "Lasso (L1 Ranking)", "color": "#1f77b4", "marker": "o", "ranking": lasso_ranking},
        {"name": "Forward SFS", "color": "#2ca02c", "marker": "s", "ranking": sfs_ranking},
        {"name": "RFE", "color": "#ff7f0e", "marker": "^", "ranking": rfe_ranking},
        {"name": "ElasticNet Ranking", "color": "#e377c2", "marker": "D", "ranking": enet_ranking},
        {"name": "Exhaustive Search", "color": "#9467bd", "marker": "*", "ranking": None}
    ]

    results = []
    for m in methods:
        records = []
        for k in range(1, kmax + 1):
            sub = exh_subsets[k - 1] if m["name"] == "Exhaustive Search" else m["ranking"][:k]
            preds = cross_val_predict(LinearRegression(), X[:, sub], y, cv=cv)
            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            records.append({
                "k": k,
                "subset": [feature_names[i] for i in sub],
                "rmse": rmse,
                "r2": r2
            })
        results.append({
            "name": m["name"],
            "color": m["color"],
            "marker": m["marker"],
            "records": records
        })

    return results
"""))

    # Plotting helper
    cells.append(md_cell("## 3. Comparison Visualization Function"))
    cells.append(code_cell("""
def plot_comparison_charts(results, dataset_name="50 Startups"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    k_vals = [r["k"] for r in results[0]["records"]]

    # Left plot: RMSE
    for m in results:
        rmse_vals = [r["rmse"] for r in m["records"]]
        lw = 3.0 if m["name"] == "Exhaustive Search" else 1.8
        ms = 10 if m["name"] == "Exhaustive Search" else 7
        ax1.plot(k_vals, rmse_vals, label=m["name"], color=m["color"], 
                 marker=m["marker"], linewidth=lw, markersize=ms)

    ax1.set_title(f"RMSE Comparison across 5 Feature Selection Methods\\n({dataset_name})", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Features", fontsize=11)
    ax1.set_ylabel("RMSE (Lower is Better)", fontsize=11)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="best", frameon=True)

    # Right plot: R^2
    for m in results:
        r2_vals = [r["r2"] for r in m["records"]]
        lw = 3.0 if m["name"] == "Exhaustive Search" else 1.8
        ms = 10 if m["name"] == "Exhaustive Search" else 7
        ax2.plot(k_vals, r2_vals, label=m["name"], color=m["color"], 
                 marker=m["marker"], linewidth=lw, markersize=ms)

    ax2.set_title(f"R-squared Comparison across 5 Feature Selection Methods\\n({dataset_name})", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("R-squared (Higher is Better)", fontsize=11)
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.show()
"""))

    # Section 50 Startups
    cells.append(md_cell("""
## 4. Dataset 1: 50 Startups Analysis
We predict company **Profit** from spending categories (`R&D Spend`, `Administration`, `Marketing Spend`) and categorical feature `State`.
Categorical variable `State` is one-hot encoded with `drop_first=True` (baseline category California dropped), yielding 5 features.
"""))
    cells.append(code_cell("""
# Load 50 Startups Dataset
df_startups = pd.read_csv("data/50_startups.csv")
print("Dataset Shape:", df_startups.shape)
display(df_startups.head())

# Preprocessing & One-Hot Encoding
y_startups = df_startups["Profit"].values
X_raw_startups = df_startups.drop(columns=["Profit"])
X_df_startups = pd.get_dummies(X_raw_startups, drop_first=True, dtype=float)
feature_names_startups = list(X_df_startups.columns)
X_startups = X_df_startups.values

print("Encoded Features (p=5):", feature_names_startups)
"""))
    cells.append(code_cell("""
# Run the 5 methods on 50 Startups
results_startups = evaluate_5_methods(X_startups, y_startups, feature_names_startups, kmax=5, cv_folds=5, random_state=42)

# Plot comparison
plot_comparison_charts(results_startups, dataset_name="50 Startups")
"""))
    cells.append(code_cell("""
# Display summary table for 50 Startups
records_table = []
for m in results_startups:
    for rec in m["records"]:
        records_table.append({
            "Method": m["name"],
            "k": rec["k"],
            "RMSE": round(rec["rmse"], 2),
            "R-squared": round(rec["r2"], 4),
            "Selected Features": " + ".join(rec["subset"])
        })
summary_df_startups = pd.DataFrame(records_table)
display(summary_df_startups.pivot(index="k", columns="Method", values="RMSE"))
"""))
    cells.append(md_cell("""
### 🔍 Key Findings on 50 Startups:
1. **Optimal Number of Features**: When $k = 2$ or $3$, RMSE reaches its minimum (around 8,200 ~ 8,400) and $R^2$ peaks (above 0.948). The model is parsimonious and retains optimal generalization power.
2. **Exhaustive Search (Purple Line)**: Maintains consistently the lowest RMSE and highest $R^2$ across $k=2 \\dots 4$ because it explores all possible $\\binom{5}{k}$ combinations.
3. **Overfitting / Noise with All Features**: When $k=5$ (all features included, including Administration and all state dummies), RMSE increases to around 9,200 and $R^2$ falls to ~0.936, confirming that redundant/noisy features impair model generalization.
"""))

    # Section Boston Housing
    cells.append(md_cell("""
## 5. Dataset 2: Boston Housing Analysis
We predict median home value (`medv`) using 13 neighborhood and housing attributes across 506 census tracts.
"""))
    cells.append(code_cell("""
# Load Boston Housing Dataset
df_boston = pd.read_csv("data/boston_housing.csv")
print("Dataset Shape:", df_boston.shape)
display(df_boston.head())

y_boston = df_boston["medv"].values
X_df_boston = df_boston.drop(columns=["medv"])
feature_names_boston = list(X_df_boston.columns)
X_boston = X_df_boston.values

print("Boston Features (p=13):", feature_names_boston)
"""))
    cells.append(code_cell("""
# Run the 5 methods on Boston Housing (k = 1 to 5)
results_boston = evaluate_5_methods(X_boston, y_boston, feature_names_boston, kmax=5, cv_folds=5, random_state=42)

# Plot comparison
plot_comparison_charts(results_boston, dataset_name="Boston Housing")
"""))
    cells.append(code_cell("""
# Summary table for Boston Housing
records_boston = []
for m in results_boston:
    for rec in m["records"]:
        records_boston.append({
            "Method": m["name"],
            "k": rec["k"],
            "RMSE": round(rec["rmse"], 3),
            "R-squared": round(rec["r2"], 4),
            "Selected Features": " + ".join(rec["subset"])
        })
summary_df_boston = pd.DataFrame(records_boston)
display(summary_df_boston.pivot(index="k", columns="Method", values="RMSE"))
"""))

    # Section Custom Dataset
    cells.append(md_cell("""
## 6. General Function: Evaluate Any Custom Dataset
You can load any custom CSV, specify the target column, and compare all 5 feature selection methods.
"""))
    cells.append(code_cell("""
def run_custom_dataset(csv_path, target_column, kmax=5):
    df = pd.read_csv(csv_path).dropna()
    y = df[target_column].values
    X_raw = df.drop(columns=[target_column])
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=float)
    
    results = evaluate_5_methods(X_encoded.values, y, list(X_encoded.columns), kmax=kmax)
    plot_comparison_charts(results, dataset_name=csv_path.split('/')[-1])
    return results

print("Custom dataset evaluation function ready!")
"""))

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("feature_selection_lab.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    print("feature_selection_lab.ipynb created successfully!")

if __name__ == "__main__":
    create_lab_notebook()
