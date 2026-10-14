"""
ml_core.py - Core Multiple Linear Regression & Feature Selection Engine in Python (scikit-learn)
==============================================================================================
Implements the 5 feature selection methods:
1. Lasso (L1 Ranking): Feature ranking based on absolute Lasso coefficients
2. Forward SFS: Sequential Forward Selection minimizing CV RMSE
3. RFE: Recursive Feature Elimination based on linear regression coefficients
4. ElasticNet Ranking: Feature ranking based on ElasticNet (L1 + L2) coefficients
5. Exhaustive Search: Exhaustive combination search (Best subset selection)

Can be executed as:
  python ml_core.py --dataset startups
  python ml_core.py --dataset boston
  python ml_core.py --all                # precomputes both datasets to JSON for web deployment
  python ml_core.py --input path/to.csv --target col_name
"""

import sys
import os
import json
import itertools
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

def load_and_preprocess(filepath_or_buffer, target_col=None, drop_first=True):
    """
    Load CSV, handle missing values, one-hot encode categorical features,
    and return dataset dictionary.
    """
    if isinstance(filepath_or_buffer, str):
        df = pd.read_csv(filepath_or_buffer)
    else:
        df = pd.read_csv(filepath_or_buffer)

    df = df.dropna().copy()

    # If target not provided, pick the last numeric column
    if target_col is None or target_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in dataset.")
        target_col = num_cols[-1]

    y = df[target_col].values.astype(float)
    X_raw = df.drop(columns=[target_col])

    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X_raw, drop_first=drop_first, dtype=float)
    feature_names = list(X_encoded.columns)
    X = X_encoded.values.astype(float)

    return {
        "df": df,
        "X_df": X_encoded,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_name": target_col,
        "n_samples": len(y),
        "n_features": len(feature_names)
    }

class FastCVEvaluator:
    """
    Precomputes CV folds and Gram matrices so that subset scoring is lightning fast.
    """
    def __init__(self, X, y, cv_folds=5, random_state=42):
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.splits = list(cv.split(X))
        
        # Precompute design matrices and Gram matrices for each fold
        self.fold_data = []
        for train_idx, test_idx in self.splits:
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]
            
            # Augmented with intercept: [1, x1, x2, ...]
            X_tr_aug = np.column_stack([np.ones(len(train_idx)), X_tr])
            X_te_aug = np.column_stack([np.ones(len(test_idx)), X_te])
            
            XtX = X_tr_aug.T @ X_tr_aug
            Xty = X_tr_aug.T @ y_tr
            
            self.fold_data.append({
                "test_idx": test_idx,
                "X_te_aug": X_te_aug,
                "y_te": y_te,
                "XtX": XtX,
                "Xty": Xty
            })
            
    def score_subset(self, subset_indices):
        """
        Fits OLS on each fold and computes out-of-fold RMSE, R2, and MAE.
        """
        # subset_indices are 0-indexed into original X features.
        # Intercept is index 0; feature j is index j + 1.
        aug_idx = [0] + [j + 1 for j in subset_indices]
        
        all_preds = np.zeros(self.n)
        for fold in self.fold_data:
            A = fold["XtX"][np.ix_(aug_idx, aug_idx)]
            b = fold["Xty"][aug_idx]
            # Ridge regularization jitter for numerical stability
            if len(aug_idx) > 1:
                A = A + np.eye(len(aug_idx)) * 1e-9
            beta = np.linalg.solve(A, b)
            
            pred = fold["X_te_aug"][:, aug_idx] @ beta
            all_preds[fold["test_idx"]] = pred
            
        rmse = float(np.sqrt(mean_squared_error(self.y, all_preds)))
        r2 = float(r2_score(self.y, all_preds))
        mae = float(mean_absolute_error(self.y, all_preds))
        return rmse, r2, mae, all_preds

def run_feature_selection(data_dict, kmax=None, cv_folds=5, random_state=42, l1_ratio=0.5, alpha_ratio=0.05):
    """
    Runs the 5 feature selection methods across k = 1..kmax:
    - Lasso (L1 Ranking)
    - Forward SFS
    - RFE
    - ElasticNet Ranking
    - Exhaustive Search
    """
    X = data_dict["X"]
    y = data_dict["y"]
    feature_names = data_dict["feature_names"]
    n, p = X.shape

    if kmax is None or kmax > p:
        kmax = min(p, 5) if p >= 5 else p

    # Standardize X and center y for regularized ranking models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_mean = np.mean(y)
    y_centered = y - y_mean

    evaluator = FastCVEvaluator(X, y, cv_folds=cv_folds, random_state=random_state)

    # -------------------------------------------------------------
    # 1. Lasso (L1 Ranking)
    # -------------------------------------------------------------
    alpha_max = np.max(np.abs(X_scaled.T @ y_centered)) / n
    lasso_alpha = max(alpha_max * alpha_ratio, 1e-4)
    lasso = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=random_state)
    lasso.fit(X_scaled, y_centered)
    lasso_ranking_idx = np.argsort(-np.abs(lasso.coef_)).tolist()

    # -------------------------------------------------------------
    # 2. ElasticNet Ranking
    # -------------------------------------------------------------
    enet_alpha = max(alpha_max * alpha_ratio, 1e-4)
    enet = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state)
    enet.fit(X_scaled, y_centered)
    enet_ranking_idx = np.argsort(-np.abs(enet.coef_)).tolist()

    # -------------------------------------------------------------
    # 3. RFE (Recursive Feature Elimination)
    # -------------------------------------------------------------
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    rfe.fit(X_scaled, y)
    rfe_ranking_idx = np.argsort(rfe.ranking_).tolist()

    # -------------------------------------------------------------
    # 4. Forward SFS (Sequential Forward Selection)
    # -------------------------------------------------------------
    sfs_selected_idx = []
    available = set(range(p))
    for step in range(kmax):
        best_rmse = float('inf')
        best_f = None
        for candidate in available:
            current_subset = sfs_selected_idx + [candidate]
            rmse, _, _, _ = evaluator.score_subset(current_subset)
            if rmse < best_rmse:
                best_rmse = rmse
                best_f = candidate
        sfs_selected_idx.append(best_f)
        available.remove(best_f)

    # -------------------------------------------------------------
    # 5. Exhaustive Search (Combinations)
    # -------------------------------------------------------------
    exhaustive_subsets = []
    for k in range(1, kmax + 1):
        best_rmse = float('inf')
        best_comb = None
        for comb in itertools.combinations(range(p), k):
            comb_list = list(comb)
            rmse, _, _, _ = evaluator.score_subset(comb_list)
            if rmse < best_rmse:
                best_rmse = rmse
                best_comb = comb_list
        exhaustive_subsets.append(best_comb)

    # -------------------------------------------------------------
    # Performance Evaluation for each method across k = 1..kmax
    # -------------------------------------------------------------
    methods_config = [
        {"id": "lasso", "name": "Lasso (L1 Ranking)", "color": "#38bdf8", "ranking": lasso_ranking_idx},
        {"id": "sfs", "name": "Forward SFS", "color": "#34d399", "ranking": sfs_selected_idx},
        {"id": "rfe", "name": "RFE", "color": "#fb923c", "ranking": rfe_ranking_idx},
        {"id": "enet", "name": "ElasticNet Ranking", "color": "#f472b6", "ranking": enet_ranking_idx},
        {"id": "exh", "name": "Exhaustive Search", "color": "#a78bfa", "ranking": None}
    ]

    results_by_method = []
    for m in methods_config:
        per_k = []
        for k in range(1, kmax + 1):
            if m["id"] == "exh":
                subset = exhaustive_subsets[k - 1]
            else:
                subset = m["ranking"][:k]

            subset_features = [feature_names[i] for i in subset]

            rmse, r2, mae, _ = evaluator.score_subset(subset)

            # Also fit on full data to get model parameters
            lr = LinearRegression()
            lr.fit(X[:, subset], y)
            coefs = {feat: float(coef) for feat, coef in zip(subset_features, lr.coef_)}
            intercept = float(lr.intercept_)

            per_k.append({
                "k": k,
                "subset_indices": subset,
                "subset_features": subset_features,
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "intercept": round(intercept, 4),
                "coefficients": {k: round(v, 4) for k, v in coefs.items()}
            })

        results_by_method.append({
            "id": m["id"],
            "name": m["name"],
            "color": m["color"],
            "per_k": per_k,
            "feature_ranking": [feature_names[i] for i in m["ranking"]] if m["ranking"] else None
        })

    # Baseline (mean-only model)
    baseline_rmse = float(np.sqrt(np.mean((y - np.mean(y))**2)))

    # Compute correlation matrix
    df_for_corr = pd.DataFrame(X, columns=feature_names)
    df_for_corr[data_dict["target_name"]] = y
    corr_matrix = df_for_corr.corr().round(4).to_dict()

    return {
        "dataset_name": data_dict.get("name", "dataset"),
        "target_name": data_dict["target_name"],
        "n_samples": n,
        "n_features": p,
        "kmax": kmax,
        "all_feature_names": feature_names,
        "baseline_rmse": round(baseline_rmse, 4),
        "lasso_coefficients": {feat: round(float(c), 4) for feat, c in zip(feature_names, lasso.coef_)},
        "enet_coefficients": {feat: round(float(c), 4) for feat, c in zip(feature_names, enet.coef_)},
        "methods": results_by_method,
        "correlation": corr_matrix
    }

def run_both_datasets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    startups_path = os.path.join(base_dir, "data", "50_startups.csv")
    boston_path = os.path.join(base_dir, "data", "boston_housing.csv")

    print("[1/2] Analyzing 50 Startups dataset...", flush=True)
    d_startups = load_and_preprocess(startups_path, target_col="Profit")
    d_startups["name"] = "50_startups"
    res_startups = run_feature_selection(d_startups, kmax=5, cv_folds=5, random_state=42)

    print("[2/2] Analyzing Boston Housing dataset...", flush=True)
    d_boston = load_and_preprocess(boston_path, target_col="medv")
    d_boston["name"] = "boston_housing"
    res_boston = run_feature_selection(d_boston, kmax=5, cv_folds=5, random_state=42)

    combined = {
        "startups": res_startups,
        "boston": res_boston
    }

    # Save to data/precomputed.json
    out_json = os.path.join(base_dir, "data", "precomputed.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved precomputed results to: {out_json}", flush=True)

    # Also save as js/precomputed.js so the page works via file:// or static GitHub Pages
    out_js = os.path.join(base_dir, "js", "precomputed.js")
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated precomputed scikit-learn results */\n")
        f.write("window.PRECOMPUTED = " + json.dumps(combined, indent=2, ensure_ascii=False) + ";\n")
    print(f"Saved JS export to: {out_js}", flush=True)

    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiple Linear Regression Feature Selection Engine")
    parser.add_argument("--all", action="store_true", help="Precompute both built-in datasets to JSON/JS")
    parser.add_argument("--dataset", choices=["startups", "boston"], help="Run on one built-in dataset")
    parser.add_argument("--input", type=str, help="Path to custom CSV dataset")
    parser.add_argument("--target", type=str, help="Target column name")
    parser.add_argument("--kmax", type=int, default=5, help="Maximum number of features")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if args.all or (not args.dataset and not args.input):
        run_both_datasets()
    elif args.dataset == "startups":
        d = load_and_preprocess("data/50_startups.csv", target_col="Profit")
        res = run_feature_selection(d, kmax=args.kmax, cv_folds=args.cv)
        print(json.dumps(res, indent=2))
    elif args.dataset == "boston":
        d = load_and_preprocess("data/boston_housing.csv", target_col="medv")
        res = run_feature_selection(d, kmax=args.kmax, cv_folds=args.cv)
        print(json.dumps(res, indent=2))
    elif args.input:
        d = load_and_preprocess(args.input, target_col=args.target)
        res = run_feature_selection(d, kmax=args.kmax, cv_folds=args.cv)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved results to {args.output}")
        else:
            print(json.dumps(res, indent=2))
