# Feature Selection Lab — Multiple Linear Regression

Interactive web page comparing **5 feature-selection methods** for multiple linear regression on
**50 Startups** and **Boston Housing** — or **any CSV you upload**. Everything runs in the browser
(pure JavaScript + Chart.js), so it deploys to GitHub Pages as a static site.

## Features
- Dataset dropdown: 50 Startups · Boston Housing · Upload your own CSV (auto type detection + one-hot encoding)
- **RMSE vs Number of Features** and **R² vs Number of Features** charts for
  Lasso (L1 Ranking), Forward SFS, RFE, ElasticNet Ranking, Exhaustive Search
- K-fold (repeated) cross-validation or hold-out train/test evaluation, adjustable seed, α, L1 ratio, max features
- Auto-generated insights (中文 / English): best k, best method, gap vs exhaustive optimum
- Feature selection map, feature rankings, Lasso vs ElasticNet coefficient chart
- Model explorer: regression equation, actual-vs-predicted, residuals, standardized coefficients,
  statsmodels-style OLS table (SE, t, p-values, Adj. R², F, AIC, BIC)
- Correlation heatmap, data preview, CSV export, PNG chart download

## Run locally
Just open `index.html`, or serve the folder:
```bash
python -m http.server 8000   # then visit http://localhost:8000
```

## Deploy on GitHub Pages
```bash
git init
git add .
git commit -m "Feature selection lab"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
Then on GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → `main` / `(root)`**.
The site will be available at `https://<user>.github.io/<repo>/`.

## Methodology
| Method | How subsets of size k are chosen |
|---|---|
| Lasso (L1 Ranking) | Rank by \|coef\| of Lasso on standardized X at α (ties → regularization-path entry order) |
| Forward SFS | Greedily add the feature minimizing CV RMSE |
| RFE | Repeatedly drop the feature with smallest \|standardized OLS coef\| |
| ElasticNet Ranking | Same as Lasso with mixed L1/L2 penalty |
| Exhaustive Search | Evaluate all C(p, k) subsets, keep best CV RMSE |

Each chosen subset is refit with OLS and scored with pooled out-of-fold RMSE and R²
(hold-out mode: selection uses inner CV on training data; scores from the test set).

## Files
```
index.html        page layout
css/style.css     design system
js/ml.js          regression + feature-selection engine
js/app.js         UI, charts, insights
js/datasets.js    embedded built-in datasets
data/*.csv        original CSV files
```
