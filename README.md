# Feature Selection Lab — Multiple Linear Regression (Python & Web)

Interactive lab and web application comparing **5 feature-selection methods** for multiple linear regression on
**50 Startups** and **Boston Housing** — or **any custom CSV dataset**.

All machine learning computations are implemented in **Python** using **scikit-learn**, with precomputed JSON for instant web visualization and full GitHub Pages compatibility.

---

## 🔬 Five Feature Selection Methods Compared

| Method | Algorithm Mechanism |
|---|---|
| **Lasso (L1 Ranking)** | Shrinkage regression with L1 penalty; features ranked by absolute standardized coefficients $\|w_j\|$ |
| **Forward SFS** | Sequential Forward Selection: greedily adds the feature that yields lowest cross-validation RMSE |
| **RFE** | Recursive Feature Elimination: iteratively removes the feature with smallest standardized OLS coefficient |
| **ElasticNet Ranking** | Combined L1 + L2 regularization; correlated features share weights and are ranked by $\|w_j\|$ |
| **Exhaustive Search** | Best subset selection: globally evaluates all $\binom{p}{k}$ combinations to find the exact optimum |

---

## 📊 Dual Visualization Requirements

1. **Left Chart: RMSE Comparison across 5 Methods**
   - **X-axis**: Number of Features ($k = 1$ to $5$)
   - **Y-axis**: RMSE (Lower is Better)
   - **Key trend**: Minimum error reached around $k = 2 \sim 3$ features (~8,200–9,200 depending on split). Exhaustive Search (purple line) consistently finds the global minimum. Adding all 5 features causes RMSE to rise due to noise/overfitting from redundant variables.

2. **Right Chart: $R^2$ Comparison across 5 Methods**
   - **X-axis**: Number of Features ($k = 1$ to $5$)
   - **Y-axis**: $R^2$ (Higher is Better)
   - **Key trend**: Inverses the RMSE curve. Exhaustive Search achieves the highest explanatory power ($R^2 > 0.94$). When all 5 features are added, $R^2$ declines to ~0.936.

---

## 🚀 Running the Python Machine Learning Code

### 1. Run Python CLI on built-in or custom datasets
```bash
# Analyze both built-in datasets and export precomputed JSON for the web UI
python ml_core.py --all

# Analyze only 50 Startups
python ml_core.py --dataset startups

# Analyze only Boston Housing
python ml_core.py --dataset boston

# Analyze ANY custom dataset
python ml_core.py --input path/to/dataset.csv --target target_col_name --kmax 5
```

### 2. Run the Step-by-Step Jupyter Notebook
Open `feature_selection_lab.ipynb` in VS Code or Jupyter:
```bash
jupyter notebook feature_selection_lab.ipynb
```
Contains full data loading, one-hot encoding, model fitting, and matplotlib publication-quality plots.

---

## 🌐 Web Application & Deployment on GitHub Pages

The web page is static and can be deployed directly to GitHub Pages without needing a server:

### Local preview
```bash
python -m http.server 8765
```
Visit: `http://localhost:8765/`

### Deploy to GitHub Pages
```bash
git init
git add .
git commit -m "Multiple Linear Regression Feature Selection Lab"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Then on GitHub:
1. Navigate to **Settings** → **Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Choose branch **`main`** and folder **`/(root)`**, then click **Save**.
4. The live site will be ready at: `https://<your-username>.github.io/<your-repo>/`.

---

## 📁 Repository Structure

```
├── ml_core.py                    # Core Python ML engine (scikit-learn, cross-validation, 5 methods)
├── feature_selection_lab.ipynb   # Interactive Jupyter notebook with complete steps & plots
├── create_notebook.py            # Utility script to generate the notebook
├── index.html                    # Web interface layout
├── css/
│   └── style.css                 # Responsive modern dark glassmorphism styling
├── js/
│   ├── app.js                    # UI logic, chart rendering, insights & explorer
│   ├── precomputed.js            # Precomputed scikit-learn results for instant loading
│   ├── datasets.js               # Embedded built-in CSVs for offline viewing
│   └── ml.js                     # Fallback in-browser computation engine
└── data/
    ├── 50_startups.csv           # 50 Startups dataset (Profit)
    ├── boston_housing.csv        # Boston Housing dataset (MEDV)
    └── precomputed.json          # Exported scikit-learn metrics
```
