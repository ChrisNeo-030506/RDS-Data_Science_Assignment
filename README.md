# Apartment Rent Prediction (CRISP-DM)

**BMDS2003 Data Science Project — Group 4**

A regression project that predicts the monthly rent (USD) of a US apartment.
Dataset: UCI *Apartment for Rent Classified* (100K version).
Four models are compared — Linear Regression (baseline), Decision Tree, Random Forest,
and Hist Gradient Boosting — and the best compact model is deployed as a Streamlit web app.

---

## 1. Project Structure

```
RDS-Data_Science_Assignment/
│
├── BMDS2003_Group4_notebook.ipynb            # Main notebook — full CRISP-DM workflow (EDA → cleaning → 4 models → evaluation → model export)
├── apartments_for_rent_classified_100K.csv   # Raw dataset (100K rows, ';' separated, cp1252 encoded)
├── README.md                                 # This file
├── .gitignore                                # Files excluded from Git (.venv, __pycache__, OS junk, etc.)
├── .gitattributes                            # Line-ending normalisation + binary file rules
│
├── eda_graphs/                           # Graphs produced in the Data Understanding / EDA stage
│   ├── eda_price_hist.png                # Distribution of monthly rent (right-skewed)
│   ├── eda_boxplots.png                  # Boxplots of rent & square feet — justifies IQR outlier removal
│   ├── eda_correlation.png               # Correlation heatmap of the numeric features
│   ├── eda_price_vs_sqft.png             # Scatter: rent vs square feet (positive, non-linear)
│   └── eda_rent_by_state.png             # Average rent by state (top 15 states by listing count)
│
├── model_graphs/                         # Three diagnostic graphs per model (m1–m4)
│   ├── m1_pred_vs_actual.png             # Model 1 Linear Regression — predicted vs actual
│   ├── m1_residual_plot.png              # Model 1 Linear Regression — residuals vs predicted
│   ├── m1_residual_hist.png              # Model 1 Linear Regression — residual distribution
│   ├── m2_validation_curve.png           # Model 2 Decision Tree — train/test R² vs max_depth
│   ├── m2_feature_importance.png         # Model 2 Decision Tree — feature importance
│   ├── m2_mae_by_tier.png                # Model 2 Decision Tree — MAE by rent tier
│   ├── m3_feature_importance.png         # Model 3 Random Forest — feature importance
│   ├── m3_residual_by_bedroom.png        # Model 3 Random Forest — residuals grouped by bedrooms
│   ├── m3_hexbin.png                     # Model 3 Random Forest — predicted vs actual (density)
│   ├── m4_learning_curve.png             # Model 4 Hist Gradient Boosting — learning curve
│   ├── m4_residual_plot.png              # Model 4 Hist Gradient Boosting — residuals vs predicted
│   └── m4_abs_error_hist.png             # Model 4 Hist Gradient Boosting — absolute error distribution
│
└── streamlit/                            # Deployment prototype (the web app)
    ├── app.py                            # The Streamlit app — sidebar inputs → predicted monthly rent
    ├── train_model.py                    # Retrains the model and regenerates the 3 .joblib files
    ├── rent_model.joblib                 # Trained Hist Gradient Boosting model (~0.4 MB)
    ├── model_columns.joblib              # Training feature columns, used to align the input row
    ├── state_geo.joblib                  # Per-state median latitude / longitude
    ├── requirements.txt                  # Python dependencies for the app
    └── README.md                         # Notes specific to the Streamlit app
```

> A `.venv/` folder may appear locally after setup — it is the virtual environment and is
> ignored by Git, so it is not part of the repository.

---

## 2. How to Open the Notebook (`.ipynb`)

Pick **either** option. Option A is best if you want to run everything locally;
Option B requires nothing to be installed.

### Option A — Anaconda (local, recommended)

1. Download and install **Anaconda** from <https://www.anaconda.com/download> (Python 3.10+).
2. Open **Anaconda Navigator** → launch **Jupyter Notebook** (or **JupyterLab**).
   *Alternatively, open Anaconda Prompt and run `jupyter notebook`.*
3. In the browser file tree, navigate to this project folder and click
   **`BMDS2003_Group4_notebook.ipynb`**.
4. Make sure `apartments_for_rent_classified_100K.csv` is in the **same folder** as the
   notebook — the notebook reads it with a relative path.
5. Install the required packages once (in Anaconda Prompt):
   ```
   pip install pandas numpy matplotlib scikit-learn joblib
   ```
6. Run the notebook top to bottom: **Cell → Run All**, or press `Shift + Enter` cell by cell.

### Option B — Google Colab (online, no installation)

1. Go to <https://colab.research.google.com> and sign in with a Google account.
2. Choose **File → Upload notebook** and upload `BMDS2003_Group4_notebook.ipynb`.
   *(Or **File → Open notebook → GitHub** and paste this repository's URL.)*
3. Upload the dataset: click the **folder icon** in the left sidebar → **Upload to session
   storage** → select `apartments_for_rent_classified_100K.csv`.
   The file is ~100 MB, so give it a moment to finish.
   > Uploaded files are deleted when the session ends. To avoid re-uploading, keep the CSV in
   > Google Drive and mount it instead:
   > ```python
   > from google.colab import drive
   > drive.mount('/content/drive')
   > ```
   > then point the `pd.read_csv(...)` path to the file in your Drive.
4. Run everything with **Runtime → Run all** (or `Ctrl + F9`).
   All libraries used are pre-installed in Colab, so no `pip install` is needed.

**Note:** the Random Forest cell trains on ~100K rows and may take a few minutes.

---

## 3. How to Run the Streamlit App

The app loads the pre-trained `.joblib` files inside `streamlit/`, supporting all **4 Machine Learning Models** (Hist Gradient Boosting, Random Forest, Decision Tree, Linear Regression Baseline).

### Step-by-Step Guide:

1. **Activate the existing virtual environment** (from the project root):
   ```bash
   source .venv/bin/activate        # macOS / Linux
   # or: .venv\Scripts\activate     # Windows
   ```

2. **Navigate into the `streamlit` folder**:
   ```bash
   cd streamlit
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   *(Alternatively, run in one command from inside `streamlit/`: `../.venv/bin/streamlit run app.py` on macOS)*

4. **Access the Web Dashboard**:
   The app will automatically open in your browser at <http://localhost:8501>.
   - Select any of the **4 Machine Learning Models** or choose **Compare All 4 Models Simultaneously**.
   - Pick your target **State** and **City** (auto-loads median price & GPS coordinates).
   - Adjust square footage, bedrooms, bathrooms, and select property amenities.
   - View real-time valuation, confidence ranges, unit rates, spatial map pin, and multi-model consensus charts!

5. To stop the web server, press `Ctrl + C` in the terminal.


### Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | The install went to a different Python. Run `python -m streamlit run app.py` instead. |
| Version warning or error while loading the model | Your scikit-learn version differs from the one that saved the model. Regenerate the artifacts with `python train_model.py` (needs `apartments_for_rent_classified_100K.csv` in the parent folder), then repeat step 4. |
| `FileNotFoundError: rent_model.joblib` | You are not inside the `streamlit` folder. Run `cd streamlit` first. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |

---

## Model Results

| Model | Notes |
|---|---|
| Linear Regression | Baseline — underfits the non-linear rent/size relationship |
| Decision Tree (`max_depth=12`) | Depth chosen from the validation curve |
| **Random Forest** | **Most accurate (Test R² ≈ 0.86)**, but the saved model is ~140 MB |
| Hist Gradient Boosting | Test R² ≈ 0.77 with a ~0.4 MB model file — **deployed in the Streamlit app** |

Random Forest wins on accuracy, but Hist Gradient Boosting is deployed because it is roughly
350× smaller, making it far more practical to ship and load in the web app.
