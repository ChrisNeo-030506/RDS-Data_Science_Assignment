# Apartment Rent Predictor — Deployment Prototype

BMDS2003 Data Science Project. A Streamlit web app that predicts the monthly
rent (USD) of a US apartment from its features. Trained on the 100K dataset.
Random Forest is the most accurate of the four models compared (Test R² ≈ 0.86),
but the app deploys the **Hist Gradient Boosting Regressor** (Test R² ≈ 0.77) —
its model file is ~0.4 MB vs ~140 MB for Random Forest, making it far more
practical to ship and load.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app (the deployment prototype) |
| `train_model.py` | Trains the model and regenerates the three `.joblib` files |
| `rent_model.joblib` | Trained Hist Gradient Boosting model |
| `model_columns.joblib` | Training feature columns (aligns the input row) |
| `state_geo.joblib` | Per-state median latitude/longitude |
| `requirements.txt` | Python dependencies |

## How to run
1. (Recommended) create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   source .venv/bin/activate    # macOS / Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. **If loading the model raises a scikit-learn version error**, regenerate the
   artifacts with your local version (needs
   `apartments_for_rent_classified_100K.csv` in the parent folder):
   ```
   python train_model.py
   ```
4. Launch the app:
   ```
   streamlit run app.py
   ```
   Open <http://localhost:8501>. If prompted for an email on first run, just
   press Enter to skip.

## How it works
Enter the apartment's features in the sidebar (size, bedrooms, bathrooms,
amenities, pets, listing photo, broker fee, US state) and click **Predict rent**.
The app builds a single feature row that matches the training columns — the
chosen state fills in that state's median latitude/longitude, which are the
model's strongest predictors — and returns the estimated monthly rent.

Dataset: UCI *Apartment for Rent Classified* (100K version).
