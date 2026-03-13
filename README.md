VC Funding Predictive Analytics Dashboard
AI-Powered Intelligence for Indian Tech Founders
> **"In a funding winter, data is the only warm coat."**
---
Table of Contents
Business Rationale
Product Overview
Architecture
Algorithms & Methodology
Setup & Execution
File Reference
Disclaimer
---
1. Business Rationale
The Indian VC Landscape in 2024–25
India's startup ecosystem crossed $140 billion in cumulative VC funding over the past decade, but the period from 2022 onward has been defined by a sharp correction — a structural funding winter that has forced founders to confront uncomfortable truths about unit economics, capital efficiency, and long-term defensibility.
Three macro forces now govern every term sheet:
Force	Implication for Founders
Profitability-First Mandate	Investors have rotated from GMV / growth metrics to EBITDA, free cash flow, and Burn Multiple < 2.0. Pre-revenue decks are dead in water at Series A.
DPI Pressure on GPs	Limited Partners are demanding Distributions to Paid-In Capital — realised returns, not paper markups. This forces VCs to be selective, shortening the list of fundable archetypes from ~50 to ~10.
Bharat vs. India1 Divergence	The next 300 million internet users (Bharat / India2) represent a fundamentally different unit-economics profile. Founders who conflate "India1 SaaS" metrics with "Bharat mass-market" models systematically miscalibrate their pitches.
Why This Dashboard Exists
Most Indian founders enter fundraising with no quantitative benchmark against their peer cohort. They know their ARR; they don't know whether their Burn Multiple places them in the top or bottom quartile for their sector. They have a pedigree score in their heads; they don't know how investors weight it against IP moat in valuation multiples.
This dashboard operationalises that institutional knowledge — packaging four machine learning models trained on a representative synthetic cohort into a real-time, interactive self-assessment tool.
---
2. Product Overview
```
┌─────────────────────────────────────────────────────────┐
│          VC Funding Analytics Dashboard (Streamlit)      │
├───────────────┬─────────────────────────────────────────┤
│  Sidebar      │  Predictions Panel                       │
│  ─────────    │  ─────────────────────────────────────   │
│  • Sector     │  [Gauge] Probability of Being Funded     │
│  • Demo       │  [Card]  Expected Pre-Money Valuation    │
│  • ARR        │  [Badge] Founder Persona / Archetype     │
│  • Burn Mult  │  [Table] Input Summary                   │
│  • Pedigree   ├─────────────────────────────────────────┤
│  • IP Score   │  Market Intelligence                     │
│  • DPI        │  [Chart 1] Funding by Sector             │
│               │  [Chart 2] Burn Multiple vs. Valuation   │
└───────────────┴─────────────────────────────────────────┘
```
---
3. Architecture
```
generate_data.py
     │
     ▼
raw_startup_data.csv   (5,000 rows, dirty)
     │
     ▼
train_models.py
     ├── Data Cleaning  →  clean_startup_data.csv
     │
     ├── RandomForest   →  models/rf_classifier.pkl
     ├── KMeans         →  models/kmeans.pkl + persona_labels.pkl
     ├── Apriori        →  models/association_rules.pkl
     ├── XGBoost/LR     →  models/valuation_regressor.pkl
     └── Preprocessors  →  models/scaler.pkl, le_*.pkl
     │
     ▼
app.py  (streamlit run app.py)
     └── Loads .pkl models + clean_startup_data.csv
         └── Renders interactive dashboard
```
---
4. Algorithms & Methodology
4.1 Classification — Random Forest (Funding Status Prediction)
Target: `Funding_Status` ∈ {Funded, Bootstrapped, Dead}
Why Random Forest? The feature space is mixed (categorical + continuous), has moderate dimensionality, and benefits from ensemble variance reduction. Random Forest handles class imbalance via `class_weight='balanced'` and provides feature importance natively.
Key hyperparameters:
`n_estimators = 300` — sufficient trees for stable out-of-bag error
`max_depth = 12` — prevents overfitting on noisy synthetic features
`min_samples_leaf = 5` — smooths decision boundaries
Output to dashboard: `predict_proba()` vector → "Probability of Getting Funded" gauge.
---
4.2 Clustering — K-Means Founder Personas
Why K-Means? The goal is interpretable, centroid-based archetypes that a non-technical founder can relate to. K=4 was selected via elbow method analysis and domain logic (we expect 4 canonical archetypes in the Indian market).
Four Archetypes:
Cluster	Label	Signature
0	🚀 Venture-Scale Disruptor	High ARR, high pedigree, high burn, AI/Fintech
1	⚙️ Deep-Tech Builder	High IP score, lower ARR, patient capital profile
2	🌾 Bharat / Mass-Market Pioneer	Bharat demographic, high CAC efficiency, agritech/D2C
3	💼 Capital-Efficient Bootstrapper	Low burn, moderate ARR, SaaS/logistics
---
4.3 Association Rule Mining — Apriori Algorithm
Purpose: Discovers non-obvious co-occurrence patterns between startup attributes and funding outcomes.
Methodology:
Continuous variables (ARR, Burn Multiple, Pedigree) are binarised at their medians.
`mlxtend.apriori` extracts frequent itemsets with `min_support = 0.05`.
`association_rules()` filters by `lift ≥ 1.2` — ensuring rules exceed random co-occurrence.
Example rules discovered (representative):
```
[AI, Pedigree_High]          →  [Funded]   lift: 2.1
[DPI_Yes, Burn_Low]          →  [Funded]   lift: 1.8
[Burn_Multiple_High, ARR_Lo] →  [Dead]     lift: 2.4
```
These rules provide qualitative signal to founders beyond the primary classification model.
---
4.4 Regression — XGBoost Valuation Forecasting
Target: `Expected_Pre_Money_Valuation_USD` (log-transformed before training, exponentiated for output)
Why XGBoost? Valuation is a skewed, non-linear function of multiple interacting features. Gradient boosting captures interaction terms (e.g., Pedigree × ARR) that linear models miss. Log-transformation of the target normalises the distribution and stabilises MAPE.
Fallback: If XGBoost is unavailable, the pipeline falls back to `sklearn.LinearRegression` automatically.
Evaluation Metric: Mean Absolute Percentage Error (MAPE) — appropriate for right-skewed financial targets.
---
5. Setup & Execution
Prerequisites
```bash
Python >= 3.9
```
Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```
Step 2 — Generate Synthetic Data
```bash
python generate_data.py
```
Expected output:
```
⏳  Generating 5,000 synthetic Indian startup records …
💉  Injecting dirty data …
✅  Saved 5,150 rows → raw_startup_data.csv
```
Step 3 — Train All Models
```bash
python train_models.py
```
Expected output:
```
STEP 1 – Loading raw data
STEP 2 – Cleaning
STEP 3 – Feature Engineering
STEP 4a – Random Forest Classification
STEP 4b – KMeans Clustering
STEP 4c – Apriori Association Rules
STEP 4d – Regression (Valuation Forecast)
ALL MODELS TRAINED & SAVED → /models/
```
> ⚠️ Training takes approximately **60–120 seconds** on a standard laptop (M1/Intel i7).
Step 4 — Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`
---
6. File Reference
```
vc-dashboard/
│
├── requirements.txt               # Pinned Python dependencies
├── generate_data.py               # Synthetic data generator (5,000 startups)
├── train_models.py                # Full ML training pipeline
├── app.py                         # Streamlit dashboard
├── README.md                      # This file
│
├── raw_startup_data.csv           # Generated by generate_data.py
├── clean_startup_data.csv         # Generated by train_models.py
│
└── models/
    ├── rf_classifier.pkl          # Random Forest (Funding Status)
    ├── kmeans.pkl                 # K-Means (Founder Personas)
    ├── persona_labels.pkl         # Cluster → Label mapping
    ├── association_rules.pkl      # Apriori rules DataFrame
    ├── valuation_regressor.pkl    # XGBoost/LR (Valuation)
    ├── scaler.pkl                 # StandardScaler
    ├── le_sector.pkl              # LabelEncoder (Sector)
    ├── le_demo.pkl                # LabelEncoder (Demographic)
    └── le_status.pkl              # LabelEncoder (Funding Status)
```
---
7. Disclaimer
> This dashboard is built on **synthetically generated data** using probabilistic models designed to approximate — but not replicate — real-world Indian startup distributions. All predictions, valuations, and probabilities produced by this tool are model outputs for **educational and analytical demonstration purposes only**.
>
> Nothing in this application constitutes investment advice, fundraising counsel, or a guarantee of VC interest. Founders are advised to supplement these insights with direct engagement with investors, CA/legal advisors, and sector-specific due diligence.
>
> Anthropic's Claude was used as a development assistant in producing this codebase.
---
Built with ❤️ for the Indian startup ecosystem.
Streamlit · scikit-learn · XGBoost · Plotly · mlxtend · Faker
