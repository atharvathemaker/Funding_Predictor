"""
train_models.py
---------------
1. Loads raw_startup_data.csv
2. Cleans & engineers features  → clean_startup_data.csv
3. Trains four ML artefacts      → models/*.pkl
   a. RandomForest  (Funding_Status classification)
   b. KMeans        (Founder Persona clustering)
   c. Apriori       (Association Rule Mining)
   d. XGBoost / LR  (Valuation regression)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_percentage_error
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading raw data")
print("=" * 60)
df = pd.read_csv("raw_startup_data.csv")
print(f"  Rows loaded : {len(df):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2 – Cleaning")

# 2a. Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates dropped : {before - len(df)}")

# 2b. Standardise Sector strings
SECTOR_CANONICAL = {
    r"a\.i\.|artificial intelligence|^ai$|^a i$": "AI",
    r"fintech|fin-tech":                          "Fintech",
    r"edtech|ed-tech":                            "Edtech",
    r"deep tech|deep-tech|deeptech":              "DeepTech",
}

def canonicalise_sector(s: str) -> str:
    s_lower = str(s).strip().lower()
    for pattern, canonical in SECTOR_CANONICAL.items():
        if re.search(pattern, s_lower):
            return canonical
    # Title-case anything else already clean
    return str(s).strip().title()

df["Sector"] = df["Sector"].apply(canonicalise_sector)
print(f"  Unique sectors after standardisation : {sorted(df['Sector'].unique())}")

# 2c. Fix string artifacts in Total_Funding_USD  ("1.5M", "300K" → float)
def parse_funding(val):
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().upper().replace(",", "")
    try:
        if val.endswith("M"):
            return float(val[:-1]) * 1_000_000
        if val.endswith("K"):
            return float(val[:-1]) * 1_000
        return float(val)
    except ValueError:
        return np.nan

df["Total_Funding_USD"] = df["Total_Funding_USD"].apply(parse_funding)
print(f"  Total_Funding_USD nulls after parse : {df['Total_Funding_USD'].isna().sum()}")

# 2d. Impute missing Revenue_ARR_USD with sector-level medians
arr_medians = df.groupby("Sector")["Revenue_ARR_USD"].median()
def impute_arr(row):
    if pd.isna(row["Revenue_ARR_USD"]):
        return arr_medians.get(row["Sector"], df["Revenue_ARR_USD"].median())
    return row["Revenue_ARR_USD"]

df["Revenue_ARR_USD"] = df.apply(impute_arr, axis=1)
print(f"  Missing ARR after imputation : {df['Revenue_ARR_USD'].isna().sum()}")

# 2e. Winsorise extreme outliers in Burn_Multiple at 99th percentile
p99 = df["Burn_Multiple"].quantile(0.99)
before_out = (df["Burn_Multiple"] > p99).sum()
df["Burn_Multiple"] = df["Burn_Multiple"].clip(upper=p99)
print(f"  Burn_Multiple capped at p99={p99:.2f}  ({before_out} rows affected)")

# 2f. Ensure numeric types
for col in ["Revenue_ARR_USD", "CAC_USD", "Total_Funding_USD",
            "Expected_Pre_Money_Valuation_USD"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df.to_csv("clean_startup_data.csv", index=False)
print(f"\n  ✅  clean_startup_data.csv saved  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING (shared preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3 – Feature Engineering")

le_sector = LabelEncoder()
le_demo   = LabelEncoder()
le_status = LabelEncoder()

df["Sector_enc"]   = le_sector.fit_transform(df["Sector"])
df["Demo_enc"]     = le_demo.fit_transform(df["Target_Demographic"])
df["Status_enc"]   = le_status.fit_transform(df["Funding_Status"])
df["DPI_int"]      = df["DPI_Leverage"].astype(int)

FEATURES = [
    "Sector_enc", "Demo_enc", "Founder_Pedigree_Score",
    "DPI_int", "Revenue_ARR_USD", "CAC_USD",
    "Burn_Multiple", "IP_Moat_Score",
]

X = df[FEATURES].values
y_class  = df["Status_enc"].values
y_reg    = np.log1p(df["Expected_Pre_Money_Valuation_USD"].values)   # log-transform

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler,    "models/scaler.pkl")
joblib.dump(le_sector, "models/le_sector.pkl")
joblib.dump(le_demo,   "models/le_demo.pkl")
joblib.dump(le_status, "models/le_status.pkl")
print("  Encoders & scaler saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 4a. CLASSIFICATION – Random Forest → Funding_Status
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4a – Random Forest Classification")
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_tr, y_tr)
print(classification_report(y_te, rf.predict(X_te),
                             target_names=le_status.classes_))
joblib.dump(rf, "models/rf_classifier.pkl")
print("  ✅  rf_classifier.pkl saved")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. CLUSTERING – KMeans Founder Personas
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4b – KMeans Clustering (4 Founder Personas)")
km = KMeans(n_clusters=4, n_init=20, random_state=42)
df["Cluster"] = km.fit_predict(X_scaled)
joblib.dump(km, "models/kmeans.pkl")

PERSONA_LABELS = {
    0: "🚀 Venture-Scale Disruptor",
    1: "⚙️  Deep-Tech Builder",
    2: "🌾 Bharat / Mass-Market Pioneer",
    3: "💼 Capital-Efficient Bootstrapper",
}
cluster_summary = (
    df.groupby("Cluster")[["Founder_Pedigree_Score", "Revenue_ARR_USD",
                            "Burn_Multiple", "IP_Moat_Score"]]
    .mean()
    .round(2)
)
print(cluster_summary)
joblib.dump(PERSONA_LABELS, "models/persona_labels.pkl")
print("  ✅  kmeans.pkl + persona_labels.pkl saved")


# ─────────────────────────────────────────────────────────────────────────────
# 4c. ASSOCIATION RULE MINING – Apriori
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4c – Apriori Association Rules")

# Bin continuous features into categories for transaction encoding
df["Pedigree_Hi"]   = df["Founder_Pedigree_Score"] >= 7
df["IP_Hi"]         = df["IP_Moat_Score"] >= 7
df["ARR_Hi"]        = df["Revenue_ARR_USD"] > df["Revenue_ARR_USD"].median()
df["Burn_Lo"]       = df["Burn_Multiple"]   < df["Burn_Multiple"].median()

apriori_cols = {
    "Sector_AI":      df["Sector"] == "AI",
    "Sector_Fintech": df["Sector"] == "Fintech",
    "Sector_SaaS":    df["Sector"] == "SaaS",
    "Sector_DeepTech":df["Sector"] == "DeepTech",
    "DPI_Yes":        df["DPI_Leverage"] == True,
    "Pedigree_High":  df["Pedigree_Hi"],
    "IP_High":        df["IP_Hi"],
    "ARR_High":       df["ARR_Hi"],
    "Burn_Low":       df["Burn_Lo"],
    "Funded":         df["Funding_Status"] == "Funded",
}

basket = pd.DataFrame(apriori_cols).astype(bool)
frequent_items = apriori(basket, min_support=0.05, use_colnames=True, low_memory=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.2)
rules = rules.sort_values("lift", ascending=False)

print("\n  Top 5 Association Rules:")
print("-" * 60)
for _, row in rules.head(5).iterrows():
    ant = ", ".join(list(row["antecedents"]))
    con = ", ".join(list(row["consequents"]))
    print(f"  [{ant}]  →  [{con}]"
          f"  | support={row['support']:.3f}"
          f"  | confidence={row['confidence']:.3f}"
          f"  | lift={row['lift']:.3f}")
print("-" * 60)

joblib.dump(rules, "models/association_rules.pkl")
print("  ✅  association_rules.pkl saved")


# ─────────────────────────────────────────────────────────────────────────────
# 4d. REGRESSION – XGBoost / Linear Regression → Valuation
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4d – Regression (Valuation Forecast)")
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)
try:
    from xgboost import XGBRegressor
    reg = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, n_jobs=-1
    )
    reg.fit(X_tr_r, y_tr_r)
    model_name = "XGBoost"
except ImportError:
    print("  XGBoost not found – falling back to LinearRegression")
    reg = LinearRegression()
    reg.fit(X_tr_r, y_tr_r)
    model_name = "LinearRegression"

y_pred_r = reg.predict(X_te_r)
mape = mean_absolute_percentage_error(np.expm1(y_te_r), np.expm1(y_pred_r))
print(f"  {model_name}  MAPE = {mape:.2%}")
joblib.dump(reg, "models/valuation_regressor.pkl")
print(f"  ✅  valuation_regressor.pkl saved ({model_name})")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL MODELS TRAINED & SAVED  →  /models/")
print("=" * 60)
print("  rf_classifier.pkl")
print("  kmeans.pkl")
print("  persona_labels.pkl")
print("  association_rules.pkl")
print("  valuation_regressor.pkl")
print("  scaler.pkl  |  le_sector.pkl  |  le_demo.pkl  |  le_status.pkl")
