"""
generate_data.py
----------------
Generates a synthetic dataset of 5,000 Indian startups with realistic
distributions, dirty data injections (missing values, typos, string artifacts,
outliers, and duplicate rows), and saves to raw_startup_data.csv.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import string

fake = Faker("en_IN")
rng = np.random.default_rng(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
N = 5000
DIRTY_MISSING_RATE = 0.15   # 15% missing in ARR
DUPLICATE_RATE     = 0.03   # 3%  duplicate rows

SECTORS_CLEAN = [
    "Fintech", "AI", "DeepTech", "SaaS", "Edtech",
    "Healthtech", "Agritech", "EV/CleanTech", "D2C/Consumer", "Logistics",
]
SECTOR_TYPO_MAP = {          # dirty variants injected later
    "AI": ["A.I.", "Artificial Intelligence", "ai", "A I"],
    "Fintech": ["FinTech", "fin-tech", "FINTECH"],
    "Edtech": ["EdTech", "ed-tech", "EDTECH"],
    "DeepTech": ["Deep Tech", "deep-tech", "Deeptech"],
}

DEMOGRAPHICS  = ["India1", "India2", "Bharat", "Global"]
FUNDING_STATI = ["Funded", "Bootstrapped", "Dead"]


# ── Helper functions ─────────────────────────────────────────────────────────
def _indian_startup_name() -> str:
    """Generate a plausible Indian startup name."""
    prefixes = ["Bharat", "Digi", "Zeta", "Nova", "Kira", "Veda", "Agni",
                "Indus", "Karma", "Shakti", "Niti", "Arya", "Aura", "Krish"]
    suffixes = ["AI", "Tech", "Labs", "Works", "Hub", "Pay", "Mart",
                "Stack", "IQ", "Base", "Flow", "Net", "X", "Go"]
    return f"{random.choice(prefixes)}{random.choice(suffixes)}"


def _arr(sector: str, status: str) -> float:
    """Revenue ARR in USD – sector-aware log-normal draw."""
    base = {
        "Fintech": 800_000, "AI": 600_000, "SaaS": 700_000,
        "DeepTech": 400_000, "Edtech": 300_000, "Healthtech": 350_000,
        "Agritech": 200_000, "EV/CleanTech": 250_000,
        "D2C/Consumer": 500_000, "Logistics": 450_000,
    }.get(sector, 400_000)
    multiplier = {"Funded": 3.0, "Bootstrapped": 1.0, "Dead": 0.3}[status]
    return max(0, rng.lognormal(np.log(base * multiplier), 1.2))


def _burn_multiple(status: str) -> float:
    """Burn Multiple: ratio of net burn to net new ARR (healthy < 2)."""
    if status == "Dead":
        return rng.uniform(3.5, 6.0)
    if status == "Funded":
        return rng.uniform(0.5, 3.5)
    return rng.uniform(0.8, 2.5)


def _total_funding(status: str, arr: float) -> float:
    if status == "Bootstrapped":
        return 0.0
    if status == "Dead":
        return rng.uniform(50_000, 2_000_000)
    return arr * rng.uniform(2, 20)


def _valuation(arr: float, pedigree: int, ip: int,
               total_funding: float, status: str) -> float:
    """Pre-money valuation via a simple multiples heuristic."""
    revenue_multiple = rng.uniform(5, 50)
    base = arr * revenue_multiple
    base *= (1 + 0.05 * pedigree) * (1 + 0.03 * ip)
    if status == "Funded":
        base *= rng.uniform(1.5, 4.0)
    elif status == "Dead":
        base *= rng.uniform(0.1, 0.5)
    return max(base, total_funding * 1.1)


# ── Core generation ──────────────────────────────────────────────────────────
def generate_clean_records(n: int) -> pd.DataFrame:
    sectors     = rng.choice(SECTORS_CLEAN, n, p=[.15,.12,.10,.14,.08,
                                                   .08,.07,.07,.11,.08])
    demographics = rng.choice(DEMOGRAPHICS, n, p=[.25,.30,.25,.20])
    statuses     = rng.choice(FUNDING_STATI, n, p=[.45,.40,.15])
    pedigrees    = rng.integers(1, 11, n)
    dpi_leverage = rng.choice([True, False], n, p=[.35, .65])
    ip_scores    = rng.integers(1, 11, n)
    cac          = np.abs(rng.lognormal(np.log(150), 1.0, n))

    rows = []
    for i in range(n):
        s, st = sectors[i], statuses[i]
        arr   = _arr(s, st)
        bm    = _burn_multiple(st)
        tf    = _total_funding(st, arr)
        val   = _valuation(arr, pedigrees[i], ip_scores[i], tf, st)
        rows.append({
            "Startup_ID":                      f"IN-{10001 + i}",
            "Startup_Name":                    _indian_startup_name(),
            "Sector":                          s,
            "Target_Demographic":              demographics[i],
            "Founder_Pedigree_Score":          int(pedigrees[i]),
            "DPI_Leverage":                    bool(dpi_leverage[i]),
            "Revenue_ARR_USD":                 round(arr, 2),
            "CAC_USD":                         round(cac[i], 2),
            "Burn_Multiple":                   round(bm, 3),
            "IP_Moat_Score":                   int(ip_scores[i]),
            "Funding_Status":                  st,
            "Total_Funding_USD":               round(tf, 2),
            "Expected_Pre_Money_Valuation_USD": round(val, 2),
        })
    return pd.DataFrame(rows)


# ── Dirty-data injection ─────────────────────────────────────────────────────
def inject_dirty_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n  = len(df)

    # 1. 15% missing in Revenue_ARR_USD
    missing_idx = rng.choice(n, int(n * DIRTY_MISSING_RATE), replace=False)
    df.loc[missing_idx, "Revenue_ARR_USD"] = np.nan

    # 2. Sector typos (~12% of rows get a dirty variant)
    typo_idx = rng.choice(n, int(n * 0.12), replace=False)
    for idx in typo_idx:
        original = df.at[idx, "Sector"]
        if original in SECTOR_TYPO_MAP:
            df.at[idx, "Sector"] = random.choice(SECTOR_TYPO_MAP[original])

    # 3. String artifacts in Total_Funding_USD (~8% of funded rows)
    funded_mask = df["Funding_Status"] == "Funded"
    funded_idx  = df[funded_mask].index.tolist()
    artifact_n  = int(len(funded_idx) * 0.08)
    artifact_idx = random.sample(funded_idx, artifact_n)
    for idx in artifact_idx:
        val = df.at[idx, "Total_Funding_USD"]
        if val >= 1_000_000:
            df.at[idx, "Total_Funding_USD"] = f"{val/1_000_000:.1f}M"
        elif val >= 1_000:
            df.at[idx, "Total_Funding_USD"] = f"{val/1_000:.0f}K"

    # 4. Unrealistic Burn_Multiple outliers (>500) in ~2% of rows
    outlier_idx = rng.choice(n, int(n * 0.02), replace=False)
    df.loc[outlier_idx, "Burn_Multiple"] = rng.uniform(501, 5000,
                                                        len(outlier_idx)).round(1)

    # 5. 3% duplicate rows
    dup_n  = int(n * DUPLICATE_RATE)
    dup_src = df.sample(n=dup_n, random_state=42)
    df = pd.concat([df, dup_src], ignore_index=True)

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⏳  Generating 5,000 synthetic Indian startup records …")
    clean_df = generate_clean_records(N)

    print("💉  Injecting dirty data (missing values, typos, artifacts, outliers, dupes) …")
    dirty_df = inject_dirty_data(clean_df)

    out_path = "raw_startup_data.csv"
    dirty_df.to_csv(out_path, index=False)

    print(f"✅  Saved {len(dirty_df):,} rows → {out_path}")
    print(f"    Missing ARR  : {dirty_df['Revenue_ARR_USD'].isna().sum():,} rows")
    print(f"    Burn > 500   : {(pd.to_numeric(dirty_df['Burn_Multiple'], errors='coerce') > 500).sum():,} rows")
    print(f"    String in Funding: {dirty_df['Total_Funding_USD'].apply(lambda x: isinstance(x, str)).sum():,} rows")
    print(f"    Total rows   : {len(dirty_df):,}  (incl. ~{int(N*DUPLICATE_RATE)} dupes)")
