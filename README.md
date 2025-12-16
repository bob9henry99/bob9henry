"""
Generic Flow Index - Requirements from December 5th, 2025

Purpose: Aggregate basic statistics across multiple products showing summed 
USD-equivalent notional/aggregated maturity by date across all products.

Requirements:
- Product types: Swaps to fixed, Swaps to floating, Pre-issuance swaps, XCCY swaps, Caps, Swaptions
- Filters: (a) Bilateral/XXXX/Off-SEF, (b) new trades, (c) at least one USD leg
- Key variables: Currencies, Notional, Effective Date, Maturity Date, Tenor, Expiration Date, Date/Time
- Focus: Aggregated statistics on notional and maturity across all products
- Minimal filtering to get the big picture
"""

import re
import time
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


# -------------------------
# Configuration
# -------------------------

PLATFORM_PATTERN = r"(?:a|\b)(BILT|XXXX|XOFF)(?:a|\b)(?:a|$)"


# -------------------------
# Utility Functions
# -------------------------

def timer_func(fn):
    """Simple timing decorator."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        t1 = time.time()
        print(f"[TIMER] {fn.__name__} completed in {(t1 - t0):.2f}s")
        return out
    return wrapper


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical format.
    Maps common variations to standard names.
    """
    mapping = {
        "Event timestamp": "Event timestamp",
        "Execution Timestamp": "Execution Timestamp",
        "Effective Date": "Effective Date",
        "Expiration Date": "Expiration Date",
        "Maturity Date": "Maturity Date",
        "Action type": "Action type",
        "Platform identifier": "Platform identifier",
        "Product name": "Product name",
        "UPI FISN": "UPI FISN",
        "Notional currency-Leg 1": "Notional currency-Leg 1",
        "Notional currency-Leg 2": "Notional currency-Leg 2",
        "Notional amount-Leg 1": "Notional amount-Leg 1",
        "Notional amount-Leg 2": "Notional amount-Leg 2",
        "Notional amount in effect on associated effective date-Leg 1":
            "Notional amount in effect on associated effective date-Leg 1",
        "Notional amount in effect on associated effective date-Leg 2":
            "Notional amount in effect on associated effective date-Leg 2",
        "Notional quantity-Leg 1": "Notional quantity-Leg 1",
        "Notional quantity-Leg 2": "Notional quantity-Leg 2",
        "Total notional quantity-Leg 1": "Total notional quantity-Leg 1",
        "Total notional quantity-Leg 2": "Total notional quantity-Leg 2",
        "Fixed rate-Leg 1": "Fixed rate-Leg 1",
        "Fixed rate-Leg 2": "Fixed rate-Leg 2",
        "Spread-Leg 1": "Spread-Leg 1",
        "Spread-Leg 2": "Spread-Leg 2",
    }

    cols = []
    for c in df.columns:
        key = c.strip()
        cols.append(mapping.get(key, key))
    df.columns = cols
    return df


def to_dt(s: pd.Series) -> pd.Series:
    """Robust datetime coercion."""
    return pd.to_datetime(s, errors='coerce')


def strip_tz(s: pd.Series) -> pd.Series:
    """Remove timezone info from datetime series."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_localize(None)
        return s
    except Exception:
        return s


def clean_numeric_col(df: pd.DataFrame, col: str):
    """
    Clean numeric column in-place.
    Handles: commas, percent strings (3.25% -> 0.0325), type coercion.
    """
    if col not in df.columns:
        return
    
    s = df[col].astype(str)
    s = s.str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan}, regex=False)

    # Percent handling
    is_pct = s.str.contains("%", na=False)
    s_pct = s.where(is_pct, None)
    if s_pct is not None:
        s_pct = s_pct.str.replace("%", "", regex=False)
        s_pct = pd.to_numeric(s_pct, errors="coerce") / 100.0

    # Non-percent numeric
    s_num = s.where(~is_pct, None)
    if s_num is not None:
        s_num = s_num.str.replace(",", "", regex=False)
    s_num = pd.to_numeric(s_num, errors="coerce")

    # Combine
    cleaned = s_num.where(~is_pct, s_pct)
    df[col] = cleaned


def has_usd_leg(row: pd.Series) -> bool:
    """
    Requirement (c): At least one leg in USD.
    Returns True if either leg currency contains "USD".
    """
    c1 = str(row.get("Notional currency-Leg 1", "")).upper()
    c2 = str(row.get("Notional currency-Leg 2", "")).upper()
    return ("USD" in c1) or ("USD" in c2)


def get_usd_notional_with_source(row: pd.Series):
    """
    CRITICAL for XCCY: Extract USD notional from the USD leg only.
    
    For cross-currency swaps, this ensures we use the USD side.
    For same-currency swaps, picks the first available notional.
    
    Priority per leg (if that leg is USD):
      1) Notional amount in effect on associated effective date-Leg X
      2) Notional amount-Leg X
      3) Total notional quantity-Leg X
    
    Returns: (notional_value, source_column_name)
    """
    def pick_numeric(*vals):
        for v in vals:
            try:
                fv = float(v)
                if np.isfinite(fv) and fv > 0:
                    return fv
            except:
                continue
        return None

    c1 = str(row.get("Notional currency-Leg 1", "")).upper()
    c2 = str(row.get("Notional currency-Leg 2", "")).upper()

    cand1 = [
        row.get("Notional amount in effect on associated effective date-Leg 1"),
        row.get("Notional amount-Leg 1"),
        row.get("Total notional quantity-Leg 1"),
    ]
    cand2 = [
        row.get("Notional amount in effect on associated effective date-Leg 2"),
        row.get("Notional amount-Leg 2"),
        row.get("Total notional quantity-Leg 2"),
    ]

    # CRITICAL: Only use notional from USD leg
    v1 = pick_numeric(*cand1) if "USD" in c1 else None
    v2 = pick_numeric(*cand2) if "USD" in c2 else None

    if v1 is not None:
        for name in [
            "Notional amount in effect on associated effective date-Leg 1",
            "Notional amount-Leg 1",
            "Total notional quantity-Leg 1",
        ]:
            try:
                if row.get(name) is not None and float(row.get(name)) == v1:
                    return v1, name
            except:
                continue
        return v1, "Leg 1 (USD)"

    if v2 is not None:
        for name in [
            "Notional amount in effect on associated effective date-Leg 2",
            "Notional amount-Leg 2",
            "Total notional quantity-Leg 2",
        ]:
            try:
                if row.get(name) is not None and float(row.get(name)) == v2:
                    return v2, name
            except:
                continue
        return v2, "Leg 2 (USD)"

    return None, None


def classify_product(row: pd.Series) -> str:
    """
    Classify products per requirements:
    - Swaps to fixed (IRS with fixed rate)
    - Swaps to floating (IRS without fixed rate)
    - Pre-issuance swaps (forward starting)
    - XCCY swaps (cross-currency)
    - Caps
    - Swaptions (if simple)
    
    Note from email: "Pre-issuance swaps, XCCY swaps, caps, swaptions 
    I'd think would be easier to ID"
    """
    name = str(row.get("Product name", "")).upper()
    upi = str(row.get("UPI FISN", "")).upper()
    text = f"{name} {upi}"
    
    # Get currencies to detect cross-currency
    c1 = str(row.get("Notional currency-Leg 1", "")).upper()
    c2 = str(row.get("Notional currency-Leg 2", "")).upper()
    is_xccy = (c1 and c2 and c1 != c2)
    
    # Check for fixed rate
    has_fixed = False
    fr1 = row.get("Fixed rate-Leg 1")
    fr2 = row.get("Fixed rate-Leg 2")
    try:
        if (pd.notna(fr1) and float(fr1) != 0) or (pd.notna(fr2) and float(fr2) != 0):
            has_fixed = True
    except:
        pass
    
    # Check if forward starting (pre-issuance)
    is_forward_start = False
    eff_date = row.get("Effective Date")
    exec_date = row.get("Execution Timestamp")
    if pd.notna(eff_date) and pd.notna(exec_date):
        try:
            eff = pd.to_datetime(eff_date)
            exc = pd.to_datetime(exec_date)
            if (eff - exc).days > 2:
                is_forward_start = True
        except:
            pass
    
    # Classification hierarchy (per email priorities)
    
    # 1. XCCY (easiest to ID per email)
    if is_xccy or re.search(r'\bXCCY\b|\bCCS\b|\bCROSS.?CURRENCY\b', text):
        return "XCCY"
    
    # 2. Caps (easiest to ID per email)
    if re.search(r'\bCAP\b', text):
        return "CAP"
    
    # 3. Swaptions (easiest to ID per email)
    if re.search(r'\bSWAPTION\b|\bOPTION\b', text):
        return "SWAPTION"
    
    # 4. Interest Rate Swaps
    if re.search(r'\bIRS\b|\bSWAP\b|\bINTEREST.?RATE', text):
        # Pre-issuance (forward starting)
        if is_forward_start:
            return "PRE-ISSUANCE"
        
        # Fixed vs Floating
        if has_fixed:
            return "SWAP-FIXED"
        else:
            return "SWAP-FLOATING"
    
    # 5. Other products
    if re.search(r'\bFRA\b|\bFORWARD.?RATE', text):
        return "FRA"
    
    if re.search(r'\bBOND\b', text):
        return "BOND"
    
    if re.search(r'\bFUTURE\b', text):
        return "FUTURE"
    
    return "OTHER"


def get_fixed_rate_with_source(row: pd.Series):
    """
    Extract fixed rate if available.
    Returns: (fixed_rate_value, source_column_name)
    """
    fr1 = row.get("Fixed rate-Leg 1")
    fr2 = row.get("Fixed rate-Leg 2")

    def to_float(x):
        try:
            v = float(x)
            if np.isfinite(v):
                return v
        except:
            pass
        return None

    v1 = to_float(fr1)
    v2 = to_float(fr2)

    if v1 is not None and v1 != 0:
        return v1, "Fixed rate-Leg 1"
    if v2 is not None and v2 != 0:
        return v2, "Fixed rate-Leg 2"
    return None, None


# -------------------------
# Core Pipeline Functions
# -------------------------

@timer_func
def load_and_filter_csv(csv_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load flows.csv and apply minimal filters per requirements:
    (a) Bilateral/XXXX/Off-SEF trades
    (b) New trades
    (c) At least one leg in USD
    
    "The least amount of filtering is preferable to get the big picture"
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    print(f"Loading data from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(p, dtype=str, low_memory=False)
    print(f"Initial rows: {len(df):,}")

    # Normalize columns
    df = normalize_columns(df)

    # Parse date columns
    date_cols = ["Event timestamp", "Execution Timestamp", "Effective Date", 
                 "Expiration Date", "Maturity Date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = to_dt(df[col])
            df[col] = strip_tz(df[col])

    # Filter by date range
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    
    if "Event timestamp" not in df.columns:
        raise KeyError("Column 'Event timestamp' missing")

    mask_date = (df["Event timestamp"] >= start_dt) & (df["Event timestamp"] <= end_dt)
    df = df.loc[mask_date].copy()
    print(f"After date filter ({start_date} to {end_date}): {len(df):,}")

    # Filter (a): Bilateral/XXXX/Off-SEF
    df["Platform identifier"] = df.get("Platform identifier", "").astype(str)
    df = df[df["Platform identifier"].str.contains(PLATFORM_PATTERN, flags=re.IGNORECASE, na=False)]
    print(f"After platform filter (BILT/XXXX/XOFF): {len(df):,}")

    # Filter (b): New trades
    df["Action type"] = df.get("Action type", "").astype(str)
    df = df[df["Action type"].str.contains("NEWT", case=False, na=False)]
    print(f"After action type filter (NEWT): {len(df):,}")

    # Filter (c): At least one USD leg
    df["Notional currency-Leg 1"] = df.get("Notional currency-Leg 1", "").astype(str)
    df["Notional currency-Leg 2"] = df.get("Notional currency-Leg 2", "").astype(str)
    df = df[df.apply(has_usd_leg, axis=1)]
    print(f"After USD leg filter: {len(df):,}")

    return df


def enrich_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add key variables per requirements:
    - Currencies of the leg(s)
    - Notional (USD-equivalent for XCCY)
    - Effective Date
    - Maturity Date
    - Tenor
    - Expiration Date (if applicable)
    - Date/Time
    - Product Type
    - Fixed Rate
    """
    print("\nEnriching data...")
    
    # Clean numeric columns
    numeric_cols = [
        "Notional amount-Leg 1", "Notional amount-Leg 2",
        "Notional amount in effect on associated effective date-Leg 1",
        "Notional amount in effect on associated effective date-Leg 2",
        "Notional quantity-Leg 1", "Notional quantity-Leg 2",
        "Total notional quantity-Leg 1", "Total notional quantity-Leg 2",
        "Fixed rate-Leg 1", "Fixed rate-Leg 2",
        "Spread-Leg 1", "Spread-Leg 2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            clean_numeric_col(df, col)

    # USD Notional (uses USD leg for XCCY)
    usd_vals = df.apply(lambda r: get_usd_notional_with_source(r), axis=1)
    df["USD Notional"] = [v for v, src in usd_vals]
    df["USD Notional Source"] = [src if src else "" for v, src in usd_vals]

    # Fixed Rate
    fr_vals = df.apply(lambda r: get_fixed_rate_with_source(r), axis=1)
    df["Fixed Rate"] = [v for v, src in fr_vals]
    df["Fixed Rate Source"] = [src if src else "" for v, src in fr_vals]

    # Tenor (years): Use Expiration Date, fallback to Maturity Date
    eff = pd.to_datetime(df.get("Effective Date"), errors="coerce")
    exp = pd.to_datetime(df.get("Expiration Date"), errors="coerce")
    mat = pd.to_datetime(df.get("Maturity Date"), errors="coerce")
    end = exp.fillna(mat)
    df["Tenor (yrs)"] = ((end - eff).dt.days / 365).round(4)

    # Product Type
    df["Product Type"] = df.apply(classify_product, axis=1)

    # Exec Date (for aggregation)
    exec_ts = pd.to_datetime(df.get("Execution Timestamp"), errors="coerce")
    df["Exec Date"] = exec_ts.dt.date

    # Select columns to keep
    keep_cols = [
        # Date/Time
        "Event timestamp",
        "Execution Timestamp",
        "Exec Date",
        "Effective Date",
        "Maturity Date",
        "Expiration Date",
        "Tenor (yrs)",
        
        # Currencies
        "Notional currency-Leg 1",
        "Notional currency-Leg 2",
        
        # Notional
        "USD Notional",
        "USD Notional Source",
        
        # Product
        "Product Type",
        "Product name",
        "UPI FISN",
        
        # Rates
        "Fixed Rate",
        "Fixed Rate Source",
        "Fixed rate-Leg 1",
        "Fixed rate-Leg 2",
        
        # Additional fields
        "Action type",
        "Platform identifier",
    ]
    
    existing = [c for c in keep_cols if c in df.columns]
    result = df[existing].copy()
    
    print(f"Enriched rows: {len(result):,}")
    return result


def aggregate_daily_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily aggregation across all products.
    Focus: notional and maturity (tenor) per requirements.
    
    Outputs:
    - USD Notional Sum
    - Trade Count
    - Tenor Weighted Average (notional-weighted)
    - Fixed Rate Weighted Average (notional-weighted)
    """
    weights = df["USD Notional"].fillna(0)
    tenors = df["Tenor (yrs)"].fillna(0)
    fixed_rates = df["Fixed Rate"].fillna(0)

    daily_sum = weights.groupby(df["Exec Date"]).sum().rename("USD_Notional_Sum")
    daily_cnt = df.groupby("Exec Date")["USD Notional"].size().rename("Trade_Count")
    
    # Weighted average tenor
    wt_tenor_sum = (weights * tenors).groupby(df["Exec Date"]).sum()
    tenor_wavg = (wt_tenor_sum / daily_sum.replace(0, np.nan)).rename("Tenor_WAvg_yrs")

    # Weighted average fixed rate
    wt_fr_sum = (weights * fixed_rates).groupby(df["Exec Date"]).sum()
    fr_wavg = (wt_fr_sum / daily_sum.replace(0, np.nan)).rename("FixedRate_WAvg")

    out = pd.concat([daily_sum, daily_cnt, tenor_wavg, fr_wavg], axis=1)
    out = out.reset_index().sort_values("Exec Date")
    return out


def aggregate_daily_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily aggregation by product type.
    Focus: notional and maturity (tenor) per product.
    
    Outputs:
    - USD Notional Sum per product
    - Trade Count per product
    - Tenor Weighted Average per product
    - Fixed Rate Weighted Average per product
    """
    keys = ["Exec Date", "Product Type"]

    weights = df["USD Notional"].fillna(0)
    tenors = df["Tenor (yrs)"].fillna(0)
    fixed_rates = df["Fixed Rate"].fillna(0)

    daily_sum = weights.groupby(df[keys].apply(tuple, axis=1)).sum()
    daily_cnt = df.groupby(keys)["USD Notional"].size()

    # Recover MultiIndex
    daily_sum.index = pd.MultiIndex.from_tuples(daily_sum.index, names=keys)
    daily_cnt.index = pd.MultiIndex.from_tuples(daily_cnt.index, names=keys)

    # Weighted average tenor
    wt_tenor_sum = (weights * tenors).groupby(df[keys].apply(tuple, axis=1)).sum()
    wt_tenor_sum.index = pd.MultiIndex.from_tuples(wt_tenor_sum.index, names=keys)
    tenor_wavg = (wt_tenor_sum / daily_sum.replace(0, np.nan)).rename("Tenor_WAvg_yrs")

    # Weighted average fixed rate
    wt_fr_sum = (weights * fixed_rates).groupby(df[keys].apply(tuple, axis=1)).sum()
    wt_fr_sum.index = pd.MultiIndex.from_tuples(wt_fr_sum.index, names=keys)
    fr_wavg = (wt_fr_sum / daily_sum.replace(0, np.nan)).rename("FixedRate_WAvg")

    out = pd.concat([
        daily_sum.rename("USD_Notional_Sum"),
        daily_cnt.rename("Trade_Count"),
        tenor_wavg,
        fr_wavg
    ], axis=1)
    
    out = out.reset_index().sort_values(keys)
    return out


def write_excel(out_path: str, trades_df: pd.DataFrame,
                daily_all: pd.DataFrame, daily_by_prod: pd.DataFrame):
    """
    Write results to Excel with three sheets:
    1. Trades (Filtered) - Individual trade details
    2. Index - Daily (All) - Aggregated across all products
    3. Index - Daily by Product - Aggregated by product type
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting Excel file: {out_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if out.exists():
            with pd.ExcelWriter(out, engine="openpyxl", mode="a", 
                              if_sheet_exists="replace") as writer:
                trades_df.to_excel(writer, sheet_name="Trades (Filtered)", index=False)
                daily_all.to_excel(writer, sheet_name="Index - Daily (All)", index=False)
                daily_by_prod.to_excel(writer, sheet_name="Index - Daily by Product", index=False)
        else:
            with pd.ExcelWriter(out, engine="openpyxl", mode="w") as writer:
                trades_df.to_excel(writer, sheet_name="Trades (Filtered)", index=False)
                daily_all.to_excel(writer, sheet_name="Index - Daily (All)", index=False)
                daily_by_prod.to_excel(writer, sheet_name="Index - Daily by Product", index=False)

    print(f"✓ Excel file created successfully")


def build_generic_flow_index(csv_path: str, excel_out_path: str,
                            start_date: str, end_date: str):
    """
    Main pipeline per requirements from December 5th, 2025.
    
    Creates generic flow index showing:
    - Summed USD-equivalent notional by date
    - Aggregated maturity (tenor) by date
    - Broken down by product type
    
    Product types: Swaps to fixed, Swaps to floating, Pre-issuance,
                   XCCY, Caps, Swaptions
    """
    print("=" * 70)
    print("GENERIC FLOW INDEX - Building per Requirements")
    print("=" * 70)
    
    # Load and filter
    df = load_and_filter_csv(csv_path, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data found in date range {start_date} to {end_date}")

    # Enrich and classify
    trades = enrich_and_classify(df)

    # Diagnostics
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECKS")
    print("=" * 70)
    
    missing = trades["USD Notional"].isna().sum()
    total = len(trades)
    pct = (missing / total * 100.0) if total else 0.0
    print(f"USD Notional missing: {missing:,} / {total:,} ({pct:.1f}%)")

    if "USD Notional Source" in trades.columns:
        print("\nTop USD Notional sources:")
        print(trades["USD Notional Source"].value_counts(dropna=False).head(5))

    if "Product Type" in trades.columns:
        print("\nProduct Type distribution:")
        print(trades["Product Type"].value_counts(dropna=False))

    if "Fixed Rate Source" in trades.columns:
        print("\nTop Fixed Rate sources:")
        print(trades["Fixed Rate Source"].value_counts(dropna=False).head(5))

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATING DATA")
    print("=" * 70)
    
    daily_all = aggregate_daily_all(trades)
    print(f"Daily aggregation (all products): {len(daily_all)} dates")
    
    daily_by_prod = aggregate_daily_by_product(trades)
    print(f"Daily aggregation (by product): {len(daily_by_prod)} rows")

    # Write Excel
    write_excel(excel_out_path, trades, daily_all, daily_by_prod)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    evt = pd.to_datetime(trades["Event timestamp"], errors="coerce")
    rng_min = evt.min().date() if evt.notna().any() else "N/A"
    rng_max = evt.max().date() if evt.notna().any() else "N/A"
    
    total_notional = trades["USD Notional"].sum()
    print(f"Total trades: {len(trades):,}")
    print(f"Date range: {rng_min} to {rng_max}")
    print(f"Total USD notional: ${total_notional:,.0f}")
    print(f"Excel output: {excel_out_path}")
    print("=" * 70)


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    # Configuration
    FLOW_CSV_PATH = r"X:\SDR\flows.csv"
    EXCEL_OUT_PATH = r"X:\SDR\Data\Source\Generic Flow Index.xlsx"
    
    # Date range
    START_DATE = "2025-11-20"
    END_DATE = date.today().strftime("%Y-%m-%d")

    # Build index
    build_generic_flow_index(
        csv_path=FLOW_CSV_PATH,
        excel_out_path=EXCEL_OUT_PATH,
        start_date=START_DATE,
        end_date=END_DATE
    )****
