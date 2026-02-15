"""
Collect historical financial statements for semiconductor companies using yfinance.
Saves income statements, balance sheets, and cash flow statements (annual + quarterly)
as transposed CSVs with rows = dates and columns = line items.
"""

import os
import time
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMPANIES_FILE = os.path.join(BASE_DIR, "companies.csv")

STATEMENT_TYPES = {
    "income_statement": {"annual": "income_stmt", "quarterly": "quarterly_income_stmt"},
    "balance_sheet": {"annual": "balance_sheet", "quarterly": "quarterly_balance_sheet"},
    "cash_flow": {"annual": "cashflow", "quarterly": "quarterly_cashflow"},
}

DIRECTORIES = [
    os.path.join(DATA_DIR, freq, stmt)
    for freq in ("annual", "quarterly")
    for stmt in ("income_statement", "balance_sheet", "cash_flow")
]


def create_directories():
    for d in DIRECTORIES:
        os.makedirs(d, exist_ok=True)


def fetch_and_save(ticker_obj, ticker_str, stmt_name, freq, summary_rows):
    """Fetch a single financial statement and save it as a CSV."""
    attr_name = STATEMENT_TYPES[stmt_name][freq]
    out_dir = os.path.join(DATA_DIR, freq, stmt_name)
    out_file = os.path.join(out_dir, f"{ticker_str}_{stmt_name}_{freq}.csv")

    row = {
        "ticker": ticker_str,
        "statement_type": stmt_name,
        "frequency": freq,
        "file": out_file,
        "status": "success",
        "rows": 0,
        "columns": 0,
        "date_min": "",
        "date_max": "",
        "error": "",
    }

    try:
        df = getattr(ticker_obj, attr_name)
        if df is None or df.empty:
            row["status"] = "empty"
            row["error"] = "No data returned"
            summary_rows.append(row)
            return

        # yfinance returns line items as rows and dates as columns — transpose
        df = df.T
        df.index.name = "date"
        df.to_csv(out_file)

        row["rows"] = len(df)
        row["columns"] = len(df.columns)
        row["date_min"] = str(df.index.min())
        row["date_max"] = str(df.index.max())
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)

    summary_rows.append(row)


def main():
    companies = pd.read_csv(COMPANIES_FILE)
    print(f"Loaded {len(companies)} companies from {COMPANIES_FILE}")

    create_directories()

    summary_rows = []
    total = len(companies)

    for idx, (_, company) in enumerate(companies.iterrows(), 1):
        ticker_str = company["ticker"]
        print(f"[{idx}/{total}] Fetching {ticker_str} ({company['name']})...")

        ticker_obj = yf.Ticker(ticker_str)

        for stmt_name in STATEMENT_TYPES:
            for freq in ("annual", "quarterly"):
                fetch_and_save(ticker_obj, ticker_str, stmt_name, freq, summary_rows)

        if idx < total:
            time.sleep(1)

    # Write collection summary
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(BASE_DIR, "collection_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # Print summary stats
    success = (summary_df["status"] == "success").sum()
    empty = (summary_df["status"] == "empty").sum()
    errors = (summary_df["status"] == "error").sum()
    print(f"\nDone! {success} succeeded, {empty} empty, {errors} errors.")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
