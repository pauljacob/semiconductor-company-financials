#!/usr/bin/env python3
"""
analyze_financials.py - Semiconductor Company Financial Analysis

Loads financial data for 64 semiconductor companies, computes key metrics,
and generates a summary CSV and visualizations.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'annual')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
COMPANIES_FILE = os.path.join(BASE_DIR, 'companies.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'analysis_results.csv')

CATEGORY_COLORS = {
    'top-tier': '#2563EB',
    'mid-cap': '#16A34A',
    'lesser-known': '#EA580C',
    'small-cap': '#DC2626',
    'equipment': '#7C3AED',
}
CATEGORY_ORDER = ['top-tier', 'mid-cap', 'lesser-known', 'small-cap', 'equipment']

# Currency normalization: companies that report in non-USD currencies.
# Rates are approximate annual averages for the most recent fiscal year.
#   TWD→USD: ~1/32.0 (2024 avg)    EUR→USD: ~1.08 (2024 avg)
CURRENCY_MAP = {
    'TSM': ('TWD', 1 / 32.0),
    'UMC': ('TWD', 1 / 32.0),
    'ASML': ('EUR', 1.08),
}


# ── Helper Functions ───────────────────────────────────────────────────────
def safe_divide(numerator, denominator):
    """Safely divide, returning NaN for zero/NaN denominators."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def load_csv(filepath):
    """Load a CSV file, returning None if it doesn't exist or is empty."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return df if not df.empty else None
    except Exception:
        return None


def get_value(df, row_idx, col_name):
    """Safely get a numeric value from a DataFrame."""
    if df is None or col_name not in df.columns or row_idx >= len(df):
        return np.nan
    val = df.iloc[row_idx][col_name]
    if pd.isna(val):
        return np.nan
    return float(val)


# ── Step 1: Load all data ─────────────────────────────────────────────────
def load_all_data():
    """Load companies list and all financial statements."""
    companies = pd.read_csv(COMPANIES_FILE)
    print(f"Loaded {len(companies)} companies from companies.csv")

    all_data = {}
    missing_data = []

    for _, row in companies.iterrows():
        ticker = row['ticker']

        income_file = os.path.join(DATA_DIR, 'income_statement', f'{ticker}_income_statement_annual.csv')
        balance_file = os.path.join(DATA_DIR, 'balance_sheet', f'{ticker}_balance_sheet_annual.csv')
        cashflow_file = os.path.join(DATA_DIR, 'cash_flow', f'{ticker}_cash_flow_annual.csv')

        income_df = load_csv(income_file)
        balance_df = load_csv(balance_file)
        cashflow_df = load_csv(cashflow_file)

        missing = []
        if income_df is None:
            missing.append('income_statement')
        if balance_df is None:
            missing.append('balance_sheet')
        if cashflow_df is None:
            missing.append('cash_flow')
        if missing:
            missing_data.append((ticker, missing))

        all_data[ticker] = {
            'name': row['name'],
            'category': row['category'],
            'income': income_df,
            'balance': balance_df,
            'cashflow': cashflow_df,
        }

    return companies, all_data, missing_data


# ── Step 2: Compute metrics ───────────────────────────────────────────────
def compute_metrics(all_data):
    """Compute key financial metrics for each company."""
    results = []

    for ticker, data in all_data.items():
        inc = data['income']
        bal = data['balance']
        cf = data['cashflow']

        # Currency conversion factor (1.0 for USD-reporting companies)
        fx = 1.0
        reporting_ccy = 'USD'
        if ticker in CURRENCY_MAP:
            reporting_ccy, fx = CURRENCY_MAP[ticker]

        # Most recent year (row 0) — all monetary values converted to USD
        revenue = get_value(inc, 0, 'Total Revenue') * fx if not pd.isna(get_value(inc, 0, 'Total Revenue')) else np.nan
        gross_profit = get_value(inc, 0, 'Gross Profit') * fx if not pd.isna(get_value(inc, 0, 'Gross Profit')) else np.nan
        operating_income = get_value(inc, 0, 'Operating Income') * fx if not pd.isna(get_value(inc, 0, 'Operating Income')) else np.nan
        net_income = get_value(inc, 0, 'Net Income') * fx if not pd.isna(get_value(inc, 0, 'Net Income')) else np.nan
        rd = get_value(inc, 0, 'Research And Development') * fx if not pd.isna(get_value(inc, 0, 'Research And Development')) else np.nan
        ebitda = get_value(inc, 0, 'EBITDA') * fx if not pd.isna(get_value(inc, 0, 'EBITDA')) else np.nan
        diluted_eps = get_value(inc, 0, 'Diluted EPS') * fx if not pd.isna(get_value(inc, 0, 'Diluted EPS')) else np.nan

        # Prior year revenue for growth calc (row 1)
        revenue_prior_raw = get_value(inc, 1, 'Total Revenue')
        revenue_prior = revenue_prior_raw * fx if not pd.isna(revenue_prior_raw) else np.nan

        # Balance sheet (most recent) — converted to USD
        stockholders_equity = get_value(bal, 0, 'Stockholders Equity') * fx if not pd.isna(get_value(bal, 0, 'Stockholders Equity')) else np.nan
        total_assets = get_value(bal, 0, 'Total Assets') * fx if not pd.isna(get_value(bal, 0, 'Total Assets')) else np.nan
        current_assets_raw = get_value(bal, 0, 'Current Assets')
        current_liabilities_raw = get_value(bal, 0, 'Current Liabilities')

        # Fallback: compute current assets from working capital
        if pd.isna(current_assets_raw):
            wc = get_value(bal, 0, 'Working Capital')
            if not pd.isna(wc) and not pd.isna(current_liabilities_raw):
                current_assets_raw = wc + current_liabilities_raw

        current_assets = current_assets_raw * fx if not pd.isna(current_assets_raw) else np.nan
        current_liabilities = current_liabilities_raw * fx if not pd.isna(current_liabilities_raw) else np.nan
        total_debt = get_value(bal, 0, 'Total Debt') * fx if not pd.isna(get_value(bal, 0, 'Total Debt')) else np.nan
        cash = get_value(bal, 0, 'Cash And Cash Equivalents') * fx if not pd.isna(get_value(bal, 0, 'Cash And Cash Equivalents')) else np.nan

        # Cash flow (most recent) — converted to USD
        fcf = get_value(cf, 0, 'Free Cash Flow') * fx if not pd.isna(get_value(cf, 0, 'Free Cash Flow')) else np.nan

        # Compute ratios (currency cancels out, but values are already in USD)
        revenue_growth = np.nan
        if not pd.isna(revenue) and not pd.isna(revenue_prior) and revenue_prior != 0:
            revenue_growth = (revenue - revenue_prior) / abs(revenue_prior)

        results.append({
            'ticker': ticker,
            'name': data['name'],
            'category': data['category'],
            'revenue': revenue,
            'revenue_growth_yoy': revenue_growth,
            'gross_margin': safe_divide(gross_profit, revenue),
            'operating_margin': safe_divide(operating_income, revenue),
            'net_margin': safe_divide(net_income, revenue),
            'rd_intensity': safe_divide(rd, revenue),
            'ebitda_margin': safe_divide(ebitda, revenue),
            'eps_diluted': diluted_eps,
            'free_cash_flow': fcf,
            'fcf_margin': safe_divide(fcf, revenue),
            'roe': safe_divide(net_income, stockholders_equity),
            'roa': safe_divide(net_income, total_assets),
            'current_ratio': safe_divide(current_assets, current_liabilities),
            'debt_to_equity': safe_divide(total_debt, stockholders_equity),
            'cash_position': cash,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('revenue', ascending=False).reset_index(drop=True)
    return df


# ── Step 4: Visualizations ────────────────────────────────────────────────
def create_charts(df):
    """Generate all visualization charts."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

    _chart_revenue_top20(df)
    _chart_margins_by_category(df)
    _chart_revenue_growth(df)
    _chart_rd_vs_margin(df)
    _chart_profitability_heatmap(df)
    _chart_financial_health(df)

    print(f"\nAll charts saved to {CHARTS_DIR}/")


def _chart_revenue_top20(df):
    """Bar chart of top 20 companies by revenue."""
    top20 = df.nlargest(20, 'revenue').copy()
    top20['revenue_b'] = top20['revenue'] / 1e9

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = [CATEGORY_COLORS.get(c, '#888') for c in top20['category']]
    bars = ax.bar(range(len(top20)), top20['revenue_b'], color=colors,
                  edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(top20['ticker'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Revenue ($B)')
    ax.set_title('Top 20 Semiconductor Companies by Revenue')

    for bar, val in zip(bars, top20['revenue_b']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'${val:.1f}B', ha='center', va='bottom', fontsize=8)

    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=c)
                       for c in CATEGORY_ORDER if c in top20['category'].values]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'revenue_top20.png'), bbox_inches='tight')
    plt.close()
    print("  > revenue_top20.png")


def _chart_margins_by_category(df):
    """Grouped box plots of gross/operating/net margin by category."""
    margin_cols = ['gross_margin', 'operating_margin', 'net_margin']
    margin_labels = ['Gross Margin', 'Operating Margin', 'Net Margin']

    melted = df.melt(
        id_vars=['ticker', 'category'],
        value_vars=margin_cols,
        var_name='Metric',
        value_name='Value',
    )
    melted['Metric'] = melted['Metric'].map(dict(zip(margin_cols, margin_labels)))
    melted['Value'] = melted['Value'] * 100
    melted = melted.dropna(subset=['Value'])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=melted, x='category', y='Value', hue='Metric',
        order=CATEGORY_ORDER, ax=ax, palette='Set2',
    )
    ax.set_xlabel('Category')
    ax.set_ylabel('Margin (%)')
    ax.set_title('Profit Margins by Company Category')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title='Metric', loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'margins_by_category.png'), bbox_inches='tight')
    plt.close()
    print("  > margins_by_category.png")


def _chart_revenue_growth(df):
    """Horizontal bar chart of YoY revenue growth for all companies."""
    growth_df = df.dropna(subset=['revenue_growth_yoy']).copy()
    growth_df = growth_df.sort_values('revenue_growth_yoy', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(10, len(growth_df) * 0.3)))
    colors = ['#16A34A' if g >= 0 else '#DC2626' for g in growth_df['revenue_growth_yoy']]
    ax.barh(range(len(growth_df)), growth_df['revenue_growth_yoy'] * 100,
            color=colors, height=0.7)

    ax.set_yticks(range(len(growth_df)))
    ax.set_yticklabels(growth_df['ticker'], fontsize=8)
    ax.set_xlabel('Revenue Growth YoY (%)')
    ax.set_title('Revenue Growth (Year-over-Year) — All Companies')
    ax.axvline(x=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'revenue_growth_ranking.png'), bbox_inches='tight')
    plt.close()
    print("  > revenue_growth_ranking.png")


def _chart_rd_vs_margin(df):
    """Scatter plot: R&D intensity vs operating margin, colored by category, sized by revenue."""
    plot_df = df.dropna(subset=['rd_intensity', 'operating_margin']).copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    max_rev = plot_df['revenue'].max()
    sizes = (plot_df['revenue'] / max_rev) * 500 + 20

    for cat in CATEGORY_ORDER:
        mask = plot_df['category'] == cat
        if mask.any():
            subset = plot_df[mask]
            ax.scatter(
                subset['rd_intensity'] * 100,
                subset['operating_margin'] * 100,
                s=sizes[mask],
                c=CATEGORY_COLORS[cat],
                label=cat,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5,
            )

    # Label notable companies
    for _, row in plot_df.iterrows():
        if row['revenue'] > 20e9 or abs(row['operating_margin']) > 0.4 or row['rd_intensity'] > 0.35:
            ax.annotate(
                row['ticker'],
                (row['rd_intensity'] * 100, row['operating_margin'] * 100),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords='offset points',
            )

    ax.set_xlabel('R&D Intensity (% of Revenue)')
    ax.set_ylabel('Operating Margin (%)')
    ax.set_title('R&D Intensity vs Operating Margin (bubble size = revenue)')
    ax.legend(title='Category')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'rd_vs_margin.png'), bbox_inches='tight')
    plt.close()
    print("  > rd_vs_margin.png")


def _chart_profitability_heatmap(df):
    """Heatmap of profitability metrics for top 30 companies by revenue."""
    top30 = df.nlargest(30, 'revenue').copy()

    metrics = ['gross_margin', 'operating_margin', 'net_margin', 'fcf_margin']
    labels = ['Gross Margin', 'Operating Margin', 'Net Margin', 'FCF Margin']

    heatmap_data = top30.set_index('ticker')[metrics].copy()
    heatmap_data.columns = labels
    heatmap_data = heatmap_data * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt='.1f',
        cmap='RdYlGn', center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Margin (%)'},
    )
    ax.set_title('Profitability Metrics — Top 30 by Revenue')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'profitability_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("  > profitability_heatmap.png")


def _chart_financial_health(df):
    """Scatter plot: current ratio vs debt-to-equity, colored by category."""
    plot_df = df.dropna(subset=['current_ratio', 'debt_to_equity']).copy()
    # Filter to readable range (exclude negative equity and extreme outliers)
    plot_df = plot_df[(plot_df['debt_to_equity'] >= 0) &
                      (plot_df['debt_to_equity'] <= 10) &
                      (plot_df['current_ratio'] <= 15)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for cat in CATEGORY_ORDER:
        mask = plot_df['category'] == cat
        if mask.any():
            subset = plot_df[mask]
            ax.scatter(
                subset['current_ratio'],
                subset['debt_to_equity'],
                c=CATEGORY_COLORS[cat],
                label=cat,
                s=80, alpha=0.7,
                edgecolors='white', linewidth=0.5,
            )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row['ticker'],
            (row['current_ratio'], row['debt_to_equity']),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
        )

    ax.set_xlabel('Current Ratio')
    ax.set_ylabel('Debt-to-Equity')
    ax.set_title('Financial Health: Liquidity vs Leverage')
    ax.legend(title='Category')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'financial_health.png'), bbox_inches='tight')
    plt.close()
    print("  > financial_health.png")


# ── Step 5: Print Summary ─────────────────────────────────────────────────
def print_summary(df, missing_data):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("SEMICONDUCTOR FINANCIAL ANALYSIS SUMMARY")
    print("=" * 80)

    # Top 10 by Revenue
    print("\n-- Top 10 by Revenue --")
    top_rev = df.nlargest(10, 'revenue')[['ticker', 'name', 'revenue']].copy()
    top_rev['revenue'] = top_rev['revenue'].apply(
        lambda x: f"${x/1e9:.1f}B" if not pd.isna(x) else "N/A")
    print(top_rev.to_string(index=False))

    # Top 10 by Gross Margin
    print("\n-- Top 10 by Gross Margin --")
    top_gm = df.dropna(subset=['gross_margin']).nlargest(10, 'gross_margin')[
        ['ticker', 'name', 'gross_margin']].copy()
    top_gm['gross_margin'] = top_gm['gross_margin'].apply(lambda x: f"{x*100:.1f}%")
    print(top_gm.to_string(index=False))

    # Top 10 by Revenue Growth
    print("\n-- Top 10 by Revenue Growth (YoY) --")
    top_growth = df.dropna(subset=['revenue_growth_yoy']).nlargest(10, 'revenue_growth_yoy')[
        ['ticker', 'name', 'revenue_growth_yoy']].copy()
    top_growth['revenue_growth_yoy'] = top_growth['revenue_growth_yoy'].apply(
        lambda x: f"{x*100:.1f}%")
    print(top_growth.to_string(index=False))

    # Category Averages
    print("\n-- Category Averages --")
    cat_metrics = ['revenue', 'gross_margin', 'operating_margin', 'net_margin',
                   'rd_intensity', 'revenue_growth_yoy']
    cat_avg = df.groupby('category')[cat_metrics].mean().reindex(CATEGORY_ORDER)

    header = (f"{'Category':<15} {'Avg Rev ($B)':>12} {'Gross Mgn':>10} "
              f"{'Op Mgn':>10} {'Net Mgn':>10} {'R&D Int':>10} {'Rev Growth':>10}")
    print(header)
    print("-" * 80)
    for cat in CATEGORY_ORDER:
        if cat in cat_avg.index:
            r = cat_avg.loc[cat]
            vals = []
            vals.append(f"${r['revenue']/1e9:.1f}B" if not pd.isna(r['revenue']) else "N/A")
            for m in ['gross_margin', 'operating_margin', 'net_margin',
                       'rd_intensity', 'revenue_growth_yoy']:
                vals.append(f"{r[m]*100:.1f}%" if not pd.isna(r[m]) else "N/A")
            print(f"{cat:<15} {vals[0]:>12} {vals[1]:>10} {vals[2]:>10} "
                  f"{vals[3]:>10} {vals[4]:>10} {vals[5]:>10}")

    # Currency normalization notes
    if CURRENCY_MAP:
        print("\n-- Currency Normalization (converted to USD) --")
        for ticker, (ccy, rate) in sorted(CURRENCY_MAP.items()):
            if rate < 1:
                print(f"  {ticker}: {ccy} -> USD (1 {ccy} = {rate:.4f} USD)")
            else:
                print(f"  {ticker}: {ccy} -> USD (1 {ccy} = {rate:.2f} USD)")

    # Missing data
    if missing_data:
        print(f"\n-- Companies with Missing Data ({len(missing_data)}) --")
        for ticker, missing in missing_data:
            print(f"  {ticker}: missing {', '.join(missing)}")
    else:
        print("\n  No missing data files detected.")

    # Column completeness
    total = len(df)
    for col in ['revenue', 'gross_margin', 'operating_margin', 'net_margin',
                'revenue_growth_yoy', 'fcf_margin', 'roe', 'current_ratio', 'debt_to_equity']:
        non_null = df[col].notna().sum()
        if non_null < total:
            print(f"  Note: {col} available for {non_null}/{total} companies")

    print("\n" + "=" * 80)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading financial data...")
    companies, all_data, missing_data = load_all_data()

    print("Computing financial metrics...")
    df = compute_metrics(all_data)

    # Step 3: Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved analysis results to {OUTPUT_FILE} ({len(df)} rows)")

    # Step 4: Generate charts
    print("\nGenerating visualizations...")
    create_charts(df)

    # Step 5: Print summary
    print_summary(df, missing_data)


if __name__ == '__main__':
    main()
