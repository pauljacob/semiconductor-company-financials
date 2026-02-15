# Financial Statement Analysis Workflow

A structured workflow for analyzing publicly traded company financial statements leverages Python tools like pandas and edgartools for automation, given expertise in data manipulation and SEC EDGAR access.

## Core Workflow Steps

Follow this repeatable pipeline: retrieve data from EDGAR, clean and standardize, compute ratios/trends, visualize insights, and validate outputs.

- **Data Retrieval**: Use Python libraries like edgartools (`pip install edgartools`) or sec-api to fetch 10-K/10-Q filings by CIK or ticker. Example: `from edgar import *; set_identity("your.email@example.com"); company = Company("AAPL"); filings = company.get_filings().filter(form="10-K")` extracts balance sheets and income statements in one line.
- **Cleaning & Parsing**: Load XBRL data into pandas DataFrames; handle inconsistencies like varying tags or dates with normalization (e.g., `pd.to_datetime()`). Preprocess footnotes and segments for completeness.
- **Analysis Layer**: Calculate key ratios (e.g., ROE, current ratio, YoY growth) using vectorized pandas operations. Build DCF models by projecting free cash flow from extracted metrics.
- **Visualization & Insights**: Plot trends with matplotlib/seaborn (e.g., revenue growth over quarters); compare peers via industry benchmarks.
- **Validation & Reporting**: Cross-check against sources like Yahoo Finance; automate reports with Jupyter notebooks or dashboards.

## Recommended Tools Comparison

| Tool/Library      | Strengths                                      | Best For                       | Limitations                    |
|-------------------|------------------------------------------------|--------------------------------|-------------------------------|
| edgartools        | XBRL parsing, financial statements in DataFrames, insider data | Python-heavy workflows, quick starts | Requires SEC user-agent setup |
| sec-api           | Full-text search, real-time filings            | Bulk queries, 8-K events       | API key needed for heavy use  |
| Manual EDGAR      | Free, no code                                  | One-off checks                 | Time-intensive scraping       |
| yfinance/pandas_datareader | Quick ratios, no filings                 | Screening, not deep XBRL       | Lacks footnotes/MD&A          |

## Python Starter Code

```python
from edgar import *; import pandas as pd
set_identity("your.email@example.com")
company = Company("MSFT")
financials = company.get_financials()
income = financials.income_statement()
print(income.head())
```
