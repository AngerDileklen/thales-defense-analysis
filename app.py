import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INPUTS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Thales Defense Analysis", layout="wide")
st.title("🛡️ Thales vs. European Defense Industry")

# Sidebar for Date Selection
st.sidebar.header("Analysis Settings")

# Default to a longer range (2015) so the "Annual Returns" chart looks good
default_start = date(2015, 1, 1)
default_end = date.today()

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if start_date >= end_date:
    st.error("Error: Start Date must be before End Date.")
    st.stop()

# Define Tickers
peers = {
    'HO.PA': 'Thales',
    'RHM.DE': 'Rheinmetall',
    'LDO.MI': 'Leonardo',
    'BA.L': 'BAE Systems',
    'AIR.PA': 'Airbus',
    'SAF.PA': 'Safran',
    'AM.PA': 'Dassault Aviation'
}


# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(start, end, tickers_dict):
    try:
        # multi_level_index=False fixes the formatting bug
        data = yf.download(list(tickers_dict.keys()), start=start, end=end, progress=False, multi_level_index=False)[
            'Close']
        # Rename columns to friendly names
        data.rename(columns=tickers_dict, inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()


with st.spinner('Downloading financial data...'):
    df_prices = load_data(start_date, end_date, peers)

if df_prices.empty:
    st.error("No data found. Please check your internet connection or date range.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. ANALYSIS & TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Annual Returns (Line)", "📈 Distribution (IQR)", "🔗 Correlation"])

# --- TAB 1: Annual Returns Comparison (The "Line Plot" you liked) ---
with tab1:
    st.subheader("Annual Returns: Thales vs. Industry Average")

    # 1. Resample to Annual Returns
    # 'YE' is Year End. We use last price of year to calculate return.
    yearly_prices = df_prices.resample('YE').last()
    annual_returns = yearly_prices.pct_change().dropna() * 100  # Convert to %

    if annual_returns.empty:
        st.warning("Not enough data to calculate Annual Returns. Please select a date range longer than 1 year.")
    else:
        # 2. Calculate Industry Average (Mean of all columns)
        annual_returns['Industry'] = annual_returns.mean(axis=1)

        # 3. Thales Stats (for the horizontal lines)
        thales_mean = annual_returns['Thales'].mean()
        thales_median = annual_returns['Thales'].median()

        # 4. Plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # Thales Line
        ax1.plot(annual_returns.index.year, annual_returns['Thales'],
                 marker='o', linewidth=3, color='firebrick', label='Thales Annual Return')

        # Industry Line
        ax1.plot(annual_returns.index.year, annual_returns['Industry'],
                 marker='s', linewidth=2, color='navy', linestyle='--', alpha=0.6, label='Industry Average')

        # Horizontal Stats Lines
        ax1.axhline(thales_mean, color='green', linestyle=':', linewidth=2,
                    label=f'Thales Mean = {thales_mean:.1f}%')
        ax1.axhline(thales_median, color='orange', linestyle='-.', linewidth=2,
                    label=f'Thales Median = {thales_median:.1f}%')

        ax1.set_xlabel("Year")
        ax1.set_ylabel("Annual Return (%)")
        ax1.set_title("Year-over-Year Performance Comparison")
        ax1.grid(True, alpha=0.5)
        ax1.legend()

        # Force integers on X-axis (Years)
        if len(annual_returns) <= 10:
            ax1.set_xticks(annual_returns.index.year)

        st.pyplot(fig1)
        st.dataframe(annual_returns.tail())

# --- TAB 2: Distribution with IQR (The "Quartile" fix) ---
with tab2:
    st.subheader("Thales Daily Return Distribution")

    # Daily Returns
    daily_ret = df_prices['Thales'].pct_change().dropna()

    # Stats
    mean_val = daily_ret.mean()
    median_val = daily_ret.median()
    std_val = daily_ret.std()
    skew_val = daily_ret.skew()
    kurt_val = daily_ret.kurt()

    # IQR Calculations
    q75, q25 = np.percentile(daily_ret, [75, 25])
    iqr = q75 - q25

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Histogram
    ax2.hist(daily_ret, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Daily Returns')

    # Normal Fit
    x = np.linspace(daily_ret.min(), daily_ret.max(), 500)
    pdf = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)
    ax2.plot(x, pdf, color='black', linewidth=2, label='Normal Fit')

    # Lines: Mean & Median
    ax2.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label='Mean')
    ax2.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label='Median')

    # Lines: Quartiles (IQR Visual)
    ax2.axvline(q25, color='purple', linestyle=':', linewidth=2, label=f'Q1 (25%): {q25:.4f}')
    ax2.axvline(q75, color='purple', linestyle=':', linewidth=2, label=f'Q3 (75%): {q75:.4f}')

    # Text Box
    stats_text = (
        f"Mean      = {mean_val:.5f}\n"
        f"Median    = {median_val:.5f}\n"
        f"Std dev   = {std_val:.5f}\n"
        f"IQR       = {iqr:.5f}\n"
        f"Skewness  = {skew_val:.4f}\n"
        f"Kurtosis  = {kurt_val:.4f}"
    )
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax2.set_title("Distribution Analysis (with Quartiles)")
    ax2.set_xlabel("Daily Return")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

# --- TAB 3: Correlation (Kept as requested) ---
with tab3:
    st.subheader("Correlation: Thales vs. Market Benchmark")

    # Use EUAD ETF or fallback to Rheinmetall if unavailable
    bench_ticker = 'EUAD'
    with st.spinner(f"Downloading {bench_ticker}..."):
        try:
            bench_df = yf.download(bench_ticker, start=start_date, end=end_date, progress=False,
                                   multi_level_index=False)
            if bench_df.empty: raise ValueError("Empty Data")
            bench_prices = bench_df['Close']
            bench_name = "Sector ETF (EUAD)"
        except:
            bench_prices = df_prices['Rheinmetall']
            bench_name = "Rheinmetall (Proxy)"
            st.warning("Could not load EUAD ETF. Using Rheinmetall as the benchmark proxy.")

    # Align Data
    aligned = pd.DataFrame({'Thales': df_prices['Thales'], 'Benchmark': bench_prices}).dropna()
    returns = aligned.pct_change().dropna()

    if not returns.empty:
        x = returns['Benchmark']
        y = returns['Thales']

        corr = x.corr(y)
        m, b = np.polyfit(x, y, 1)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(x, y, alpha=0.6, color='darkblue', edgecolors='white', s=60)

        # Regression Line
        x_line = np.linspace(x.min(), x.max(), 100)
        ax3.plot(x_line, m * x_line + b, color='red', linewidth=2, label=f'Beta = {m:.2f}')

        ax3.set_title(f"Correlation: {corr:.3f}")
        ax3.set_xlabel(f"{bench_name} Returns")
        ax3.set_ylabel("Thales Returns")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig3)