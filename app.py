import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INPUTS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Thales Market Analysis", layout="wide")
st.title("🛡️ Thales vs. Market & Industry")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Analysis Settings")

# Date Selection
default_start = date(2020, 1, 1)
default_end = date.today()

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if start_date >= end_date:
    st.error("Error: Start Date must be before End Date.")
    st.stop()

# Define Tickers (Including Euronext 100)
all_assets = {
    'HO.PA': 'Thales',
    '^N100': 'Euronext 100 (Market)',
    'RHM.DE': 'Rheinmetall',
    'LDO.MI': 'Leonardo',
    'BA.L': 'BAE Systems',
    'AIR.PA': 'Airbus',
    'SAF.PA': 'Safran',
    'AM.PA': 'Dassault Aviation'
}

# Asset Selection Widget
st.sidebar.subheader("Choose Assets to Compare")
selected_assets = st.sidebar.multiselect(
    "Select companies/indices for charts:",
    options=list(all_assets.values()),
    default=['Thales', 'Euronext 100 (Market)', 'Rheinmetall']
)


# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(start, end, asset_dict):
    try:
        tickers_list = list(asset_dict.keys())
        data = yf.download(tickers_list, start=start, end=end, progress=False, multi_level_index=False)['Close']
        data.rename(columns=asset_dict, inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()


with st.spinner('Downloading financial data...'):
    df_all = load_data(start_date, end_date, all_assets)

if df_all.empty:
    st.error("No data found. Please check your dates or internet connection.")
    st.stop()

# Filter data
required_cols = list(set(selected_assets + ['Thales']))
df_selected = df_all[required_cols].dropna(how='all')

# -----------------------------------------------------------------------------
# 3. ANALYSIS TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Comparison & Trends", "📈 Thales Distribution", "🔗 Correlation"])

# --- TAB 1: MARKET COMPARISON ---
with tab1:
    st.subheader("Market & Industry Comparison")

    if not selected_assets:
        st.warning("Please select at least one asset in the sidebar.")
    else:
        df_comp = df_selected[selected_assets].dropna()

        if df_comp.empty:
            st.warning("Not enough overlapping data for the selected assets.")
        else:
            # 1. ANNUAL RETURNS (Bar/Line Chart)
            st.write("### 1. Annual Returns (Year-over-Year)")

            yearly_prices = df_comp.resample('YE').last()
            annual_ret = yearly_prices.pct_change().dropna() * 100

            if not annual_ret.empty:
                fig1, ax1 = plt.subplots(figsize=(12, 5))

                # Plot lines for all selected assets
                markers = ['o', 's', '^', 'D', 'v', '<', '>']
                for i, col in enumerate(annual_ret.columns):
                    style = '--' if col == 'Euronext 100 (Market)' else '-'
                    width = 3 if col == 'Thales' else 2
                    color = 'firebrick' if col == 'Thales' else None

                    ax1.plot(annual_ret.index.year, annual_ret[col],
                             marker=markers[i % len(markers)],
                             linestyle=style, linewidth=width,
                             color=color, label=col)

                # --- ADDED: Mean/Median Lines for Thales ---
                if 'Thales' in annual_ret.columns:
                    t_mean = annual_ret['Thales'].mean()
                    t_median = annual_ret['Thales'].median()

                    ax1.axhline(t_mean, color='green', linestyle=':', linewidth=2, label=f'Thales Mean ({t_mean:.1f}%)')
                    ax1.axhline(t_median, color='orange', linestyle='-.', linewidth=2,
                                label=f'Thales Median ({t_median:.1f}%)')

                ax1.set_ylabel("Annual Return (%)")
                ax1.set_title("Annual Performance Comparison")
                ax1.axhline(0, color='black', linewidth=1)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                if len(annual_ret) <= 12: ax1.set_xticks(annual_ret.index.year)
                st.pyplot(fig1)

            # 2. VALUE CHANGE
            st.write("### 2. Value Change Evolution (Rebased to 0%)")
            st.info("Shows percentage growth from the start date.")

            normalized_prices = (df_comp / df_comp.iloc[0]) - 1
            normalized_prices = normalized_prices * 100

            fig2, ax2 = plt.subplots(figsize=(12, 6))

            for col in normalized_prices.columns:
                if col == 'Thales':
                    color = 'firebrick';
                    width = 3;
                    alpha = 1.0;
                    zorder = 10
                elif col == 'Euronext 100 (Market)':
                    color = 'black';
                    width = 2.5;
                    alpha = 0.8;
                    zorder = 9
                else:
                    color = None;
                    width = 1.5;
                    alpha = 0.6;
                    zorder = 5

                ax2.plot(normalized_prices.index, normalized_prices[col],
                         linewidth=width, color=color, alpha=alpha, zorder=zorder, label=col)

            ax2.set_ylabel("Cumulative Return (%)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend()
            st.pyplot(fig2)

# --- TAB 2: THALES DISTRIBUTION ---
with tab2:
    st.subheader("Thales (HO.PA) Return Distribution")

    if 'Thales' not in df_selected.columns:
        st.error("Thales data missing.")
    else:
        daily_ret = df_selected['Thales'].pct_change().dropna()

        mean_val = daily_ret.mean()
        median_val = daily_ret.median()
        std_val = daily_ret.std()
        q75, q25 = np.percentile(daily_ret, [75, 25])
        iqr = q75 - q25

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.hist(daily_ret, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Daily Returns')

        x = np.linspace(daily_ret.min(), daily_ret.max(), 500)
        pdf = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)
        ax3.plot(x, pdf, color='black', linewidth=2, label='Normal Fit')

        ax3.axvline(mean_val, color='red', linestyle='--', label='Mean')
        ax3.axvline(median_val, color='green', linestyle='-.', label='Median')
        ax3.axvline(q25, color='purple', linestyle=':', linewidth=2, label=f'Q1: {q25:.4f}')
        ax3.axvline(q75, color='purple', linestyle=':', linewidth=2, label=f'Q3: {q75:.4f}')

        stats_text = f"Mean = {mean_val:.5f}\nMedian = {median_val:.5f}\nStd = {std_val:.5f}\nIQR = {iqr:.5f}"
        ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

# --- TAB 3: CORRELATION ---
with tab3:
    st.subheader("Dynamic Correlation Analysis")
    col_x, col_y = st.columns(2)
    available_cols = list(df_all.columns)

    def_x = 'Euronext 100 (Market)' if 'Euronext 100 (Market)' in available_cols else available_cols[0]
    def_y = 'Thales' if 'Thales' in available_cols else available_cols[-1]

    with col_x:
        asset_x = st.selectbox("X-Axis (Benchmark)", options=available_cols,
                               index=available_cols.index(def_x) if def_x in available_cols else 0)
    with col_y:
        asset_y = st.selectbox("Y-Axis (Asset)", options=available_cols,
                               index=available_cols.index(def_y) if def_y in available_cols else 0)

    data_corr = df_all[[asset_x, asset_y]].pct_change().dropna()

    if not data_corr.empty:
        x_vals = data_corr[asset_x]
        y_vals = data_corr[asset_y]

        m, b = np.polyfit(x_vals, y_vals, 1)

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.scatter(x_vals, y_vals, alpha=0.6, color='darkblue', edgecolors='white', s=60)

        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax4.plot(x_line, m * x_line + b, color='red', linewidth=2, label=f'Beta = {m:.2f}')

        ax4.set_title(f"Correlation: {asset_y} vs. {asset_x}\nCorrelation = {x_vals.corr(y_vals):.3f}")
        ax4.set_xlabel(f"{asset_x} Returns")
        ax4.set_ylabel(f"{asset_y} Returns")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig4)
