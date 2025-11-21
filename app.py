import os
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# ================================
# Basic Settings
# ================================
# 使用通用样式
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 为了支持 7 年回测，默认开始日期设为 2010 年 ---
START_DEFAULT = date(2010, 1, 1)
END_DEFAULT = date.today()

OUTPUT_DIR = "liquidity_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Configuration (FRED + YFinance)
# ================================
FRED_SERIES = {
    "bank_reserves":    "WRESBAL",
    "on_rrp":           "RRPONTSYD",
    "fed_balance_sheet":"WALCL",
    "tga":              "WTREGEN",
    "sofr":             "SOFR",
    "t_bill_1m":        "DGS1MO",
    "t_bill_3m":        "DGS3MO",
    "hy_spread":        "BAMLH0A0HYM2",
    "vix":              "VIXCLS",
    "repo_gc":          "TGCRRATE",
    # Global Macros
    "ecb_assets":       "ECBASSETSW",
    "boj_assets":       "JPNASSETS",
    "real_yield_10y":   "DFII10",
}

YF_SYMBOLS = {
    "russell2000": "IWM",
    "dxy":         "DX-Y.NYB",
    "usd_jpy":     "JPY=X",
}

LIQUIDITY_CONFIG = {
    "bank_reserves":      {"sign": -1, "weight": 1.5},
    "fed_balance_sheet":  {"sign": -1, "weight": 1.0},
    "on_rrp":             {"sign": +1, "weight": 1.0},
    "tga":                {"sign": +1, "weight": 1.0},
    "sofr":               {"sign": +1, "weight": 1.5},
    "repo_gc":            {"sign": +1, "weight": 1.2},
    "t_bill_3m":          {"sign": +1, "weight": 1.0},
    "hy_spread":          {"sign": +1, "weight": 1.5},
    "dxy":                {"sign": +1, "weight": 1.0},
    "vix":                {"sign": +1, "weight": 1.0},
    "ecb_assets":         {"sign": -1, "weight": 0.5},
    "boj_assets":         {"sign": -1, "weight": 0.5},
    "real_yield_10y":     {"sign": +1, "weight": 1.2},
    "usd_jpy":            {"sign": -1, "weight": 0.8},
}

# ================================
# Data Fetching Functions
# ================================
@st.cache_data(show_spinner=False)
def fetch_fred_series(series_dict, start_date, end_date):
    series_list = []
    for name, code in series_dict.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            df = pd.read_csv(url)
            if "DATE" in df.columns:
                df.rename(columns={"DATE": "date"}, inplace=True)
            elif "observation_date" in df.columns:
                df.rename(columns={"observation_date": "date"}, inplace=True)
            else:
                continue
            
            if code not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            s = df[code].loc[
                (df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))
            ].copy()
            s.name = name
            series_list.append(s)
        except Exception:
            continue

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)

@st.cache_data(show_spinner=False)
def fetch_yfinance_series(symbols_dict, start_date, end_date):
    series_list = []
    for name, symbol in symbols_dict.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(symbol, axis=1, level=1)
                except KeyError:
                    df.columns = df.columns.get_level_values(0)

            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                continue

            s.name = name
            series_list.append(s)
        except Exception:
            continue

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)

def build_panel(start_date, end_date):
    fred_df = fetch_fred_series(FRED_SERIES, start_date, end_date)
    yf_df = fetch_yfinance_series(YF_SYMBOLS, start_date, end_date)
    raw_df = pd.concat([fred_df, yf_df], axis=1).sort_index()
    all_df = raw_df.resample("D").last().ffill()
    return all_df

# ================================
# Plotting Functions (Enhanced with Z-Window)
# ================================
def plot_overlay_with_correlation(df, indicator_col, target_col="russell2000", window=90, z_window=252, title_prefix=""):
    """
    Overlay plot with Divergence Analysis (Z-Score Gap).
    Accepts z_window for customizable lookback.
    """
    if indicator_col not in df.columns or target_col not in df.columns:
        st.warning(f"Missing data: {indicator_col} or {target_col}")
        return

    plot_df = df[[indicator_col, target_col]].dropna()
    if plot_df.empty:
        return

    # 1. Rolling Correlation
    rolling_corr = plot_df[indicator_col].rolling(window=window).corr(plot_df[target_col])
    valid_corr = rolling_corr.dropna()
    
    if valid_corr.empty:
        latest_corr = 0.0
        last_date = plot_df.index[-1]
    else:
        latest_corr = valid_corr.iloc[-1]
        last_date = valid_corr.index[-1]

    # 2. Z-Score Gap Calculation
    lookback = z_window
    if len(plot_df) > lookback:
        subset = plot_df.iloc[-lookback:]
    else:
        subset = plot_df
    
    def calc_z(series):
        if series.std() == 0: return 0
        return (series.iloc[-1] - series.mean()) / series.std()
    
    z_target = calc_z(subset[target_col])
    z_indic = calc_z(subset[indicator_col])
    
    corr_sign = np.sign(latest_corr) if latest_corr != 0 else 1
    gap = z_target - (corr_sign * z_indic)
    
    # Gap Status Logic
    if gap > 1.5:
        gap_status = "OVERVALUED ●"
        gap_color = "#D62728"
    elif gap < -1.5:
        gap_status = "UNDERVALUED ●"
        gap_color = "#2CA02C"
    else:
        gap_status = "Fair Value"
        gap_color = "#333333"

    # 3. Plotting
    fig, (ax1, ax_corr) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
    
    color_ind = 'tab:blue'
    color_target = 'tab:gray'
    
    # Top Chart
    ax1.plot(plot_df.index, plot_df[indicator_col], color=color_ind, label=indicator_col, linewidth=2)
    ax1.set_ylabel(indicator_col, color=color_ind, fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_ind)
    
    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df[target_col], color=color_target, label=target_col, linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel(target_col, color=color_target, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_target)
    
    corr_str = f"{latest_corr:+.2f}"
    title_text = f"{title_prefix} {indicator_col} vs {target_col} (Corr: {corr_str})"
    ax1.set_title(title_text, fontsize=18, fontweight='bold', pad=20)
    
    # Z-Score Gap Label (Large Font)
    gap_text = f"Z-Score Gap ({lookback}d): {gap:+.2f}σ  [{gap_status}]"
    ax1.text(0.5, 0.92, gap_text, transform=ax1.transAxes, ha='center', 
             fontsize=18, fontweight='heavy', color=gap_color, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    ax1.grid(True, linestyle='--', alpha=0.3)

    # Bottom Chart
    if not valid_corr.empty:
        ax_corr.plot(valid_corr.index, valid_corr, color='black', linewidth=1)
        ax_corr.fill_between(valid_corr.index, 0, valid_corr, where=(valid_corr > 0), color='green', alpha=0.3)
        ax_corr.fill_between(valid_corr.index, 0, valid_corr, where=(valid_corr < 0), color='red', alpha=0.3)
        
        text_color = 'green' if latest_corr > 0 else 'red'
        ax_corr.annotate(
            f"{latest_corr:.2f}", 
            xy=(last_date, latest_corr), 
            xytext=(5, 0), 
            textcoords="offset points", 
            fontsize=14, fontweight='bold', color=text_color, va='center'
        )
        ax_corr.scatter(last_date, latest_corr, color=text_color, s=60, zorder=5)

    ax_corr.set_ylabel(f"{window}d Rolling Corr", fontsize=12)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_corr.grid(True, linestyle='--', alpha=0.3)
    plt.subplots_adjust(hspace=0.1)
    st.pyplot(fig)

# ================================
# Logic: Score & Signal
# ================================
def compute_liquidity_score(df, config=LIQUIDITY_CONFIG, window_days=365):
    valid_df = df.dropna(how="all")
    if valid_df.empty:
        raise ValueError("No valid data for scoring")

    end_date = valid_df.index.max()
    start_date = end_date - pd.Timedelta(days=window_days)
    window_df = df.loc[start_date:end_date]

    z_details = []
    total_weight = 0.0
    weighted_z = 0.0

    for col, meta in config.items():
        if col not in window_df.columns:
            continue
        series = window_df[col].dropna()
        if len(series) < 30:
            continue
        
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            continue

        z = (series.iloc[-1] - mean) / std
        z_tight = meta["sign"] * z
        weight = meta["weight"]

        z_details.append({
            "indicator": col, "latest_value": series.iloc[-1],
            "mean": mean, "std": std, "z_raw": z, "z_tight": z_tight, "weight": weight
        })
        weighted_z += z_tight * weight
        total_weight += weight

    if total_weight == 0:
        raise ValueError("No available indicators")

    score = 50 - 10 * (weighted_z / total_weight)
    score = max(0, min(100, score))

    if score >= 60:
        label = "Loose / Accommodative"
    elif score <= 40:
        label = "Tight / Restrictive"
    else:
        label = "Neutral"

    detail_df = pd.DataFrame(z_details).set_index("indicator")
    return score, label, detail_df, (start_date, end_date)

def analyze_market_signal(df, score, target_col="russell2000", window=90):
    deviation = score - 50
    strength = abs(deviation) * 2
    
    if score >= 60:
        signal, color, sentiment = "LONG IWM", "green", "Bullish / Liquidity Supported"
    elif score >= 52:
        signal, color, sentiment = "WEAK LONG IWM", "lightgreen", "Mildly Bullish"
    elif score <= 40:
        signal, color, sentiment = "SHORT IWM", "red", "Bearish / Liquidity Drain"
    elif score <= 48:
        signal, color, sentiment = "WEAK SHORT IWM", "lightcoral", "Mildly Bearish"
    else:
        signal, color, sentiment = "NEUTRAL", "gray", "Choppy / No Clear Trend"

    macro_vars = list(LIQUIDITY_CONFIG.keys())
    valid_vars = [c for c in macro_vars if c in df.columns]
    
    dominant_driver = "None"
    if valid_vars:
        recent_df = df.iloc[-window:]
        corrs = recent_df[valid_vars].corrwith(recent_df[target_col])
        abs_corrs = corrs.abs().sort_values(ascending=False)
        if not abs_corrs.empty:
            driver_name = abs_corrs.index[0]
            driver_val = corrs[driver_name]
            dominant_driver = f"{driver_name} ({driver_val:+.2f})"

    return {"signal": signal, "strength": f"{strength:.1f}%", "color": color, 
            "sentiment": sentiment, "driver": dominant_driver}

# ================================
# Analysis Text Generators (Smart & Safe)
# ================================
def _get_driver_narrative(driver):
    mapping = {
        "t_bill_3m": "Fed Policy & Cost of Capital",
        "bank_reserves": "Central Bank Liquidity Support",
        "hy_spread": "Credit Risk & Recession Fears",
        "dxy": "Currency Headwinds & Global Capital Flows",
        "tga": "Fiscal Spending & Treasury General Account",
        "usd_jpy": "Global Carry Trade Dynamics",
        "vix": "Market Fear & Hedging Demand",
        "fed_balance_sheet": "Quantitative Tightening (QT)",
        "on_rrp": "Liquidity Buffer Dynamics",
        "ecb_assets": "Global Central Bank Liquidity (ECB)",
        "boj_assets": "Global Central Bank Liquidity (BOJ)",
        "real_yield_10y": "Real Cost of Capital"
    }
    return mapping.get(driver, "Macro Factors")

def _render_indicator_card(title, tag, value, distinct_impact, watch_item):
    with st.expander(f"{title}", expanded=True):
        st.markdown(f"**Status:** `{tag}` | **Value:** `{value}`")
        st.markdown(f"**💥 Impact on IWM:** {distinct_impact}")
        st.markdown(f"**👀 Watch:** *{watch_item}*")

def display_analysis_section(df, score, signal_data):
    if df.empty: return

    st.markdown("### 📝 Strategic Analysis & Key Watchlist")
    
    driver_raw = str(signal_data.get('driver', 'None'))
    driver_name = driver_raw.split(" ")[0] if " " in driver_raw else driver_raw
    driver_desc = _get_driver_narrative(driver_name)
    
    narrative = f"""
    **Current Market Regime:** The market is currently in a **{signal_data['sentiment']}** state (Score: {score:.1f}). 
    The Russell 2000 is showing highest sensitivity to **{driver_name}**, indicating that 
    **{driver_desc}** is the primary theme driving price action.
    """
    
    geo_context = """
    **🌐 Global Context & Geopolitics:**
    With the current disconnect between US economic resilience and global slowdowns (China/Europe), 
    volatility in **Currency Markets (Yen/Euro)** and **Oil** remains a key external risk. 
    """
    st.info(narrative + geo_context)

    latest = df.iloc[-1]
    val_tbill = latest.get('t_bill_3m', 0)
    val_tga = latest.get('tga', 0)
    val_spread = latest.get('hy_spread', 0)
    
    # --- Smart Logic: Switch between JPY and DXY ---
    recent_df = df.iloc[-90:]
    corr_jpy = recent_df['usd_jpy'].corr(recent_df['russell2000']) if 'usd_jpy' in df else 0
    corr_dxy = recent_df['dxy'].corr(recent_df['russell2000']) if 'dxy' in df else 0
    
    # Default to JPY
    card2_title = "🇯🇵 2. USD/JPY (Carry Trade)"
    card2_tag = "External Risk"
    card2_val = f"{latest.get('usd_jpy', 0):.2f}"
    txt_card2_impact = "A rapid drop in USD/JPY (Yen strength) signals a Carry Trade unwind. This forces hedge funds to liquidate risk assets like IWM."
    txt_card2_watch = "Watch for: BOJ rate hikes or intervention."
    
    # Switch to DXY if correlation is stronger
    if abs(corr_dxy) > abs(corr_jpy):
        card2_title = "🌍 2. DXY (Euro/Global FX)"
        card2_tag = "Global Capital Flow"
        card2_val = f"{latest.get('dxy', 0):.2f}"
        txt_card2_impact = "A rising Dollar (often due to weak Euro/CNY) tightens global financial conditions, hurting earnings and causing capital flight."
        txt_card2_watch = "Watch for: ECB/PBOC Policy divergence vs Fed."

    # Safe Variables
    txt_us_impact = "The Russell 2000 is composed of floating-rate debt zombies. If Rates stay high, refinancing walls will crush earnings."
    txt_us_watch = "Watch for: CPI prints & Fed 'Higher for Longer' rhetoric."
    txt_tga_impact = "If TGA rises (Treasury issuing debt) while ON RRP is flat/empty, liquidity is drained directly from Bank Reserves (Stocks down)."
    txt_tga_watch = "Watch for: Treasury Quarterly Refunding Announcement (QRA)."
    txt_credit_impact = "The ultimate truth-teller. If Spreads widen (>400bps), the 'Soft Landing' narrative is dead. IWM will underperform SPY significantly."
    txt_credit_watch = "Watch for: Corporate defaults or weak earnings guidance."

    st.subheader("🔍 Critical Indicators to Watch")
    c1, c2 = st.columns(2)
    
    with c1:
        _render_indicator_card("🇺🇸 1. US Rates", "High Impact", f"{val_tbill:.2f}%", txt_us_impact, txt_us_watch)
        _render_indicator_card(card2_title, card2_tag, card2_val, txt_card2_impact, txt_card2_watch)

    with c2:
        tga_dis = val_tga / 1000 if pd.notnull(val_tga) else 0
        _render_indicator_card("🏦 3. Liquidity (TGA)", "Flow Dynamics", f"${tga_dis:.0f}B", txt_tga_impact, txt_tga_watch)
        _render_indicator_card("📉 4. Credit Spreads", "Recession Alarm", f"{val_spread:.0f} bps", txt_credit_impact, txt_credit_watch)

# ================================
# Main Application
# ================================
def main():
    st.set_page_config(page_title="US Russell 2000 Investment Reference", layout="wide")
    st.title("🧊 US Russell 2000 Investment Reference")

    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input("Start Date", START_DEFAULT)
        end_date = st.date_input("End Date", END_DEFAULT)
        
        st.markdown("---")
        st.subheader("Parameters")
        
        # === Updated: English Help & Caption ===
        window_days = st.slider(
            "Scoring Window (Days)", 
            min_value=180, 
            max_value=730, 
            value=365,
            help="Determines the baseline period for calculating the Liquidity Score.\n\n- 365 days (Default): Compares current liquidity against the past year.\n- Lower: More sensitive to recent changes (Short-term view).\n- Higher: Filters noise to show long-term trends (Long-term view)."
        )
        st.caption("💡 Sets the 'Memory Length' for Liquidity Scoring (Z-Score Baseline).")
        
        # === Updated: English Help & Caption ===
        z_lookback = st.selectbox(
            "Z-Score Lookback Period",
            options=[252, 504, 756, 1260, 1764], 
            index=0, 
            format_func=lambda x: f"{x} Days ({x//252} Year{'s' if x>252 else ''})",
            help="Determines the historical window for the 'Z-Score Gap' analysis above the charts. For example, selecting 5 Years checks if the current divergence is extreme relative to the past 5 years."
        )
        st.caption("💡 Sets the historical window for Divergence (Gap) Analysis.")

        if start_date >= end_date:
            st.error("Start Date must be before End Date")
            return
        st.markdown("---")
        st.caption("Disclaimer: Not financial advice.")

    all_df = build_panel(start_date, end_date)
    if all_df.empty:
        st.error("Data Fetch Failed: DataFrame is empty.")
        return

    # --- Score & Signal ---
    score_res = None
    try:
        score, label, detail_df, _ = compute_liquidity_score(all_df, LIQUIDITY_CONFIG, window_days)
        signal_data = analyze_market_signal(all_df, score)
        score_res = (score, label, detail_df)
        
        st.markdown("### 🎯 Market Signal & Conclusion")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Signal for IWM", signal_data["signal"], help="Specific to Russell 2000 Small Caps")
        with c2: st.metric("Signal Strength", signal_data["strength"], delta=label, delta_color="normal")
        with c3: st.metric("Liquidity Score", f"{score:.1f}")
        with c4: st.metric("Dominant Driver", signal_data["driver"])

        if signal_data["color"] == "green":
            st.success(f"✅ {signal_data['sentiment']}")
        elif signal_data["color"] == "red":
            st.error(f"🛑 {signal_data['sentiment']}")
        else:
            st.warning(f"⚠️ {signal_data['sentiment']}")

        # Call Analysis Section
        display_analysis_section(all_df, score, signal_data)

    except Exception as e:
        st.error(f"Error in analysis: {e}")

    st.markdown("---")

    # --- Charts ---
    st.header("🔬 Deep Dive: Macro Factors vs Russell 2000")
    
    st.subheader("1. Core Liquidity Dynamics")
    c1, c2 = st.columns(2)
    with c1: plot_overlay_with_correlation(all_df, "bank_reserves", title_prefix="[Central Bank Liquidity]", z_window=z_lookback)
    with c2: plot_overlay_with_correlation(all_df, "fed_balance_sheet", title_prefix="[Fed Balance Sheet]", z_window=z_lookback)

    st.subheader("2. Liquidity Drain & Buffer")
    c3, c4 = st.columns(2)
    with c3: plot_overlay_with_correlation(all_df, "tga", title_prefix="[Treasury Account]", z_window=z_lookback)
    with c4: plot_overlay_with_correlation(all_df, "on_rrp", title_prefix="[Reverse Repo]", z_window=z_lookback)
    
    st.subheader("3. Rates & Risk Sentiment")
    c5, c6 = st.columns(2)
    with c5: plot_overlay_with_correlation(all_df, "t_bill_3m", title_prefix="[Risk-Free Rate]", z_window=z_lookback)
    with c6: plot_overlay_with_correlation(all_df, "dxy", title_prefix="[US Dollar Index]", z_window=z_lookback)

    c7, c8 = st.columns(2)
    with c7: plot_overlay_with_correlation(all_df, "hy_spread", title_prefix="[Credit Spreads]", z_window=z_lookback)
    with c8: plot_overlay_with_correlation(all_df, "vix", title_prefix="[Volatility]", z_window=z_lookback)

    st.subheader("4. Global Liquidity & External Shocks")
    c9, c10 = st.columns(2)
    with c9: plot_overlay_with_correlation(all_df, "ecb_assets", title_prefix="[ECB Assets]", z_window=z_lookback)
    with c10: plot_overlay_with_correlation(all_df, "boj_assets", title_prefix="[BOJ Assets]", z_window=z_lookback)
    
    c11, c12 = st.columns(2)
    with c11: plot_overlay_with_correlation(all_df, "usd_jpy", title_prefix="[USD/JPY Carry]", z_window=z_lookback)
    with c12: plot_overlay_with_correlation(all_df, "real_yield_10y", title_prefix="[Real Yields]", z_window=z_lookback)

    if score_res:
        with st.expander("📊 See Liquidity Score Details"):
            st.dataframe(score_res[2])

if __name__ == "__main__":
    main()
