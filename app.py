# app.py 顶部引入部分
import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st

# ================================
# Basic Settings
# ================================
# 设置为英文风格
plt.style.use('seaborn-v0_8-whitegrid') # 或者 'ggplot'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

START_DEFAULT = date(2018, 1, 1)
END_DEFAULT = date.today()

OUTPUT_DIR = "liquidity_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# 数据源（FRED + yfinance）
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
    "repo_gc":          "TGCRRATE",     # Tri-party GC Repo Rate

}



YF_SYMBOLS = {
    "russell2000": "^RUT",   # stooq 格式
    "dxy": "DX-Y.NYB"
}



# ================================
# 流动性评分配置
# ================================
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
}

# ================================
# 工具函数：抓数据
# ================================
@st.cache_data(show_spinner=False)
def fetch_fred_series(series_dict, start_date, end_date):
    series_list = []

    for name, code in series_dict.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            df = pd.read_csv(url)

            # --- 修复：兼容 FRED 两种 CSV 格式 ---
            if "DATE" in df.columns:
                df.rename(columns={"DATE": "date"}, inplace=True)
            elif "observation_date" in df.columns:
                df.rename(columns={"observation_date": "date"}, inplace=True)
            else:
                st.warning(f"FRED 指标 {name} ({code}) CSV 缺少 DATE/observation_date，列为：{df.columns}")
                continue

            if code not in df.columns:
                st.warning(f"FRED 指标 {name} ({code}) 缺少主数据列，列为：{df.columns}")
                continue

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # 按日期过滤
            s = df[code].loc[
                (df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))
            ].copy()
            s.name = name
            series_list.append(s)

        except Exception as e:
            st.warning(f"FRED 指标 {name} ({code}) 获取失败：{e}")

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)


@st.cache_data(show_spinner=False)
def fetch_yfinance_series(symbols_dict, start_date, end_date):
    series_list = []
    for name, symbol in symbols_dict.items():
        try:
            # 下载数据
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.warning(f"yfinance 指标 {name} ({symbol}) 下载为空")
                continue

            # === 修复核心：处理 yfinance 新版的多层索引 (MultiIndex) ===
            if isinstance(df.columns, pd.MultiIndex):
                # 如果列是多层的 (Price, Ticker)，尝试提取该 Ticker 的一层
                try:
                    df = df.xs(symbol, axis=1, level=1)
                except KeyError:
                    # 如果 level 1 不是 ticker，尝试直接降维
                    df.columns = df.columns.get_level_values(0)

            # 优先使用 Adj Close，否则使用 Close
            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                st.warning(f"yfinance 指标 {name} ({symbol}) 缺少 Close/Adj Close 列")
                continue

            s.name = name
            series_list.append(s)
        except Exception as e:
            st.warning(f"yfinance 指标 {name} ({symbol}) 获取失败：{e}")

    if not series_list:
        return pd.DataFrame()

    return pd.concat(series_list, axis=1)


def build_panel(start_date, end_date):
    fred_df = fetch_fred_series(FRED_SERIES, start_date, end_date)
    yf_df = fetch_yfinance_series(YF_SYMBOLS, start_date, end_date)

    raw_df = pd.concat([fred_df, yf_df], axis=1).sort_index()

    all_df = (
        raw_df
        .resample("D")
        .last()
        .ffill()
    )
    return all_df

# ================================
# 画图函数
# ================================
def plot_series(df, columns, title="", ylabel="", rolling=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    plotted_any = False
    for col in columns:
        if col not in df.columns:
            st.warning(f"列 {col} 不存在，跳过")
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if rolling:
            series = series.rolling(rolling).mean()
        ax.plot(series.index, series.values, label=col)
        plotted_any = True

    if not plotted_any:
        st.warning(f"{title} 没有可画的数据")
        return

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def plot_onrrp_tga(df):
    if "on_rrp" not in df.columns or "tga" not in df.columns:
        st.warning("缺少 on_rrp 或 tga，无双轴图")
        return

    ser_on = df["on_rrp"].dropna()
    ser_tga = df["tga"].dropna()
    if ser_on.empty or ser_tga.empty:
        st.warning("on_rrp 或 tga 数据为空，无法画双轴图")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ser_on.index, ser_on.values, label="ON RRP", color="tab:blue", linewidth=2)
    ax1.set_ylabel("ON RRP", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ser_tga.index, ser_tga.values, label="TGA", color="tab:orange", linewidth=2, linestyle="--")
    ax2.set_ylabel("TGA", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    st.pyplot(fig)


def plot_equity_indices(df):
    # 现在只看 Russell 2000
    col = "russell2000"
    if col not in df.columns:
        st.warning("Russell 2000 数据不足，无法绘制。")
        return

    series = df[col].dropna()
    if series.empty:
        st.warning("Russell 2000 数据为空。")
        return

    # 归一化，从 1 开始
    norm = series / series.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(norm.index, norm.values, label="russell2000")

    ax.set_title("Russell 2000 Index (Normalized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (normalized to 1)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def plot_overlay_with_correlation(df, indicator_col, target_col="russell2000", window=90, title_prefix=""):
    """
    Plot Overlay: Indicator vs Target with Rolling Correlation
    Includes dynamic labeling of the latest correlation value.
    """
    if indicator_col not in df.columns or target_col not in df.columns:
        st.warning(f"Missing data: {indicator_col} or {target_col}")
        return

    plot_df = df[[indicator_col, target_col]].dropna()
    if plot_df.empty:
        return

    # Calculate Rolling Correlation
    rolling_corr = plot_df[indicator_col].rolling(window=window).corr(plot_df[target_col])
    
    # Get the latest valid correlation value
    valid_corr = rolling_corr.dropna()
    if valid_corr.empty:
        latest_corr = 0.0
        last_date = plot_df.index[-1]
    else:
        latest_corr = valid_corr.iloc[-1]
        last_date = valid_corr.index[-1]

    # Create Figure
    fig, (ax1, ax_corr) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
    
    # --- Top Chart: Dual Axis ---
    color_ind = 'tab:blue'
    color_target = 'tab:gray'
    
    # Left Axis
    ax1.plot(plot_df.index, plot_df[indicator_col], color=color_ind, label=indicator_col, linewidth=1.5)
    ax1.set_ylabel(indicator_col, color=color_ind, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_ind)
    
    # Right Axis
    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df[target_col], color=color_target, label=target_col, linestyle='--', alpha=0.6, linewidth=1)
    ax2.set_ylabel(target_col, color=color_target, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_target)
    
    # --- Dynamic Title with Score ---
    # 标题格式：[Category] Indicator vs Target (Corr: +0.85)
    corr_str = f"{latest_corr:+.2f}" # 强制显示正负号
    title_text = f"{title_prefix} {indicator_col} vs {target_col} (Corr: {corr_str})"
    
    # 根据相关性正负改变标题中数值部分的颜色需要复杂的富文本支持，这里简单处理：
    # 我们直接把标题设为黑色，但在下面标注颜色
    ax1.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Bottom Chart: Rolling Correlation ---
    if not valid_corr.empty:
        ax_corr.plot(valid_corr.index, valid_corr, color='black', linewidth=1)
        
        # Fill areas
        ax_corr.fill_between(valid_corr.index, 0, valid_corr, where=(valid_corr > 0), color='green', alpha=0.3)
        ax_corr.fill_between(valid_corr.index, 0, valid_corr, where=(valid_corr < 0), color='red', alpha=0.3)
        
        # === 【新增功能】在曲线最右端添加数值标签 ===
        text_color = 'green' if latest_corr > 0 else 'red'
        ax_corr.annotate(
            f"{latest_corr:.2f}", 
            xy=(last_date, latest_corr), 
            xytext=(5, 0),             # 稍微向右偏移 5 个点
            textcoords="offset points", 
            fontsize=12, 
            fontweight='bold', 
            color=text_color,
            va='center'
        )
        # 画一个圆点标记最后的位置
        ax_corr.scatter(last_date, latest_corr, color=text_color, s=50, zorder=5)

    ax_corr.set_ylabel(f"{window}d Rolling Corr", fontsize=10)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_corr.grid(True, linestyle='--', alpha=0.3)
    
    plt.subplots_adjust(hspace=0.05)
    st.pyplot(fig)

# ================================
# 新增：相关性分析函数
# ================================
def plot_correlation_analysis(df, target_col="russell2000"):
    if target_col not in df.columns:
        st.warning(f"缺少目标列 {target_col}，无法进行相关性分析")
        return

    # 1. 计算相关性矩阵
    # 只需要数值型列
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # 提取与 target 的相关性，并去掉 target 自身
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=True)
    else:
        return

    # --- 图表 1：静态相关性排行 ---
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    # 根据正负设定颜色
    colors = ['#ff9999' if x < 0 else '#99ff99' for x in target_corr.values]
    bars = ax.barh(target_corr.index, target_corr.values, color=colors)
    
    ax.set_title(f"Correlation with {target_col} (Selected Period)")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 在柱子旁标注数值
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.05 if width > 0 else width - 0.15
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=9)

    fig1.tight_layout()
    
    
    # --- 图表 2：滚动相关性 (Rolling Correlation) ---
    # 选取几个最重要的宏观变量进行观察
    key_macro_vars = ["bank_reserves", "dxy", "hy_spread", "t_bill_3m"]
    valid_vars = [c for c in key_macro_vars if c in df.columns]
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # 计算 90 天滚动相关性
    window = 90
    for col in valid_vars:
        rolling_corr = df[target_col].rolling(window).corr(df[col])
        ax2.plot(rolling_corr.index, rolling_corr, label=f"{col} (90d roll)")
        
    ax2.set_title(f"90-Day Rolling Correlation with {target_col}")
    ax2.set_ylabel("Correlation")
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_ylim(-1, 1)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    fig2.tight_layout()

    return fig1, fig2


# ================================
# 流动性评分
# ================================
# ================================
# Liquidity Scoring Function
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
            "indicator": col,
            "latest_value": series.iloc[-1],
            "mean": mean,
            "std": std,
            "z_raw": z,
            "z_tight": z_tight,
            "weight": weight,
        })

        weighted_z += z_tight * weight
        total_weight += weight

    if total_weight == 0:
        raise ValueError("No available indicators for scoring")

    # --- 关键修复：这里必须先计算 score，才能在下面使用 ---
    score = 50 - 10 * (weighted_z / total_weight)
    score = max(0, min(100, score))

    # --- 设置英文标签 ---
    if score >= 60:
        label = "Loose / Accommodative"
    elif score <= 40:
        label = "Tight / Restrictive"
    else:
        label = "Neutral"

    detail_df = pd.DataFrame(z_details).set_index("indicator")

    return score, label, detail_df, (start_date, end_date)
# ================================
# Signal Generation
# ================================
def analyze_market_signal(df, score, target_col="russell2000", window=90):
    """
    根据流动性评分和相关性生成交易结论
    """
    # 1. 定义信号方向与强度
    # Score 范围 0-100, 中位数 50
    deviation = score - 50
    strength = abs(deviation) * 2  # 将 0-50 的偏离放大到 0-100%
    
    if score >= 60:
        signal = "LONG (Buy)"
        bias_color = "green"
        sentiment = "Bullish / Liquidity Supported"
    elif score >= 52:
        signal = "WEAK LONG"
        bias_color = "lightgreen"
        sentiment = "Mildly Bullish"
    elif score <= 40:
        signal = "SHORT (Sell)"
        bias_color = "red"
        sentiment = "Bearish / Liquidity Drain"
    elif score <= 48:
        signal = "WEAK SHORT"
        bias_color = "lightcoral"
        sentiment = "Mildly Bearish"
    else:
        signal = "NEUTRAL"
        bias_color = "gray"
        sentiment = "Choppy / No Clear Trend"

    # 2. 寻找当前的主导因子 (Dominant Driver)
    # 计算主要因子与 Russell 2000 的最新 90 天相关性
    macro_vars = ["bank_reserves", "tga", "on_rrp", "t_bill_3m", "hy_spread", "dxy", "vix", "fed_balance_sheet"]
    valid_vars = [c for c in macro_vars if c in df.columns]
    
    max_corr_val = 0
    dominant_driver = "None"
    
    if valid_vars:
        # 获取最近 window 天的数据进行计算
        recent_df = df.iloc[-window:]
        corrs = recent_df[valid_vars].corrwith(recent_df[target_col])
        
        # 找到绝对值最大的
        abs_corrs = corrs.abs().sort_values(ascending=False)
        if not abs_corrs.empty:
            driver_name = abs_corrs.index[0]
            driver_val = corrs[driver_name]
            dominant_driver = f"{driver_name} ({driver_val:+.2f})"
            max_corr_val = driver_val

    return {
        "signal": signal,
        "strength": f"{strength:.1f}%",
        "color": bias_color,
        "sentiment": sentiment,
        "driver": dominant_driver
    }
# ================================
# Streamlit 主程序
# ================================
def main():
    st.set_page_config(page_title="USD Liquidity Dashboard", layout="wide")
    st.title("🧊 USD Macro Liquidity Dashboard")

    # ==== Sidebar ====
    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input("Start Date", START_DEFAULT)
        end_date = st.date_input("End Date", END_DEFAULT)
        window_days = st.slider("Scoring Window (Days)", 180, 730, 365)

        if start_date >= end_date:
            st.error("Start Date must be before End Date")
            return
            
        st.markdown("---")
        st.caption("Disclaimer: This dashboard is for informational purposes only, not financial advice.")

    # 删除了 st.info("Fetching...")
    
    # 获取数据
    all_df = build_panel(start_date, end_date)
    
    if all_df.empty:
        st.error("Data Fetch Failed: DataFrame is empty.")
        return
    
    # 删除了 st.success("Data Updated Successfully")

    # =======================
    # 1. Calculate Score FIRST
    # =======================
    score_res = None
    try:
        # 计算分数
        score, label, detail_df, (s, e) = compute_liquidity_score(
            all_df, LIQUIDITY_CONFIG, window_days
        )
        # 生成交易结论
        signal_data = analyze_market_signal(all_df, score)
        score_res = (score, label, detail_df)
        
        # =======================
        # 2. Display Conclusion Row
        # =======================
        st.markdown("### 🎯 Market Signal & Conclusion")
        
        con_col1, con_col2, con_col3, con_col4 = st.columns(4)
        
        with con_col1:
            st.metric("Recommendation", signal_data["signal"])
        with con_col2:
            st.metric("Signal Strength", signal_data["strength"], delta=label, delta_color="normal")
        with con_col3:
            st.metric("Liquidity Score", f"{score:.1f}", help="0-100, >50 is Bullish")
        with con_col4:
            st.metric("Dominant Driver", signal_data["driver"], help="The factor with highest correlation right now")

        # 显示带颜色的状态条
        if signal_data["color"] == "green":
            st.success(f"✅ **Conclusion:** {signal_data['sentiment']}. Liquidity conditions are supportive.")
        elif signal_data["color"] == "red":
            st.error(f"🛑 **Conclusion:** {signal_data['sentiment']}. Liquidity is tightening, caution advised.")
        else:
            st.warning(f"⚠️ **Conclusion:** {signal_data['sentiment']}. Market lacks clear liquidity direction.")

    except Exception as e:
        st.error(f"Could not calculate signal: {e}")

    st.markdown("---")

    # =======================
    # 3. Chart Section
    # =======================
    st.header("🔬 Deep Dive: Macro Factors vs Russell 2000")
    st.caption("Upper: Dual-Axis Price Action | Lower: 90-Day Rolling Correlation")

    st.subheader("1. Core Liquidity Dynamics")
    col1, col2 = st.columns(2)
    with col1:
        plot_overlay_with_correlation(all_df, "bank_reserves", title_prefix="[Central Bank Liquidity]")
    with col2:
        plot_overlay_with_correlation(all_df, "fed_balance_sheet", title_prefix="[Fed Balance Sheet]")

    st.subheader("2. Liquidity Drain & Buffer")
    col3, col4 = st.columns(2)
    with col3:
        plot_overlay_with_correlation(all_df, "tga", title_prefix="[Treasury Account (TGA)]")
    with col4:
        plot_overlay_with_correlation(all_df, "on_rrp", title_prefix="[Reverse Repo (ON RRP)]")
    
    st.subheader("3. Rates & Risk Sentiment")
    col5, col6 = st.columns(2)
    with col5:
        plot_overlay_with_correlation(all_df, "t_bill_3m", title_prefix="[Risk-Free Rate (3M)]")
    with col6:
        plot_overlay_with_correlation(all_df, "dxy", title_prefix="[US Dollar Index]")

    col7, col8 = st.columns(2)
    with col7:
        plot_overlay_with_correlation(all_df, "hy_spread", title_prefix="[Credit Spreads]")
    with col8:
        plot_overlay_with_correlation(all_df, "vix", title_prefix="[Volatility (VIX)]")

    # =======================
    # 4. Detail Table
    # =======================
    if score_res:
        with st.expander("📊 See Liquidity Score Details"):
            st.dataframe(score_res[2])

if __name__ == "__main__":
    main()






















