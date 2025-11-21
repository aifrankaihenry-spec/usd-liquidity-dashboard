# app.py
import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# ================================
# 基本设置
# ================================
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
    "dxy":              "TWEXBMTH",
    "vix":              "VIXCLS",
    "repo_gc":          "TGCRRATE",     # Tri-party GC Repo Rate

    # ⬇️ 这里是新增的三大股指（FRED 代码）
    "sp500":            "SP500",        # S&P 500 :contentReference[oaicite:0]{index=0}
    "nasdaq":           "NASDAQCOM",    # Nasdaq Composite :contentReference[oaicite:1]{index=1}
    "dow":              "DJIA",         # Dow Jones Industrial Average :contentReference[oaicite:2]{index=2}
    "russell2000":      "RUT",



}


YF_SYMBOLS = {
   
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
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                st.warning(f"yfinance 指标 {name} ({symbol}) 下载为空")
                continue

            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            else:
                s = df["Close"]

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
    # 这里的列名必须和 FRED_SERIES 里的 key 完全一致
    cols = ["sp500", "nasdaq", "dow", "russell2000"]
    available = [c for c in cols if c in df.columns]

    if not available:
        st.warning("指数数据不足")
        return

    data = df[available].dropna(how="all")
    if data.empty:
        st.warning("指数数据为空")
        return

    # 归一化（从 1 开始）
    norm = data / data.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col)

    ax.set_title("US Equity Indices (Normalized)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)



# ================================
# 流动性评分
# ================================
def compute_liquidity_score(df, config=LIQUIDITY_CONFIG, window_days=365):

    valid_df = df.dropna(how="all")
    if valid_df.empty:
        raise ValueError("没有有效数据用于评分")

    end_date = valid_df.index.max()
    start_date = end_date - pd.Timedelta(days=window_days)
    window_df = df.loc[start_date:end_date]

    z_details = []
    total_weight = 0.0
    weighted_z = 0.0

    for col, meta in config.items():
        if col not in window_df.columns:
            st.info(f"[评分提示] 缺少 {col}")
            continue

        series = window_df[col].dropna()
        if len(series) < 30:
            st.info(f"[评分提示] {col} 数据不足（<30）")
            continue

        mean = series.mean()
        std = series.std()

        if std == 0 or np.isnan(std):
            st.info(f"[评分提示] {col} 标准差为 0 或 NaN，跳过")
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
        raise ValueError("没有可用指标计算评分（所有配置的指标都被跳过了）")

    score = 50 - 10 * (weighted_z / total_weight)
    score = max(0, min(100, score))

    if score >= 60:
        label = "流动性偏宽松"
    elif score <= 40:
        label = "流动性偏紧"
    else:
        label = "流动性中性"

    detail_df = pd.DataFrame(z_details).set_index("indicator")

    return score, label, detail_df, (start_date, end_date)

# ================================
# Streamlit 主程序
# ================================
def main():
    st.set_page_config(page_title="USD Liquidity Dashboard", layout="wide")
    st.title("🧊 USD 宏观流动性 Dashboard")

    # ==== 左侧参数 ====
    with st.sidebar:
        st.header("参数设置")
        start_date = st.date_input("开始日期", START_DEFAULT)
        end_date = st.date_input("结束日期", END_DEFAULT)
        window_days = st.slider("评分窗口（天）", 180, 730, 365)

        if start_date >= end_date:
            st.error("开始日期必须早于结束日期")
            return

    st.info("数据正在获取...")
    all_df = build_panel(start_date, end_date)
    
    if all_df.empty:
        st.error("数据获取失败：all_df 为空")
        return
    st.success("数据更新完成")

    st.subheader("最新一行数据")
    st.dataframe(all_df.tail(1))
    


    # =======================
    # 图表区
    # =======================
    st.header("📊 流动性 & 利率")
    col1, col2 = st.columns(2)
    with col1:
        plot_series(
            all_df,
            ["bank_reserves", "fed_balance_sheet"],
            title="Bank Reserves vs Fed Balance Sheet",
            ylabel="Millions",
            rolling=7,
        )
    with col2:
        plot_onrrp_tga(all_df)

    col3, col4 = st.columns(2)
    with col3:
        plot_series(
            all_df,
            ["sofr", "t_bill_1m", "t_bill_3m", "repo_gc"],
            title="SOFR / T-bill / Repo",
            ylabel="Rate (%)",
            rolling=7,
        )
    with col4:
        plot_series(
            all_df,
            ["hy_spread"],
            title="HY Spread",
            ylabel="bps",
            rolling=7,
        )

    col5, col6 = st.columns(2)
    with col5:
        plot_series(
            all_df,
            ["dxy"],
            title="DXY",
            ylabel="Index",
            rolling=7,
        )
    with col6:
        plot_series(
            all_df,
            ["vix"],
            title="VIX",
            ylabel="Index",
            rolling=7,
        )

    st.header("📈 美股主要指数（归一化）")
    plot_equity_indices(all_df)

    # =======================
    # 流动性评分
    # =======================
    st.header("🧠 宏观流动性评分")

    try:
        score, label, detail_df, (s, e) = compute_liquidity_score(
            all_df, LIQUIDITY_CONFIG, window_days
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("流动性评分", f"{score:.1f}")
        with c2:
            st.metric("状态", label)

        st.caption(f"评分区间：{s.date()} → {e.date()}")
        st.dataframe(detail_df)

    except Exception as e:
        st.error(f"无法计算流动性评分：{e}")

if __name__ == "__main__":
    main()








