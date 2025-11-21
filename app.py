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
    画一张组合图：
    - 上半部分：指标(左轴) vs Russell 2000(右轴)
    - 下半部分：90天滚动相关性
    """
    if indicator_col not in df.columns or target_col not in df.columns:
        st.warning(f"缺少数据：{indicator_col} 或 {target_col}")
        return

    # 准备数据
    plot_df = df[[indicator_col, target_col]].dropna()
    if plot_df.empty:
        return

    # 计算滚动相关性
    rolling_corr = plot_df[indicator_col].rolling(window=window).corr(plot_df[target_col])

    # 创建画布：2个子图，高度比例 2:1
    fig, (ax1, ax_corr) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
    
    # --- 上半部分：双轴走势对比 ---
    color_ind = 'tab:blue'
    color_target = 'tab:gray'
    
    # 左轴：宏观指标
    ax1.plot(plot_df.index, plot_df[indicator_col], color=color_ind, label=indicator_col, linewidth=1.5)
    ax1.set_ylabel(indicator_col, color=color_ind, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_ind)
    
    # 右轴：Russell 2000
    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df[target_col], color=color_target, label=target_col, linestyle='--', alpha=0.6, linewidth=1)
    ax2.set_ylabel(target_col, color=color_target, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_target)
    
    ax1.set_title(f"{title_prefix} {indicator_col} vs {target_col}", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- 下半部分：滚动相关性 ---
    # 根据正负值填充颜色
    corr_vals = rolling_corr.dropna()
    if not corr_vals.empty:
        ax_corr.plot(corr_vals.index, corr_vals, color='black', linewidth=1)
        ax_corr.fill_between(corr_vals.index, 0, corr_vals, where=(corr_vals > 0), color='green', alpha=0.3, label='Positive Corr')
        ax_corr.fill_between(corr_vals.index, 0, corr_vals, where=(corr_vals < 0), color='red', alpha=0.3, label='Negative Corr')
    
    ax_corr.set_ylabel(f"{window}d Rolling Corr", fontsize=10)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_corr.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
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
    # ... (在 st.dataframe(all_df.tail(1)) 之后)

    # =======================
    # 图表区：深度相关性分析
    # =======================
    st.header("🔬 深度透视：宏观因子 vs Russell 2000")
    st.caption("上图：双轴价格走势（蓝色=宏观指标，灰色虚线=Russell 2000） | 下图：90天滚动相关性（红色=负相关，绿色=正相关）")

    # 1. 核心流动性：准备金
    st.subheader("1. 核心流动性")
    col1, col2 = st.columns(2)
    with col1:
        plot_overlay_with_correlation(all_df, "bank_reserves", title_prefix="[央行水龙头]")
    with col2:
        plot_overlay_with_correlation(all_df, "fed_balance_sheet", title_prefix="[美联储资产表]")

    # 2. 抽水效应：TGA & ON RRP
    st.subheader("2. 抽水与缓冲")
    col3, col4 = st.columns(2)
    with col3:
        plot_overlay_with_correlation(all_df, "tga", title_prefix="[财政部账户]")
    with col4:
        plot_overlay_with_correlation(all_df, "on_rrp", title_prefix="[逆回购工具]")
    
    # 3. 利率与避险
    st.subheader("3. 利率痛点 & 避险情绪")
    col5, col6 = st.columns(2)
    with col5:
        # 这里特意选了 T-Bill 3M，因为你刚才发现它相关性最高
        plot_overlay_with_correlation(all_df, "t_bill_3m", title_prefix="[短期无风险利率]")
    with col6:
        plot_overlay_with_correlation(all_df, "dxy", title_prefix="[美元指数]")

    col7, col8 = st.columns(2)
    with col7:
        plot_overlay_with_correlation(all_df, "hy_spread", title_prefix="[信用利差]")
    with col8:
        plot_overlay_with_correlation(all_df, "vix", title_prefix="[恐慌指数]")

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

# ... (接在 main 函数后面)
    
    # =======================
    # 相关性分析板块
    # =======================
    st.markdown("---")
    st.header("🔗 相关性分析 (vs Russell 2000)")
    
    # 确保 russell2000 存在且有数据
    if "russell2000" in all_df.columns:
        try:
            fig_static, fig_rolling = plot_correlation_analysis(all_df, target_col="russell2000")
            
            col_corr1, col_corr2 = st.columns(2)
            
            with col_corr1:
                st.subheader("全时段相关性排行")
                st.pyplot(fig_static)
                st.caption("正相关(绿)表示同涨同跌，负相关(红)表示走势相反。")
                
            with col_corr2:
                st.subheader("滚动相关性 (90天)")
                st.pyplot(fig_rolling)
                st.caption("观察宏观因子与股市的关系是否发生'机制转换' (Regime Shift)。")
                
        except Exception as e:
            st.error(f"生成相关性图表出错: {e}")
    else:
        st.info("缺少 Russell 2000 数据，跳过相关性分析。")

if __name__ == "__main__":
    main()

















