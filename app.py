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
# 使用通用样式，无需安装 seaborn
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

START_DEFAULT = date(2018, 1, 1)
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
