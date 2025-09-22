#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import datetime
from typing import List, Dict, Optional

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from bs4 import BeautifulSoup

# ─── Logging & Config ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
REQUEST_TIMEOUT = 12
SCRAPER_SLEEP = 0.5

EXCLUDED_SECTORS = {
    "Alcoholic Beverages",
    "Breweries & Distilleries",
    "Media",
    "Media & Entertainment",
    "Banking",
    "Finance",
    "Financial Services",
    "NBFC"
}

# ─── Telegram Alerts ────────────────────────────────────
def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.warning("Telegram credentials not set.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logging.warning(f"Telegram send error: {e}")

# ─── Market Regime ──────────────────────────────────────
def classify_market_regime() -> str:
    try:
        nifty = yf.download("^NSEI", period="2y", interval="1d", progress=False, auto_adjust=False)["Close"].dropna()
        vix = yf.download("^INDIAVIX", period="6mo", interval="1d", progress=False, auto_adjust=False)["Close"].dropna()
        if len(nifty) < 220 or len(vix) < 10:
            return "Neutral"
        sma_200 = float(nifty.rolling(200).mean().iloc[-1])
        price = float(nifty.iloc[-1])
        vix_last = float(vix.iloc[-1])
        if price > sma_200 and vix_last < 15:
            return "Bullish"
        if price < sma_200 or vix_last > 20:
            return "Bearish"
        return "Neutral"
    except Exception as e:
        logging.warning(f"Regime classification error: {e}")
        return "Neutral"

# ─── Dynamic Stock Universe ─────────────────────────────
def fetch_nifty500_symbols(max_retries: int = 3) -> List[str]:
    wiki_url = "https://en.wikipedia.org/wiki/NIFTY_500"
    nse_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(wiki_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            for df in tables:
                if "Symbol" in df.columns:
                    symbols = df["Symbol"].dropna().astype(str).str.strip().unique().tolist()
                    logging.info(f"Fetched {len(symbols)} NIFTY 500 symbols from Wikipedia.")
                    return symbols
        except Exception as e:
            logging.warning(f"NIFTY 500 fetch attempt {attempt} from Wikipedia failed: {e}")
            time.sleep(1.5)

    # Fallback to NSE CSV
    try:
        df = pd.read_csv(nse_url)
        if "Symbol" in df.columns:
            symbols = df["Symbol"].dropna().astype(str).str.strip().unique().tolist()
            logging.info(f"Fetched {len(symbols)} NIFTY 500 symbols from NSE CSV.")
            return symbols
    except Exception as e:
        logging.error(f"Fallback to NSE CSV failed: {e}")

    raise RuntimeError("Failed to fetch NIFTY 500 symbols from both Wikipedia and NSE.")

# ─── Screener Scraping ──────────────────────────────────
def _extract_text(soup: BeautifulSoup, label: str) -> Optional[str]:
    tag = soup.find(string=lambda t: isinstance(t, str) and t.strip() == label)
    if tag:
        nxt = tag.find_next()
        if nxt and hasattr(nxt, "get_text"):
            return nxt.get_text(strip=True)
    return None

def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        clean = val.replace("Cr", "").replace("%", "").replace(",", "").strip()
        return float(clean)
    except Exception:
        return None

def get_fundamentals(symbol: str) -> Optional[Dict[str, float]]:
    url = f"https://www.screener.in/company/{symbol}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        market_cap = _to_float(_extract_text(soup, "Market Cap"))
        roce = _to_float(_extract_text(soup, "ROCE"))
        d2e = _to_float(_extract_text(soup, "Debt to equity"))
        sales_g = _to_float(_extract_text(soup, "Sales growth"))
        profit_g = _to_float(_extract_text(soup, "Profit growth"))
        if None in (market_cap, roce, d2e, sales_g, profit_g):
            return None
        return {
            "market_cap": market_cap,
            "roce": roce,
            "debt_to_equity": d2e,
            "sales_growth_5y": sales_g,
            "profit_growth_5y": profit_g
        }
    except Exception:
        return None

def get_sector(symbol: str) -> Optional[str]:
    url = f"https://www.screener.in/company/{symbol}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        sector_label = soup.find("span", string=lambda t: isinstance(t, str) and t.strip() == "Sector")
        if sector_label:
            link = sector_label.find_next("a")
            if link:
                return link.get_text(strip=True)
        return None
    except Exception:
        return None

# ─── Filters ────────────────────────────────────────────
def passes_fundamental_filters(f: Dict[str, float]) -> bool:
    return (
        f["market_cap"] > 500 and
        f["debt_to_equity"] < 0.2 and
        f["roce"] > 20 and
        f["sales_growth_5y"] > 10 and
        f["profit_growth_5y"] > 15
    )

def passes_technical_filters(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df) < 210:
        return False
    close = df["Close"]
    price = close.iloc[-1]
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_100 = close.rolling(100).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]
    return price > sma_20 and price > sma_50 and price > sma_100 and price > sma_200

# ─── Signals ────────────────────────────────────────────
def detect_pullback(df: pd.DataFrame) -> bool:
    close = df["Close"]
    ema_20 = close.ewm(span=20).mean()
    sma_200 = close.rolling(200).mean()
    return close.iloc[-1] > sma_200.iloc[-1] and close.iloc[-2] < ema_20.iloc[-2] and close.iloc[-1] > ema_20.iloc[-1]

def detect_breakout(df: pd.DataFrame) -> bool:
    close = df["Close"]
    high = df["High"]
    resistance = close.iloc[-10:-3].max()
    return close.iloc[-1] > resistance and high.iloc[-1] > resistance

# ─── Risk & Journaling ──────────────────────────────────
def calculate_position_size(capital: float, risk_per_trade: float, entry: float, stop: float) -> int:
    stop_points = abs(entry - stop)
    if stop_points <= 0:
        return 0
    return int((capital * risk_per_trade) // stop_points)

def save_chart(df: pd.DataFrame, symbol: str) -> str:
    plt.figure(figsize=(10, 4))
    plt.plot(df["Close"], label="Close", color="#1f77b4")
    plt.plot(df["Close"].rolling(20).mean(), label="SMA20", color="#ff7f0e", alpha=0.8)
    plt.plot(df["Close"].rolling(50).mean(), label="SMA50", color="#2ca02c", alpha=0.8)
    plt.plot(df["Close"].rolling(200).mean(), label="SMA200", color="#d62728", alpha=0.8)
    plt.title(f"{symbol} Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    filename = f"{symbol}_chart.png"
    plt.savefig(filename)
    plt.close()
    return filename

def log_trade(stock: str, signal_type: str, entry_price: float, stop_loss: float, target_price: float, quantity: int):
    log = {
        'timestamp': datetime.datetime.now(),
        'stock': stock,
        'signal': signal_type,
        'entry': entry_price,
        'stop_loss': stop_loss,
        'target': target_price,
        'quantity': quantity
    }
    df = pd.DataFrame([log])
    df.to_csv('trade_log.csv', mode='a', header=not os.path.exists('trade_log.csv'), index=False)

# ─── Main Bot Logic ──────────────────────────────────────
def run_bot():
    logging.info("Running SAMS bot...")

    regime = classify_market_regime()
    send_telegram_message(f"📊 Market Regime: {regime}")

    if regime == "Bearish":
        send_telegram_message("🚫 Market is bearish. No trades today.")
        return

    symbols = fetch_nifty500_symbols()
    qualified_fundamentals = []

    for symbol in symbols:
        sector = get_sector(symbol)
        if sector and sector in EXCLUDED_SECTORS:
            continue
        data = get_fundamentals(symbol)
        time.sleep(SCRAPER_SLEEP)
        if data and passes_fundamental_filters(data):
            qualified_fundamentals.append(symbol)

    qualified_stocks = []
    for stock in qualified_fundamentals:
        try:
            data = yf.download(f"{stock}.NS", period='6mo', interval='1d', progress=False, auto_adjust=False)
            if passes_technical_filters(data):
                qualified_stocks.append(stock)
        except Exception as e:
            logging.warning(f"Error fetching data for {stock}: {e}")

    capital = 100000
    risk_per_trade = 0.02
    signaled = []

    for stock in qualified_stocks:
        try:
            data = yf.download(f"{stock}.NS", period='6mo', interval='1d', progress=False, auto_adjust=False)
            if data is None or data.empty:
                continue
            entry = data['Close'].iloc[-1]
            stop = entry * 0.96
            target = entry * 1.06
            qty = calculate_position_size(capital, risk_per_trade, entry, stop)

            if detect_pullback(data):
                log_trade(stock, "Pullback", entry, stop, target, qty)
                send_telegram_message(f"📥 Pullback in {stock}: Buy {qty} @ ₹{entry:.2f}, SL ₹{stop:.2f}, Target ₹{target:.2f}")
                signaled.append(stock)
            elif detect_breakout(data):
                log_trade(stock, "Breakout", entry, stop, target, qty)
                send_telegram_message(f"🚀 Breakout in {stock}: Buy {qty} @ ₹{entry:.2f}, SL ₹{stop:.2f}, Target ₹{target:.2f}")
                signaled.append(stock)

            save_chart(data, stock)

        except Exception as e:
            logging.warning(f"Signal error for {stock}: {e}")

    if signaled:
        send_telegram_message(f"✅ Signals today: {', '.join(signaled)}")
    else:
        send_telegram_message("📭 No actionable signals today.")

# ─── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    run_bot()