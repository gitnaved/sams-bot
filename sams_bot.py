# ─── Imports ─────────────────────────────────────────────
import requests
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ─── Telegram Alert Function ─────────────────────────────
def send_telegram_message(message):
    bot_token = '7827996384:AAF8DvSLjHM78Kyhb4YjRDGt-pm5twW27jI'
    chat_id = '5831499682'
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}
    requests.post(url, data=payload)

# ─── Market Regime Engine ────────────────────────────────
def classify_market_regime(nifty, sma_200, vix, fii_net, dii_net):
    regime = "Neutral"
    if nifty > sma_200 and vix < 15 and fii_net > 0 and dii_net > 0:
        regime = "Bullish"
    elif nifty < sma_200 or vix > 20 or (fii_net < 0 and dii_net < 0):
        regime = "Bearish"
    return regime

# ─── Dynamic Stock List ──────────────────────────────────
def fetch_nifty500_symbols():
    url = "https://en.wikipedia.org/wiki/NIFTY_500"
    tables = pd.read_html(url)
    df = tables[1]
    symbols = df['Symbol'].dropna().unique().tolist()
    return symbols

# ─── Screener Scraper ────────────────────────────────────
def get_fundamentals(symbol):
    url = f"https://www.screener.in/company/{symbol}/"
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')

        def extract(label):
            tag = soup.find(text=label)
            return tag.find_next().text if tag else None

        def clean(val):
            return float(val.replace("Cr", "").replace("%", "").replace(",", "").strip())

        market_cap = clean(extract("Market Cap"))
        roce = clean(extract("ROCE"))
        debt_to_equity = clean(extract("Debt to equity"))
        sales_growth = clean(extract("Sales growth"))
        profit_growth = clean(extract("Profit growth"))

        return {
            "market_cap": market_cap,
            "roce": roce,
            "debt_to_equity": debt_to_equity,
            "sales_growth_5y": sales_growth,
            "profit_growth_5y": profit_growth
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_sector(symbol):
    url = f"https://www.screener.in/company/{symbol}/"
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        tag = soup.find("span", text="Sector")
        if tag:
            sector = tag.find_next("a").text.strip()
            return sector
    except Exception as e:
        print(f"Sector fetch error for {symbol}: {e}")
    return None

# ─── Sector Filter ───────────────────────────────────────
EXCLUDED_SECTORS = ["Alcoholic Beverages", "Media", "Banking", "Finance"]

# ─── Fundamental Filter ──────────────────────────────────
def passes_fundamental_filters(stock):
    return (
        stock['market_cap'] > 500 and
        stock['debt_to_equity'] < 0.2 and
        stock['roce'] > 20 and
        stock['sales_growth_5y'] > 10 and
        stock['profit_growth_5y'] > 15
    )

# ─── Technical Filter ────────────────────────────────────
def passes_technical_filters(stock_data):
    close = stock_data['Close']
    price = close.iloc[-1]
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_100 = close.rolling(100).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]
    return price > sma_20 and price > sma_50 and price > sma_100 and price > sma_200

# ─── Entry Signal Generator ──────────────────────────────
def detect_pullback(stock_data):
    close = stock_data['Close']
    ema_20 = close.ewm(span=20).mean()
    sma_200 = close.rolling(200).mean()
    uptrend = close.iloc[-1] > sma_200.iloc[-1]
    pullback = close.iloc[-2] < ema_20.iloc[-2] and close.iloc[-1] > ema_20.iloc[-1]
    return uptrend and pullback

def detect_breakout(stock_data):
    close = stock_data['Close']
    high = stock_data['High']
    resistance = max(close[-10:-3])
    breakout = close.iloc[-1] > resistance and high.iloc[-1] > resistance
    return breakout

# ─── Risk Manager ────────────────────────────────────────
def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price):
    risk_amount = capital * risk_per_trade
    stop_loss_points = abs(entry_price - stop_loss_price)
    quantity = int(risk_amount / stop_loss_points)
    return quantity

# ─── Chart Snapshot ──────────────────────────────────────
def save_chart(stock_data, stock_symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.title(f"{stock_symbol} Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    filename = f"{stock_symbol}_chart.png"
    plt.savefig(filename)
    plt.close()
    return filename

# ─── Journaling Module ───────────────────────────────────
def log_trade(stock, signal_type, entry_price, stop_loss, target_price, quantity):
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
    df.to_csv('trade_log.csv', mode='a', header=False, index=False)

# ─── Main Bot Logic ──────────────────────────────────────
def run_bot():
    print(f"[{datetime.datetime.now()}] Running SAMS bot...")

    # ── Sample Market Regime Inputs ──
    nifty = 24741
    sma_200 = 24000
    vix = 9.88
    fii_net = -1304.91
    dii_net = 1821.23

    regime = classify_market_regime(nifty, sma_200, vix, fii_net, dii_net)
    send_telegram_message(f"📊 Market Regime: {regime}")

    if regime == "Bearish":
        send_telegram_message("🚫 Market is bearish. No trades today.")
        return

    # ── Fetch Dynamic Stock List ──
    symbols = fetch_nifty500_symbols()
    qualified_fundamentals = []

    for symbol in symbols:
        sector = get_sector(symbol)
        if sector and sector in EXCLUDED_SECTORS:
            continue

        data = get_fundamentals(symbol)
        if data and passes_fundamental_filters(data):
            qualified_fundamentals.append(symbol)

    # ── Apply Technical Filters ──
    qualified_stocks = []
    for stock in qualified_fundamentals:
        try:
            data = yf.download(f"{stock}.NS", period='6mo', interval='1d')
            if passes_technical_filters(data):
                qualified_stocks.append(stock)
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")

    # ── Entry Signal Detection ──
    capital = 100000
    risk_per_trade = 0.02

    for stock in qualified_stocks:
        try:
            data = yf.download(f"{stock}.NS", period='6mo', interval='1d')
            entry = data['Close'].iloc[-1]
            stop = entry * 0.96
            target = entry * 1.06
            qty = calculate_position_size(capital, risk_per_trade, entry, stop)

            if detect_pullback(data):
                log_trade(stock, "Pullback", entry, stop, target, qty)
                send_telegram_message(f"📥 Pullback in {stock}: Buy {qty} @ ₹{entry:.2f}, SL ₹{stop:.2f}, Target ₹{target:.2f}")
            elif detect_breakout(data):
                log_trade(stock, "Breakout", entry, stop, target, qty)
                send_telegram_message(f"🚀 Breakout in {stock}: Buy {qty} @ ₹{entry:.2f}, SL ₹{stop:.2f}, Target ₹{target:.2f}")

            save_chart(data, stock)

        except Exception as e:
            print(f"Signal error for {stock}: {e}")

    # ── Summary Alert ──
    if qualified_stocks:
        send_telegram_message(f"✅ Stocks passing SAMS filters: {', '.join(qualified_stocks)}")
    else:
        send_telegram_message("📭 No stocks passed SAMS filters today.")

# ─── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    run_bot()

