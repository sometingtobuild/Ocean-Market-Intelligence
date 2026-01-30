import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- 🌊 OCEAN PROTOCOL CONFIGURATION 🌊 ---
if os.path.exists("/data/outputs"):
    OUTPUT_DIR = "/data/outputs"
    print("🌊 Running in Ocean Protocol Environment")
else:
    OUTPUT_DIR = "results"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print("💻 Running Locally")

REPORT_FILE = os.path.join(OUTPUT_DIR, "market_report.json")
CHART_FILE = os.path.join(OUTPUT_DIR, "ai_prediction_chart.png")

# --- 1. REAL DATA FETCHING ---
def fetch_real_data(coin_id="bitcoin", days=60):
    """Fetches real market data from CoinGecko."""
    print(f"🌍 Connecting to CoinGecko API for {coin_id}...")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        # --- THE FIX: Clean data immediately ---
        original_len = len(df)
        df = df.dropna()
        print(f"✅ Data Cleaned: Kept {len(df)}/{original_len} valid days.")
        
        return df
    except Exception as e:
        print(f"❌ API Error: {e}")
        return generate_synthetic_backup()

def generate_synthetic_backup(days=60):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    prices = [45000 + np.random.normal(0, 500) * i for i in range(days)]
    return pd.DataFrame({'price': prices}, index=dates)

# --- 2. MARKET INTELLIGENCE ---
def calculate_indicators(df):
    print("🧮 Calculating Market Intelligence Indicators...")
    df['returns'] = df['price'].pct_change()
    volatility = df['returns'].std() * 100
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df, volatility

# --- 3. REAL AI PREDICTION ---
def run_ai_prediction(df):
    print("🤖 Training AI Model (Linear Regression)...")
    
    # Since df is already cleaned, we can use it directly
    df['days_from_start'] = (df.index - df.index[0]).days
    
    X = df[['days_from_start']].values
    y = df['price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Tomorrow
    last_day_idx = X[-1][0]
    next_day_idx = last_day_idx + 1
    prediction = model.predict([[next_day_idx]])[0]
    
    return model, prediction, X

# --- 4. VISUALIZATION & REPORTING ---
def save_results(df, model, X, prediction, volatility):
    print("🎨 Generating Intelligence Report...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Actual Prices
    plt.scatter(df.index, df['price'], color='gray', alpha=0.5, label='Actual Prices (CoinGecko)')
    
    # Plot 2: AI Trend Line (Now sizes match perfectly!)
    plt.plot(df.index, model.predict(X), color='red', linewidth=2, label='AI Trend Line')
    
    # Plot 3: Prediction
    last_date = df.index[-1]
    next_date = last_date + timedelta(days=1)
    plt.scatter([next_date], [prediction], color='green', s=150, zorder=5, label=f'Forecast: ${prediction:,.0f}')
    
    plt.title(f'AI Market Intelligence: BTC Prediction (Vol: {volatility:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(CHART_FILE)
    print(f"🖼️ Chart saved to: {CHART_FILE}")
    
    # JSON Report
    rsi_val = df['RSI'].iloc[-1]
    # Handle NaN RSI if dataset is small
    rsi_str = f"{rsi_val:.2f}" if not pd.isna(rsi_val) else "N/A"
    
    report = {
        "status": "success",
        "market_metrics": {
            "volatility_index": f"{volatility:.2f}%",
            "RSI_14": rsi_str
        },
        "ai_prediction": {
            "forecast_price": f"${prediction:,.2f}"
        }
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"💾 Data report saved to: {REPORT_FILE}")

def main():
    print("🚀 Starting Hybrid AI Engine...")
    df = fetch_real_data('bitcoin', days=60)
    
    # Safety Check: Ensure we have data
    if df is not None and not df.empty:
        df, vol = calculate_indicators(df)
        model, pred, X = run_ai_prediction(df)
        save_results(df, model, X, pred, vol)
        print("✅ Job Complete.")
    else:
        print("❌ Error: No data available to process.")

if __name__ == "__main__":
    main()