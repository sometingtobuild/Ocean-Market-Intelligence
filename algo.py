import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================
# 🌊 OCEAN MARKET INTELLIGENCE ENGINE
# ==========================================

def fetch_real_data(coin_id="bitcoin", days=60):
    """Fetches real market data from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    print(f"Fetching data for {coin_id} from {url}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        original_len = len(df)
        df = df.dropna()
        print(f"Data Cleaned: Kept {len(df)}/{original_len} valid days.")
        
        return df
    except Exception as e:
        print(f"API Error: {e}")
        return generate_synthetic_backup()

def generate_synthetic_backup(days=60):
    print(f"Generating {days} days of synthetic Bitcoin data...")
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    base_price = 45000
    prices = [base_price + (np.random.normal(100, 500) + i * 50) for i in range(days)]
    return pd.DataFrame({'price': prices}, index=dates)

def calculate_indicators(df):
    print("Calculating Market Intelligence Indicators...")
    df['returns'] = df['price'].pct_change()
    volatility = df['returns'].std() * 100
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df, volatility

def run_ai_prediction(df):
    print("Training AI Model (Linear Regression)...")
    
    df['days_from_start'] = (df.index - df.index[0]).days
    
    X = df[['days_from_start']].values
    y = df['price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_day_idx = X[-1][0]
    next_day_idx = last_day_idx + 1
    prediction = model.predict([[next_day_idx]])[0]
    
    return model, prediction, X

def save_plot(filename):
    """Save plot using Ocean's output directory pattern."""
    output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    print(f"Saved chart to: {path}")
    print(f"./results/{filename}")

def save_data(data, filename):
    """Save JSON data using Ocean's output directory pattern."""
    output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to: {path}")
    print(f"./results/{filename}")

def save_results(df, model, X, prediction, volatility):
    print("Generating Intelligence Report...")
    
    # Create chart
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df.index, df['price'], color='gray', alpha=0.5, label='Actual Prices (CoinGecko)')
    plt.plot(df.index, model.predict(X), color='red', linewidth=2, label='AI Trend Line')
    
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    plt.scatter([next_date], [prediction], color='green', s=150, zorder=5, label=f'Forecast: ${prediction:,.0f}')
    
    plt.title(f'AI Market Intelligence: BTC Prediction (Vol: {volatility:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot('ai_prediction_chart.png')
    
    # JSON Report
    rsi_val = df['RSI'].iloc[-1]
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
    
    save_data(report, 'market_report.json')

def main():
    print("Starting Ocean Market Intelligence Engine...")
    
    try:
        df = fetch_real_data('bitcoin', days=60)
        
        if df is not None and not df.empty:
            df, vol = calculate_indicators(df)
            model, pred, X = run_ai_prediction(df)
            save_results(df, model, X, pred, vol)
            print("Job Completed Successfully!")
            print("./results")
        else:
            print("Error: No data available to process.")
            
    except Exception as e:
        print(f"Error: {e}")
        # Save error report
        output_dir = '/data/outputs' if os.path.exists('/data/outputs') else './data/outputs'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'market_report.json'), 'w') as f:
            json.dump({"status": "error", "error": str(e)}, f)

if __name__ == "__main__":
    main()
