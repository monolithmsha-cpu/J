import os
os.system('pip install yfinance backtesting pandas numpy bokeh==3.1.1')

from backtesting import Backtest, Strategy
import yfinance as yf
import pandas as pd

class Elliott100TradesStrategy(Strategy):
    # Reduced lookback to 20 to find more frequent patterns
    lookback = 20 
    
    def init(self):
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low

    def next(self):
        if len(self.data) < self.lookback:
            return

        # 1. Identify Swing Points
        recent_low = min(self.low[-self.lookback:-5])
        recent_high = max(self.high[-self.lookback:-5])
        
        # 2. Entry Signal (Breakout of Wave 1 High)
        if not self.position and self.close[-1] > recent_high:
            entry_price = self.close[-1]
            stop_loss = recent_low
            
            # 3. THE 2:1 CALCULATION
            # Risk = Entry - Stop Loss
            risk_amount = entry_price - stop_loss
            
            # Reward = Risk * 2
            take_profit = entry_price + (risk_amount * 2.0)
            
            # Execute trade only if take_profit is actually above entry
            if take_profit > entry_price:
                self.buy(tp=take_profit, sl=stop_loss)

# --- THE DATA ENGINE ---
# Downloading maximum available 1h data to aim for 100 trades
raw_data = yf.download("BTC-USD", start="2024-03-01", interval="1h")

# Flatten MultiIndex (Fixes your previous error)
data = raw_data.copy()
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Run Backtest
bt = Backtest(data, Elliott100TradesStrategy, cash=10000, commission=0.001)
stats = bt.run()

# CHECK THIS IN THE OUTPUT:
print(f"Total Trades: {stats['# Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']}%")
print(f"2:1 RR check (Profit Factor): {stats['Profit Factor']}")
print(stats)
bt.plot()
