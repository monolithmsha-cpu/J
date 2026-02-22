# 1. SETUP: Installs the specific versions needed to show the chart
import os
os.system('pip install yfinance backtesting pandas numpy bokeh==3.1.1')

from backtesting import Backtest, Strategy
import yfinance as yf
import pandas as pd
import numpy as np

class ElliottFibStrategy(Strategy):
    # These are your "Trajectory Goals"
    fib_3_target = 1.618  # Target for a standard Wave 3
    
    def init(self):
        # We define what price data to use (Close, High, Low)
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low

    def next(self):
        # 2. MEASUREMENT: Look at the last 30 bars to find Wave 1
        recent_low = min(self.low[-30:-5])
        recent_high = max(self.high[-30:-5])
        current_pullback = min(self.low[-5:]) # This is our Wave 2
        
        wave1_size = recent_high - recent_low
        
        # 3. TRAJECTORY: Calculate where Wave 3 should end
        target_1618 = current_pullback + (wave1_size * self.fib_3_target)

        # 4. TRADING: Buy if we break the Wave 1 high
        if not self.position and self.close[-1] > recent_high:
            # TP = Take Profit (Sell at the Fib target)
            # SL = Stop Loss (Sell if Wave 2 is broken)
            self.buy(tp=target_1618, sl=current_pullback)

# 5. DATA: Choose your asset. Crypto = "BTC-USD", Forex = "EURUSD=X"
# 'interval' is set to 1 hour ("1h") for better wave detail
data = yf.download("BTC-USD", start="2024-01-01", interval="1h")

# 6. RESULTS: Run the simulation with $10,000
bt = Backtest(data, ElliottFibStrategy, cash=10000, commission=0.001)
stats = bt.run()

print(stats) # Shows the math results
bt.plot()    # Generates the visual chart
