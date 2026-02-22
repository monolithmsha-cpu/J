import os
os.system('pip install yfinance backtesting pandas')

from backtesting import Backtest, Strategy
import yfinance as yf
import pandas as pd

class ElliottWave3Strategy(Strategy):
    def init(self):
        self.close = self.data.Close

    def next(self):
        # Basic Logic: Buy if price breaks the 20-day high (Proxy for Wave 3 breakout)
        if not self.position and self.close[-1] > max(self.close[-20:-1]):
            self.buy(sl=self.close[-1] * 0.95, tp=self.close[-1] * 1.15)

# 1. Pull data for Bitcoin (or change to "AAPL", "TSLA", etc.)
data = yf.download("BTC-USD", start="2023-01-01", end="2026-01-01")

# 2. Run the Backtest
bt = Backtest(data, ElliottWave3Strategy, cash=10000, commission=.002)
stats = bt.run()

# 3. Print the results
print(stats)
