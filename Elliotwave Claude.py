import os
os.system('pip install yfinance backtesting pandas numpy matplotlib bokeh==3.1.1')

from backtesting import Backtest, Strategy
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ── INDICATORS ───────────────────────────────────────────────
def compute_rsi(series, n=14):
    s     = pd.Series(series)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + gain / (loss + 1e-10))).values

def compute_ema(series, n):
    return pd.Series(series).ewm(span=n, adjust=False).mean().values

def compute_macd(series, fast=12, slow=26, signal=9):
    s      = pd.Series(series)
    fast_e = s.ewm(span=fast,   adjust=False).mean()
    slow_e = s.ewm(span=slow,   adjust=False).mean()
    macd   = fast_e - slow_e
    sig    = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, sig.values

def compute_atr(high, low, close, n=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


# ── STRATEGY ─────────────────────────────────────────────────
class AggressiveCryptoStrategy(Strategy):
    """
    ENTRY — needs 3 of 4 signals:
      1. Golden Cross  : EMA50 crossed above EMA200 in last 3 bars
      2. RSI Cross     : RSI crossed above 50 in last 4 bars
      3. MACD Cross    : MACD crossed above signal line last bar
      4. Breakout      : Close > highest close of last 10 bars

    EXIT:
      - TP  : entry + ATR × 3.0  (3:1 R:R)
      - SL  : entry - ATR × 1.0  (tight)
      - Trailing stop once up 1.5× ATR (managed manually via close)
    """
    ema_fast    = 50
    ema_slow    = 200
    rsi_period  = 14
    rsi_level   = 50
    macd_fast   = 12
    macd_slow   = 26
    macd_signal = 9
    atr_period  = 14
    atr_sl      = 1.0
    atr_tp      = 3.0
    breakout_n  = 10
    min_signals = 3

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low

        self.ema50    = self.I(compute_ema,  c, self.ema_fast)
        self.ema200   = self.I(compute_ema,  c, self.ema_slow)
        self.rsi      = self.I(compute_rsi,  c, self.rsi_period)
        self.atr      = self.I(compute_atr,  h, l, c, self.atr_period)

        macd, sig     = compute_macd(c, self.macd_fast, self.macd_slow, self.macd_signal)
        self.macd     = self.I(lambda: macd)
        self.macd_sig = self.I(lambda: sig)

        # track entry + trailing stop manually
        self._entry   = 0.0
        self._trail   = 0.0   # current trailing stop level

    def next(self):
        c   = self.data.Close
        cur = c[-1]

        if len(c) < self.ema_slow + 5:
            return

        atr = self.atr[-1]
        if atr <= 0:
            return

        # ── SIGNAL 1: Golden Cross ──
        golden_cross = (self.ema50[-1] > self.ema200[-1] and
                        self.ema50[-3] <= self.ema200[-3])

        # ── SIGNAL 2: RSI crossed above 50 ──
        rsi_cross = (self.rsi[-1] > self.rsi_level and
                     self.rsi[-4] <= self.rsi_level)

        # ── SIGNAL 3: MACD bullish crossover ──
        macd_cross = (self.macd[-1] > self.macd_sig[-1] and
                      self.macd[-2] <= self.macd_sig[-2])

        # ── SIGNAL 4: 10-bar price breakout ──
        recent_high = max(c[-self.breakout_n - 1:-1])
        price_break = cur > recent_high

        signals = sum([golden_cross, rsi_cross, macd_cross, price_break])

        # ── ENTRY ──
        if not self.position:
            if signals >= self.min_signals:
                sl = cur - (atr * self.atr_sl)
                tp = cur + (atr * self.atr_tp)
                self._entry = cur
                self._trail = sl       # initialise trailing stop at entry SL
                self.buy(tp=tp, sl=sl)

        # ── TRAILING STOP (managed by closing manually) ──
        # backtesting.py Position has no .sl attribute — we manage it ourselves
        elif self.position:
            gained = cur - self._entry
            if gained >= atr * 1.5:
                new_trail = self._entry + (atr * 0.5)
                if new_trail > self._trail:
                    self._trail = new_trail  # ratchet stop up

            # if price drops below our trailing stop, close manually
            if cur <= self._trail:
                self.position.close()


# ── PAIRS ────────────────────────────────────────────────────
pairs = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
}

START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"


def load_pair(name, ticker):
    print(f"  Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        print(f"  WARNING: No data for {name}, skipping.")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [col.strip().title() for col in df.columns]
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0)
    df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df


# ── RUN ──────────────────────────────────────────────────────
print("=" * 55)
print("  AGGRESSIVE CRYPTO STRATEGY — BTC & ETH")
print("  Signals: Golden Cross + RSI + MACD + Breakout")
print("  R:R = 3:1 | Trailing Stop | Min 3/4 signals")
print("=" * 55)

all_trades   = []
pair_results = {}

for name, ticker in pairs.items():
    df = load_pair(name, ticker)
    if df is None:
        continue
    bt     = Backtest(df, AggressiveCryptoStrategy, cash=10000, commission=0.001)
    stats  = bt.run()
    trades = stats['_trades']
    n      = len(trades)
    pair_results[name] = {
        'trades':     trades,
        'n_trades':   n,
        'win_rate':   stats.get('Win Rate [%]', 0),
        'return_pct': stats.get('Return [%]', 0),
        'max_dd':     stats.get('Max. Drawdown [%]', 0),
        'sharpe':     stats.get('Sharpe Ratio', 0),
        'final_eq':   stats.get('Equity Final [$]', 10000),
    }
    print(f"  {name:<10} → {n:>4} trades | "
          f"WR: {pair_results[name]['win_rate']:.1f}% | "
          f"Return: {pair_results[name]['return_pct']:.1f}% | "
          f"MaxDD: {pair_results[name]['max_dd']:.1f}%")
    if n > 0:
        all_trades.append(trades['ReturnPct'].values)

combined_returns = np.concatenate(all_trades) if all_trades else np.array([])
total_trades     = len(combined_returns)
print(f"\n  TOTAL COMBINED TRADES: {total_trades}")


# ── PERFORMANCE TABLE ─────────────────────────────────────────
print("\n" + "=" * 75)
print(f"  {'Pair':<10} {'Trades':>7} {'Win Rate':>10} {'Return %':>10} {'Max DD %':>10} {'Sharpe':>8} {'Final $':>10}")
print("=" * 75)
for name, r in pair_results.items():
    print(f"  {name:<10} {r['n_trades']:>7} {r['win_rate']:>9.1f}% "
          f"{r['return_pct']:>9.1f}% {r['max_dd']:>9.1f}% "
          f"{r['sharpe']:>8.2f} ${r['final_eq']:>9,.0f}")
print("=" * 75)


# ── MONTE CARLO ───────────────────────────────────────────────
if total_trades > 5:
    TARGET_TRADES = 1000
    SIMULATIONS   = 1000
    STARTING_BAL  = 10000

    print(f"\n  Running {SIMULATIONS} Monte Carlo paths × {TARGET_TRADES} trades...")

    all_paths       = np.zeros((SIMULATIONS, TARGET_TRADES + 1))
    all_paths[:, 0] = STARTING_BAL

    for i in range(SIMULATIONS):
        sim_returns      = np.random.choice(combined_returns, size=TARGET_TRADES, replace=True)
        path             = STARTING_BAL * np.cumprod(1 + sim_returns / 100)
        all_paths[i, 1:] = path

    median_path = np.median(all_paths,  axis=0)
    p5          = np.percentile(all_paths,  5, axis=0)
    p25         = np.percentile(all_paths, 25, axis=0)
    p75         = np.percentile(all_paths, 75, axis=0)
    p95         = np.percentile(all_paths, 95, axis=0)
    final_bals  = all_paths[:, -1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Aggressive Crypto Strategy — BTC & ETH | {total_trades} Real Trades\n"
        f"Golden Cross + RSI Cross + MACD + Breakout | 3:1 R:R | Trailing Stop",
        fontsize=13, fontweight='bold'
    )

    # Top-left: Monte Carlo fan
    ax = axes[0, 0]
    for i in range(0, SIMULATIONS, 5):
        ax.plot(all_paths[i], color='steelblue', alpha=0.03, linewidth=0.8)
    ax.fill_between(range(TARGET_TRADES+1), p5,  p95, color='steelblue', alpha=0.15, label='5–95th pct')
    ax.fill_between(range(TARGET_TRADES+1), p25, p75, color='steelblue', alpha=0.25, label='25–75th pct')
    ax.plot(median_path, color='red',    linewidth=2.5, label=f'Median: ${median_path[-1]:,.0f}')
    ax.plot(p5,          color='orange', linewidth=1.5, linestyle='--', label=f'5th pct: ${p5[-1]:,.0f}')
    ax.plot(p95,         color='green',  linewidth=1.5, linestyle='--', label=f'95th pct: ${p95[-1]:,.0f}')
    ax.axhline(STARTING_BAL, color='black', linewidth=1, linestyle=':', alpha=0.5, label='$10,000 Start')
    ax.set_title("Monte Carlo Simulation (1000 Paths)")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Account Balance ($)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Top-right: Bell curve histogram
    ax = axes[0, 1]
    ax.hist(final_bals, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(final_bals),         color='red',    linewidth=2,
               label=f'Median: ${np.median(final_bals):,.0f}')
    ax.axvline(np.percentile(final_bals,  5), color='orange', linewidth=1.5, linestyle='--',
               label=f'5th: ${np.percentile(final_bals, 5):,.0f}')
    ax.axvline(np.percentile(final_bals, 95), color='green',  linewidth=1.5, linestyle='--',
               label=f'95th: ${np.percentile(final_bals, 95):,.0f}')
    ax.axvline(STARTING_BAL, color='black', linewidth=1, linestyle=':', label='Start: $10,000')
    ax.set_title("Distribution of Final Balances")
    ax.set_xlabel("Final Account Balance ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Bottom-left: Win rate bars
    ax        = axes[1, 0]
    names     = list(pair_results.keys())
    win_rates = [pair_results[n]['win_rate'] for n in names]
    colors    = ['gold', 'mediumslateblue']
    bars = ax.bar(names, win_rates, color=colors, edgecolor='white', alpha=0.85)
    ax.axhline(50, color='black', linewidth=1, linestyle='--', alpha=0.5, label='50% line')
    ax.set_title("Win Rate by Asset")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Bottom-right: Return % bars
    ax           = axes[1, 1]
    returns_list = [pair_results[n]['return_pct'] for n in names]
    bars2 = ax.bar(names, returns_list, color=colors, edgecolor='white', alpha=0.85)
    ax.axhline(0, color='black', linewidth=1, alpha=0.4)
    ax.set_title("Total Return % by Asset")
    ax.set_ylabel("Return (%)")
    for bar, val in zip(bars2, returns_list):
        ypos = bar.get_height()+1 if val >= 0 else bar.get_height()-3
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ── FINAL SUMMARY ─────────────────────────────────────────
    prob_profit = np.mean(final_bals > STARTING_BAL)       * 100
    prob_double = np.mean(final_bals > STARTING_BAL * 2)   * 100
    prob_ruin   = np.mean(final_bals < STARTING_BAL * 0.5) * 100

    print("\n" + "=" * 55)
    print("  MONTE CARLO SUMMARY (1000 trades, 1000 simulations)")
    print("=" * 55)
    print(f"  Starting Balance:      ${STARTING_BAL:>10,.2f}")
    print(f"  Median Final Balance:  ${np.median(final_bals):>10,.2f}")
    print(f"  Best Case  (95th pct): ${np.percentile(final_bals, 95):>10,.2f}")
    print(f"  Worst Case  (5th pct): ${np.percentile(final_bals,  5):>10,.2f}")
    print(f"  Prob. of Profit:       {prob_profit:>9.1f}%")
    print(f"  Prob. of Doubling:     {prob_double:>9.1f}%")
    print(f"  Prob. of 50% Ruin:     {prob_ruin:>9.1f}%")
    print("=" * 55)

else:
    print(f"\nOnly {total_trades} trades found. Try widening the date range.")
