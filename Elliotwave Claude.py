import os
os.system('pip install yfinance backtesting pandas numpy matplotlib bokeh==3.1.1')

from backtesting import Backtest, Strategy
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# --- STRATEGY ---
class HighVolumeWaveStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        close = self.data.Close
        if len(close) < 4:
            return
        if not self.position and close[-1] > close[-4]:
            entry_price = close[-1]
            stop_loss   = entry_price * 0.99
            take_profit = entry_price * 1.02
            self.buy(tp=take_profit, sl=stop_loss)


# --- 1. FAMOUS FOREX PAIRS ---
pairs = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

# Go back far enough to accumulate 1000+ trades across all pairs
START_DATE = "2000-01-01"
END_DATE   = "2025-01-01"


def load_pair(name, ticker):
    """Download and clean one forex pair."""
    print(f"  Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)
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


def run_backtest(name, df):
    """Run the backtest on a single pair and return stats + trades."""
    bt = Backtest(df, HighVolumeWaveStrategy, cash=10000, commission=0.0002)
    stats = bt.run()
    trades = stats['_trades']
    return stats, trades


# --- 2. RUN ALL PAIRS ---
print("=" * 55)
print("  DOWNLOADING DATA & RUNNING BACKTESTS")
print("=" * 55)

all_trades    = []
pair_results  = {}

for name, ticker in pairs.items():
    df = load_pair(name, ticker)
    if df is None:
        continue
    stats, trades = run_backtest(name, df)
    n = len(trades)
    pair_results[name] = {
        'stats':      stats,
        'trades':     trades,
        'n_trades':   n,
        'win_rate':   stats.get('Win Rate [%]', 0),
        'return_pct': stats.get('Return [%]', 0),
        'max_dd':     stats.get('Max. Drawdown [%]', 0),
        'sharpe':     stats.get('Sharpe Ratio', 0),
        'final_eq':   stats.get('Equity Final [$]', 10000),
    }
    print(f"  {name:<10} → {n:>4} trades | Win Rate: {pair_results[name]['win_rate']:.1f}% | Return: {pair_results[name]['return_pct']:.1f}%")
    if n > 0:
        all_trades.append(trades['ReturnPct'].values)

# Flatten all trade returns into one pool
combined_returns = np.concatenate(all_trades) if all_trades else np.array([])
total_trades     = len(combined_returns)

print(f"\n  TOTAL COMBINED TRADES: {total_trades}")


# --- 3. PERFORMANCE TABLE ---
print("\n" + "=" * 75)
print(f"  {'Pair':<10} {'Trades':>7} {'Win Rate':>10} {'Return %':>10} {'Max DD %':>10} {'Sharpe':>8} {'Final $':>10}")
print("=" * 75)
for name, r in pair_results.items():
    print(f"  {name:<10} {r['n_trades']:>7} {r['win_rate']:>9.1f}% {r['return_pct']:>9.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>8.2f} ${r['final_eq']:>9,.0f}")
print("=" * 75)


# --- 4. MONTE CARLO (uses all combined trades, targets 1000) ---
if total_trades > 5:
    TARGET_TRADES = 1000
    SIMULATIONS   = 1000
    STARTING_BAL  = 10000

    print(f"\n  Running {SIMULATIONS} Monte Carlo paths × {TARGET_TRADES} trades each...")

    all_paths = np.zeros((SIMULATIONS, TARGET_TRADES + 1))
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

    # --- PLOT 1: Monte Carlo Paths ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Multi-Pair Forex Strategy | {total_trades} Real Trades Across {len(pair_results)} Pairs\n"
        f"Monte Carlo: {SIMULATIONS} Simulations × {TARGET_TRADES} Trades Each",
        fontsize=14, fontweight='bold'
    )

    # Top-left: Monte Carlo fan
    ax = axes[0, 0]
    for i in range(0, SIMULATIONS, 5):   # plot every 5th path to keep it clean
        ax.plot(all_paths[i], color='steelblue', alpha=0.03, linewidth=0.8)
    ax.fill_between(range(TARGET_TRADES + 1), p5,  p95, color='steelblue', alpha=0.15, label='5–95th pct')
    ax.fill_between(range(TARGET_TRADES + 1), p25, p75, color='steelblue', alpha=0.25, label='25–75th pct')
    ax.plot(median_path, color='red',    linewidth=2.5, label=f'Median: ${median_path[-1]:,.0f}')
    ax.plot(p5,          color='orange', linewidth=1.5, linestyle='--', label=f'5th pct: ${p5[-1]:,.0f}')
    ax.plot(p95,         color='green',  linewidth=1.5, linestyle='--', label=f'95th pct: ${p95[-1]:,.0f}')
    ax.axhline(STARTING_BAL, color='black', linewidth=1, linestyle=':', alpha=0.5, label='Starting Balance')
    ax.set_title("Monte Carlo Simulation (1000 Paths)")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Account Balance ($)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Top-right: Final balance distribution histogram
    ax = axes[0, 1]
    final_balances = all_paths[:, -1]
    ax.hist(final_balances, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.median(final_balances), color='red',    linewidth=2,   label=f'Median: ${np.median(final_balances):,.0f}')
    ax.axvline(np.percentile(final_balances,  5), color='orange', linewidth=1.5, linestyle='--', label=f'5th pct: ${np.percentile(final_balances, 5):,.0f}')
    ax.axvline(np.percentile(final_balances, 95), color='green',  linewidth=1.5, linestyle='--', label=f'95th pct: ${np.percentile(final_balances, 95):,.0f}')
    ax.axvline(STARTING_BAL, color='black', linewidth=1, linestyle=':', label='Starting: $10,000')
    ax.set_title("Distribution of Final Balances")
    ax.set_xlabel("Final Account Balance ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Bottom-left: Win rate per pair bar chart
    ax = axes[1, 0]
    names     = list(pair_results.keys())
    win_rates = [pair_results[n]['win_rate'] for n in names]
    colors    = ['green' if w >= 50 else 'tomato' for w in win_rates]
    bars = ax.bar(names, win_rates, color=colors, edgecolor='white', alpha=0.85)
    ax.axhline(50, color='black', linewidth=1, linestyle='--', alpha=0.5, label='50% breakeven')
    ax.set_title("Win Rate by Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # Bottom-right: Return % per pair bar chart
    ax = axes[1, 1]
    returns_list = [pair_results[n]['return_pct'] for n in names]
    colors2      = ['green' if r >= 0 else 'tomato' for r in returns_list]
    bars2 = ax.bar(names, returns_list, color=colors2, edgecolor='white', alpha=0.85)
    ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.4)
    ax.set_title("Total Return % by Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("Return (%)")
    for bar, val in zip(bars2, returns_list):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- 5. FINAL SUMMARY ---
    prob_profit = np.mean(final_balances > STARTING_BAL) * 100
    prob_double = np.mean(final_balances > STARTING_BAL * 2) * 100
    prob_ruin   = np.mean(final_balances < STARTING_BAL * 0.5) * 100

    print("\n" + "=" * 55)
    print("  MONTE CARLO SUMMARY (1000 trades, 1000 simulations)")
    print("=" * 55)
    print(f"  Starting Balance:      ${STARTING_BAL:>10,.2f}")
    print(f"  Median Final Balance:  ${np.median(final_balances):>10,.2f}")
    print(f"  Best Case  (95th pct): ${np.percentile(final_balances, 95):>10,.2f}")
    print(f"  Worst Case  (5th pct): ${np.percentile(final_balances,  5):>10,.2f}")
    print(f"  Prob. of Profit:       {prob_profit:>9.1f}%")
    print(f"  Prob. of Doubling:     {prob_double:>9.1f}%")
    print(f"  Prob. of 50% Ruin:     {prob_ruin:>9.1f}%")
    print("=" * 55)

else:
    print("\nNot enough trades found across all pairs. Try widening the date range.")
