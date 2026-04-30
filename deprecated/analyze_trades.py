import sqlite3
import statistics

conn = sqlite3.connect('data/bot.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# All closed trades
cur.execute("""SELECT * FROM trades WHERE module='crypto' AND status IN ('CLOSED','SIMULATED') 
               ORDER BY id""")
trades = [dict(r) for r in cur.fetchall()]

wins = [t for t in trades if t['profit_loss'] > 0]
losses = [t for t in trades if t['profit_loss'] < 0]
zeros = [t for t in trades if t['profit_loss'] == 0]

print("=" * 60)
print("TRADE ANALYSIS REPORT")
print("=" * 60)
print(f"Total closed trades: {len(trades)}")
print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"Break-even: {len(zeros)}")
print()

if wins:
    win_pnls = [t['profit_loss'] for t in wins]
    print(f"Avg WIN:  ${statistics.mean(win_pnls):.4f}")
    print(f"Max WIN:  ${max(win_pnls):.4f}")
    print(f"Med WIN:  ${statistics.median(win_pnls):.4f}")

if losses:
    loss_pnls = [t['profit_loss'] for t in losses]
    print(f"Avg LOSS: ${statistics.mean(loss_pnls):.4f}")
    print(f"Max LOSS: ${min(loss_pnls):.4f}")
    print(f"Med LOSS: ${statistics.median(loss_pnls):.4f}")

total = sum(t['profit_loss'] for t in trades)
print(f"\nTotal P&L: ${total:.4f}")
print(f"Avg P&L per trade: ${total/len(trades):.4f}")

if wins and losses:
    ratio = statistics.mean(win_pnls) / abs(statistics.mean(loss_pnls))
    print(f"Win/Loss ratio: {ratio:.2f}")

# Per symbol analysis
print("\n--- PER SYMBOL ---")
for sym in ['BTCUSDT', 'ETHUSDT']:
    sym_trades = [t for t in trades if t['market_id'] == sym]
    sym_wins = [t for t in sym_trades if t['profit_loss'] > 0]
    sym_losses = [t for t in sym_trades if t['profit_loss'] < 0]
    if sym_trades:
        sym_pnl = sum(t['profit_loss'] for t in sym_trades)
        wr = len(sym_wins)/len(sym_trades)*100 if sym_trades else 0
        avg_w = statistics.mean([t['profit_loss'] for t in sym_wins]) if sym_wins else 0
        avg_l = statistics.mean([t['profit_loss'] for t in sym_losses]) if sym_losses else 0
        print(f"{sym}: {len(sym_trades)} trades | WR={wr:.1f}% | P&L=${sym_pnl:.4f} | AvgW=${avg_w:.4f} AvgL=${avg_l:.4f}")

# Size distribution analysis
print("\n--- POSITION SIZE ANALYSIS ---")
sizes = [t['size_usdc'] for t in trades]
print(f"Avg size: ${statistics.mean(sizes):.2f}")
print(f"Min/Max: ${min(sizes):.2f} / ${max(sizes):.2f}")

# Win by size bucket
small = [t for t in trades if t['size_usdc'] < 200]
medium = [t for t in trades if 200 <= t['size_usdc'] < 300]
large = [t for t in trades if t['size_usdc'] >= 300]
for label, bucket in [("Small <$200", small), ("Med $200-300", medium), ("Large >$300", large)]:
    if bucket:
        bw = len([t for t in bucket if t['profit_loss'] > 0])
        bp = sum(t['profit_loss'] for t in bucket)
        print(f"  {label}: {len(bucket)} trades, WR={bw/len(bucket)*100:.0f}%, P&L=${bp:.4f}")

# Last 30 trades to see recent trend
print("\n--- LAST 30 TRADES ---")
for t in trades[-30:]:
    pnl = t['profit_loss']
    marker = "WIN " if pnl > 0 else "LOSS"
    print(f"  {marker} {t['market_id']:8} | ${pnl:+.4f} | size=${t['size_usdc']:.0f}")

conn.close()
