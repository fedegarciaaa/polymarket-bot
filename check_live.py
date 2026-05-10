"""Check live_focused activity."""
import sqlite3, time
db = sqlite3.connect("/home/botuser/polymarket-bot/data/bot.db", timeout=10)
db.row_factory = sqlite3.Row
since = int(time.time()) - 600  # last 10 min

print("=== live_focused activity last 10 min ===")
rows = list(db.execute(
    "SELECT status, COUNT(*) n FROM crypto_lag_quotes "
    "WHERE variant=? AND ts >= ? GROUP BY status",
    ("live_focused", since),
))
if not rows:
    print("  (no live_focused activity yet)")
for r in rows:
    print(f"  status={r['status']} n={r['n']}")

print()
print("=== Recent decisions for live_focused (snapshots) ===")
rows = list(db.execute(
    "SELECT ts, symbol, p_model, poly_mid, edge_bid, edge_ask, decision "
    "FROM crypto_lag_state_snapshots "
    "WHERE variant=? AND ts >= ? ORDER BY ts DESC LIMIT 8",
    ("live_focused", since),
))
if not rows:
    print("  (no snapshots — variant may not be in event window)")
for r in rows:
    print(
        f"  {r['symbol']} p_model={r['p_model']:.3f} poly_mid={r['poly_mid']:.3f} "
        f"edge_bid={r['edge_bid']:+.4f} edge_ask={r['edge_ask']:+.4f} "
        f"decision={r['decision']}"
    )

print()
print("=== Comparison: maker_focused (paper twin) last 10 min ===")
for r in db.execute(
    "SELECT status, COUNT(*) n FROM crypto_lag_quotes "
    "WHERE variant=? AND ts >= ? GROUP BY status",
    ("maker_focused", since),
):
    print(f"  status={r['status']} n={r['n']}")
