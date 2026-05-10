"""Quick audit script — uploaded to /tmp/ for hourly cron use."""
import sqlite3, time
db = sqlite3.connect("/home/botuser/polymarket-bot/data/bot.db", timeout=15)
db.row_factory = sqlite3.Row

since_1h = int(time.time()) - 3600
print("=== 1h por variante ===")
for r in db.execute(
    "SELECT variant, COUNT(*) n, ROUND(SUM(realized_pnl_usdc),2) pnl "
    "FROM crypto_lag_closes WHERE ts >= ? GROUP BY variant ORDER BY pnl DESC",
    (since_1h,),
):
    print(f"  {r['variant']:22s} closes={r['n']:3d} pnl=${r['pnl']:+8.2f}")

since_24h = int(time.time()) - 86400
print()
print("=== Top 5 outliers ratio 24h ===")
for r in db.execute(
    """SELECT c.variant, c.symbol,
       ROUND(c.realized_pnl_usdc,2) pnl,
       ROUND((SELECT SUM(fill_size_usdc) FROM crypto_lag_quotes
              WHERE condition_id=c.condition_id AND variant=c.variant
                AND status LIKE '%fill%'),2) gross
       FROM crypto_lag_closes c WHERE c.ts >= ? AND c.realized_pnl_usdc IS NOT NULL
       ORDER BY (c.realized_pnl_usdc / NULLIF(
         (SELECT SUM(fill_size_usdc) FROM crypto_lag_quotes
          WHERE condition_id=c.condition_id AND variant=c.variant
            AND status LIKE '%fill%'),0)) DESC
       LIMIT 5""",
    (since_24h,),
):
    g = r["gross"] or 0.01
    ratio = r["pnl"] / max(0.01, g)
    print(
        f"  {r['variant']:22s} {r['symbol']} pnl={r['pnl']:+8.2f} "
        f"gross={g:7.2f} ratio={ratio:5.1f}x"
    )

print()
print("=== Mid vs Tail PnL 24h ===")
for r in db.execute(
    """SELECT
       CASE WHEN avg_p < 0.10 OR avg_p > 0.90 THEN 'tail' ELSE 'mid' END bucket,
       COUNT(*) n, ROUND(SUM(pnl),2) pnl_tot, ROUND(AVG(pnl),2) pnl_avg
       FROM (SELECT c.realized_pnl_usdc pnl,
             (SELECT AVG(price) FROM crypto_lag_quotes q
              WHERE q.condition_id=c.condition_id AND q.variant=c.variant
                AND q.status LIKE '%fill%') avg_p
             FROM crypto_lag_closes c
             WHERE c.ts >= ? AND c.realized_pnl_usdc IS NOT NULL)
       WHERE avg_p IS NOT NULL GROUP BY bucket""",
    (since_24h,),
):
    print(
        f"  {r['bucket']:6s} n={r['n']:4d} total=${r['pnl_tot']:+9.2f} "
        f"avg=${r['pnl_avg']:+6.2f}/close"
    )

print()
print("=== Tail fills 1h ===")
for r in db.execute(
    """SELECT
       SUM(CASE WHEN price < 0.10 OR price > 0.90 THEN 1 ELSE 0 END) tail,
       COUNT(*) total
       FROM crypto_lag_quotes WHERE status LIKE '%fill%' AND ts >= ?""",
    (since_1h,),
):
    t = r["total"] or 0
    if t > 0:
        print(f"  tail={r['tail']}/{t} ({100 * r['tail'] / t:.1f}%)")

print()
print("=== Bankrolls cumulative ===")
for r in db.execute(
    "SELECT variant, ROUND(bankroll_usdc,2) bk, ROUND(realized_pnl_lifetime_usdc,2) pnl "
    "FROM crypto_lag_variant_state ORDER BY pnl DESC"
):
    roi = r["pnl"] / 10.0
    print(f"  {r['variant']:22s} ${r['bk']:7.2f} PnL=${r['pnl']:+8.2f} ROI={roi:+6.1f}%")
