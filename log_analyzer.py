"""
Weather Bot v2 log & DB analyzer.

Reads:
  - logs/events.jsonl  (structured events)
  - data/bot.db        (trades, skips, source_reliability, market_resolutions)

Emits a Markdown report at reports/analysis_YYYYMMDD_HHMM.md with:
  - PnL and win rate (global, by side, by hour)
  - Source reliability (Brier, MAE, trades_used)
  - Top-5 losing markets
  - Top-5 skip reasons
  - Calibration histogram (prob_real_estimated vs actual outcome)
  - Confidence distribution of executed trades

Usage:
    python log_analyzer.py --last-hours 24 --output reports/
"""

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _iso_cutoff(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def global_pnl(conn, cutoff: str) -> dict:
    c = conn.cursor()
    c.execute(
        """SELECT COUNT(*) as n,
                  SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                  COALESCE(SUM(profit_loss), 0) as pnl,
                  AVG(profit_loss) as avg_pnl,
                  MAX(profit_loss) as best,
                  MIN(profit_loss) as worst,
                  AVG(confidence_score) as avg_conf
           FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND closed_at >= ?""",
        (cutoff,),
    )
    r = c.fetchone()
    n = r["n"] or 0
    wins = r["wins"] or 0
    return {
        "closed": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": (wins / n * 100) if n else 0.0,
        "pnl": r["pnl"] or 0.0,
        "avg_pnl": r["avg_pnl"] or 0.0,
        "best": r["best"] or 0.0,
        "worst": r["worst"] or 0.0,
        "avg_confidence": r["avg_conf"] or 0.0,
    }


def pnl_by_side(conn, cutoff: str) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT side, COUNT(*) as n,
                  SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                  COALESCE(SUM(profit_loss), 0) as pnl
           FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND closed_at >= ?
           GROUP BY side""",
        (cutoff,),
    )
    rows = []
    for r in c.fetchall():
        n = r["n"]
        rows.append({
            "side": r["side"],
            "n": n,
            "wins": r["wins"],
            "win_rate": (r["wins"] / n * 100) if n else 0.0,
            "pnl": r["pnl"],
        })
    return rows


def pnl_by_hour(conn, cutoff: str) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT timestamp, profit_loss FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND closed_at >= ?""",
        (cutoff,),
    )
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for r in c.fetchall():
        try:
            hour = datetime.fromisoformat(r["timestamp"]).hour
        except (ValueError, TypeError):
            continue
        buckets[hour]["n"] += 1
        buckets[hour]["pnl"] += r["profit_loss"] or 0.0
        if (r["profit_loss"] or 0) > 0:
            buckets[hour]["wins"] += 1
    out = []
    for h in sorted(buckets.keys()):
        b = buckets[h]
        out.append({
            "hour_utc": h,
            "n": b["n"],
            "win_rate": (b["wins"] / b["n"] * 100) if b["n"] else 0.0,
            "pnl": b["pnl"],
        })
    return out


def top_losing_markets(conn, cutoff: str, limit: int = 5) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT market_question, COUNT(*) as n, COALESCE(SUM(profit_loss),0) as pnl
           FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND closed_at >= ?
           GROUP BY market_id
           ORDER BY pnl ASC LIMIT ?""",
        (cutoff, limit),
    )
    return [dict(r) for r in c.fetchall()]


def top_skip_reasons(conn, cutoff: str, limit: int = 10) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT reason_code, COUNT(*) as n FROM opportunity_skips
           WHERE timestamp >= ?
           GROUP BY reason_code ORDER BY n DESC LIMIT ?""",
        (cutoff, limit),
    )
    return [dict(r) for r in c.fetchall()]


def source_reliability(conn) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT source_name, metric, trades_used, brier_score, mae,
                  mean_latency_ms, consecutive_failures, last_updated
           FROM source_reliability
           ORDER BY metric, brier_score ASC"""
    )
    return [dict(r) for r in c.fetchall()]


def calibration_buckets(conn, cutoff: str) -> list[dict]:
    c = conn.cursor()
    c.execute(
        """SELECT prob_real_estimated, profit_loss FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND prob_real_estimated IS NOT NULL
             AND closed_at >= ?""",
        (cutoff,),
    )
    bins = [[] for _ in range(10)]
    for r in c.fetchall():
        p = max(0.0, min(0.999999, r["prob_real_estimated"]))
        idx = int(p * 10)
        outcome = 1.0 if (r["profit_loss"] or 0) > 0 else 0.0
        bins[idx].append((p, outcome))
    out = []
    for i, b in enumerate(bins):
        if not b:
            out.append({"bin": f"{i*10}-{(i+1)*10}%", "n": 0, "predicted": 0.0, "actual": 0.0})
        else:
            out.append({
                "bin": f"{i*10}-{(i+1)*10}%",
                "n": len(b),
                "predicted": sum(p for p, _ in b) / len(b),
                "actual": sum(o for _, o in b) / len(b),
            })
    return out


def confidence_distribution(conn, cutoff: str) -> dict:
    c = conn.cursor()
    c.execute(
        """SELECT confidence_score, profit_loss FROM trades
           WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
             AND confidence_score IS NOT NULL AND closed_at >= ?""",
        (cutoff,),
    )
    bucketed = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for r in c.fetchall():
        cs = r["confidence_score"] or 0
        base = int(cs // 5) * 5
        key = f"{base}-{base + 5}"
        bucketed[key]["n"] += 1
        bucketed[key]["pnl"] += r["profit_loss"] or 0.0
        if (r["profit_loss"] or 0) > 0:
            bucketed[key]["wins"] += 1
    return dict(sorted(bucketed.items(), key=lambda kv: int(kv[0].split("-")[0])))


def events_summary(log_path: str, cutoff_dt: datetime) -> dict:
    if not os.path.exists(log_path):
        return {"total": 0, "by_type": {}}
    counts: Counter = Counter()
    total = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("timestamp") or ev.get("timestamp_iso")
            if ts:
                try:
                    if datetime.fromisoformat(ts.replace("Z", "+00:00")) < cutoff_dt:
                        continue
                except ValueError:
                    pass
            counts[ev.get("type") or ev.get("event_type") or "unknown"] += 1
            total += 1
    return {"total": total, "by_type": dict(counts.most_common(20))}


def render_markdown(hours: int, data: dict) -> str:
    lines: list[str] = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Weather Bot v2 — Analysis Report")
    lines.append(f"Generated: **{now_str}**  |  Window: **last {hours}h**\n")

    g = data["global"]
    lines.append("## Global performance")
    lines.append("")
    lines.append(f"- Closed trades: **{g['closed']}** (wins {g['wins']}, losses {g['losses']})")
    lines.append(f"- Win rate: **{g['win_rate']:.1f}%**")
    lines.append(f"- Total PnL: **${g['pnl']:+.4f}**")
    lines.append(f"- Avg PnL/trade: ${g['avg_pnl']:+.4f}")
    lines.append(f"- Best / Worst: ${g['best']:+.4f} / ${g['worst']:+.4f}")
    lines.append(f"- Avg confidence at entry: {g['avg_confidence']:.1f}\n")

    lines.append("## PnL by side")
    lines.append("")
    lines.append("| Side | N | Win rate | PnL |")
    lines.append("|------|---|----------|-----|")
    for row in data["by_side"]:
        lines.append(f"| {row['side']} | {row['n']} | {row['win_rate']:.1f}% | ${row['pnl']:+.4f} |")
    if not data["by_side"]:
        lines.append("| — | 0 | — | — |")
    lines.append("")

    lines.append("## PnL by hour (UTC)")
    lines.append("")
    lines.append("| Hour | N | Win rate | PnL |")
    lines.append("|------|---|----------|-----|")
    for row in data["by_hour"]:
        lines.append(f"| {row['hour_utc']:02d} | {row['n']} | {row['win_rate']:.1f}% | ${row['pnl']:+.4f} |")
    if not data["by_hour"]:
        lines.append("| — | 0 | — | — |")
    lines.append("")

    lines.append("## Top-5 losing markets")
    lines.append("")
    lines.append("| Market | N trades | PnL |")
    lines.append("|--------|----------|-----|")
    for row in data["top_losers"]:
        q = (row.get("market_question") or "?")[:90].replace("|", "/")
        lines.append(f"| {q} | {row['n']} | ${row['pnl']:+.4f} |")
    if not data["top_losers"]:
        lines.append("| — | — | — |")
    lines.append("")

    lines.append("## Top skip reasons")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for row in data["skip_reasons"]:
        lines.append(f"| `{row['reason_code']}` | {row['n']} |")
    if not data["skip_reasons"]:
        lines.append("| — | 0 |")
    lines.append("")

    lines.append("## Source reliability")
    lines.append("")
    lines.append("| Source | Metric | Trades | Brier | MAE | Latency ms | Consec. fails |")
    lines.append("|--------|--------|--------|-------|-----|-----------:|---------------|")
    for s in data["sources"]:
        lines.append(
            f"| {s['source_name']} | {s['metric']} | {s['trades_used'] or 0} | "
            f"{(s['brier_score'] or 0):.4f} | {(s['mae'] or 0):.3f} | "
            f"{(s['mean_latency_ms'] or 0):.0f} | {s['consecutive_failures'] or 0} |"
        )
    if not data["sources"]:
        lines.append("| — | — | — | — | — | — | — |")
    lines.append("")

    lines.append("## Calibration (prob_real_estimated vs outcome)")
    lines.append("")
    lines.append("Well-calibrated means `predicted ≈ actual` in each row.\n")
    lines.append("| Bucket | N | Predicted avg | Actual win rate |")
    lines.append("|--------|---|---------------|-----------------|")
    for b in data["calibration"]:
        if b["n"] == 0:
            lines.append(f"| {b['bin']} | 0 | — | — |")
        else:
            lines.append(f"| {b['bin']} | {b['n']} | {b['predicted']:.3f} | {b['actual']:.3f} |")
    lines.append("")

    lines.append("## Confidence distribution (executed trades)")
    lines.append("")
    lines.append("| Confidence bucket | N | Wins | PnL |")
    lines.append("|-------------------|---|------|-----|")
    if not data["confidence_dist"]:
        lines.append("| — | 0 | — | — |")
    for bucket, v in data["confidence_dist"].items():
        lines.append(f"| {bucket} | {v['n']} | {v['wins']} | ${v['pnl']:+.4f} |")
    lines.append("")

    lines.append("## Event log summary")
    lines.append("")
    ev = data["events"]
    lines.append(f"Total events in window: **{ev['total']}**\n")
    if ev["by_type"]:
        lines.append("| Event type | Count |")
        lines.append("|------------|-------|")
        for k, n in ev["by_type"].items():
            lines.append(f"| `{k}` | {n} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Weather Bot v2 logs and DB")
    parser.add_argument("--last-hours", type=int, default=24)
    parser.add_argument("--output", default="reports/")
    parser.add_argument("--db", default="data/bot.db")
    parser.add_argument("--events", default="logs/events.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: database not found at {args.db}")
        return 1

    cutoff = _iso_cutoff(args.last_hours)
    cutoff_dt = datetime.fromisoformat(cutoff)

    conn = _connect(args.db)
    try:
        data = {
            "global": global_pnl(conn, cutoff),
            "by_side": pnl_by_side(conn, cutoff),
            "by_hour": pnl_by_hour(conn, cutoff),
            "top_losers": top_losing_markets(conn, cutoff),
            "skip_reasons": top_skip_reasons(conn, cutoff),
            "sources": source_reliability(conn),
            "calibration": calibration_buckets(conn, cutoff),
            "confidence_dist": confidence_distribution(conn, cutoff),
            "events": events_summary(args.events, cutoff_dt),
        }
    finally:
        conn.close()

    md = render_markdown(args.last_hours, data)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = Path(args.output) / f"analysis_{stamp}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
