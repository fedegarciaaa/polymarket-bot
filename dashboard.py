"""
Dashboard - Flask web interface for the Weather Bot v2.

Panels:
  - Health: PID, uptime, sources status (green/amber/red).
  - Portfolio: capital, exposure, PnL, WR.
  - Confidence: distribution of confidence scores.
  - Calibration: predicted vs actual buckets.
  - Rejected opportunities: recent skips by reason code.
  - Trades: open + recent closed.

Runs independently from main.py; just reads data/bot.db and logs/events.jsonl.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml
from flask import Flask, jsonify, render_template_string

from database import Database

app = Flask(__name__)

CONFIG_PATH = os.environ.get("BOT_CONFIG", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = CONFIG.get("database", {}).get("path", "data/bot.db")
INITIAL_CAPITAL = CONFIG["bot"]["demo_capital"]
LOCK_PATH = Path("data/bot.lock")
EVENTS_PATH = Path("logs/events.jsonl")


def db() -> Database:
    return Database(DB_PATH)


# ============================================================
# API
# ============================================================
@app.route("/api/health")
def api_health():
    pid = None
    alive = False
    started_at = None
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            alive = psutil.pid_exists(pid)
            if alive:
                started_at = datetime.fromtimestamp(
                    psutil.Process(pid).create_time(), tz=timezone.utc
                ).isoformat()
        except (ValueError, OSError, psutil.NoSuchProcess):
            pass
    uptime_sec = None
    if started_at:
        uptime_sec = int((datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds())
    return jsonify({
        "pid": pid, "alive": alive, "started_at": started_at, "uptime_seconds": uptime_sec,
        "lock_file_exists": LOCK_PATH.exists(),
    })


@app.route("/api/sources")
def api_sources():
    d = db()
    rows = d.get_source_reliability()
    d.close()
    return jsonify({"sources": rows})


@app.route("/api/stats")
def api_stats():
    d = db()
    stats = d.get_statistics(INITIAL_CAPITAL)
    d.close()
    return jsonify(stats)


@app.route("/api/calibration")
def api_calibration():
    d = db()
    buckets = d.get_calibration_buckets()
    d.close()
    return jsonify({"buckets": buckets})


@app.route("/api/skips")
def api_skips():
    d = db()
    recent = d.get_recent_skips(50)
    summary = d.get_skip_reasons_summary(24)
    d.close()
    return jsonify({"recent": recent, "summary": summary})


@app.route("/api/trades")
def api_trades():
    d = db()
    open_pos = d.get_open_positions()
    closed = d.get_recent_closed_trades(30)
    d.close()
    return jsonify({"open": open_pos, "closed": closed})


@app.route("/api/cycles")
def api_cycles():
    d = db()
    cycles = d.get_cycles_history(50)
    d.close()
    return jsonify({"cycles": cycles})


# ============================================================
# v3 — per-bet detail, open-bet summary, sources health, skips
# ============================================================
@app.route("/api/bets/open")
def api_bets_open():
    d = db()
    open_pos = d.get_open_positions()
    # Enrich with latest evaluation (if any).
    for t in open_pos:
        try:
            evs = d.get_bet_evaluations(t["id"], limit=1)
            t["latest_eval"] = evs[0] if evs else None
        except Exception:
            t["latest_eval"] = None
    d.close()
    return jsonify({"open": open_pos})


@app.route("/api/bet/<int:trade_id>")
def api_bet_detail(trade_id: int):
    d = db()
    trade = d.get_trade_by_id(trade_id)
    if not trade:
        d.close()
        return jsonify({"error": "not_found"}), 404
    try:
        evals = d.get_bet_evaluations(trade_id, limit=200)
    except Exception:
        evals = []
    try:
        snapshots = d.get_forecast_snapshots(trade_id, limit=300)
    except Exception:
        snapshots = []
    # Group snapshots by (timestamp, source)
    d.close()
    return jsonify({
        "trade": trade,
        "evaluations": evals,
        "forecast_snapshots": snapshots,
    })


@app.route("/api/skips/recent")
def api_skips_recent():
    from flask import request
    n = int(request.args.get("n", 100))
    d = db()
    recent = d.get_recent_skips(n)
    d.close()
    return jsonify({"recent": recent})


@app.route("/api/sources/health")
def api_sources_health():
    d = db()
    rows = d.get_source_reliability()
    d.close()
    return jsonify({"sources": rows})


@app.route("/api/events/recent")
def api_events_recent():
    from flask import request
    n = int(request.args.get("n", 80))
    events = []
    if EVENTS_PATH.exists():
        try:
            lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
            for raw in reversed(lines[-500:]):
                if not raw.strip():
                    continue
                try:
                    ev = json.loads(raw)
                    # Only surface user-relevant events
                    if ev.get("type") in (
                        "TRADE_OPEN", "TRADE_CLOSE", "CYCLE_START", "CYCLE_END",
                        "BET_REEVALUATED", "OPPORTUNITY_SKIP", "MARKETS_SCANNED",
                    ):
                        events.append(ev)
                        if len(events) >= n:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception:
            pass
    return jsonify({"events": events})


@app.route("/api/portfolio")
def api_portfolio():
    """Portfolio with unrealized P&L included."""
    d = db()
    stats = d.get_statistics(INITIAL_CAPITAL)
    open_pos = d.get_open_positions()
    d.close()
    unrealized = sum(float(p.get("unrealized_pnl") or 0) for p in open_pos)
    total_exposure = sum(float(p.get("size_usdc") or 0) for p in open_pos)
    collapsed = [
        p for p in open_pos
        if (p.get("best_bid_current") or 0) > 0
        and float(p.get("price_entry") or 1) > 0
        and (float(p.get("best_bid_current") or 0) / float(p.get("price_entry") or 1)) < 0.1
    ]
    return jsonify({
        **stats,
        "unrealized_pnl": round(unrealized, 2),
        "total_exposure": round(total_exposure, 2),
        "open_count": len(open_pos),
        "collapsed_count": len(collapsed),
        "total_pnl_including_unrealized": round(stats.get("total_pnl", 0) + unrealized, 2),
    })


# ============================================================
# Crypto-Lag API endpoints
# ------------------------------------------------------------
# Reads the three crypto_lag_* tables created by database.py. All endpoints
# are safe even when the module is disabled (they just return empty arrays).
# ============================================================
def _has_crypto_lag_tables(conn) -> bool:
    """Cheap check; the tables are CREATE IF NOT EXISTS but a fresh DB before
    the bot has restarted with crypto_lag enabled may still lack them."""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='crypto_lag_state_snapshots'"
        ).fetchall()
        return bool(rows)
    except Exception:
        return False


def _has_variant_column(conn) -> bool:
    """Older DBs (pre-shadow-mode migration) lack the `variant` column. We
    detect that and skip variant-filtering when absent so the dashboard keeps
    working during a partial deploy / rollback."""
    try:
        rows = conn.execute("PRAGMA table_info(crypto_lag_state_snapshots)").fetchall()
        return any(r["name"] == "variant" for r in rows)
    except Exception:
        return False


def _variant_arg() -> str:
    """Return the requested variant name from `?variant=X`, defaulting to
    'main' so legacy clients (without the query param) keep getting the
    primary bot's data."""
    from flask import request
    return str(request.args.get("variant", "main"))


def _configured_variants() -> list[str]:
    """Read the configured variant list from config.yaml.

    Falls back to `['main']` if the user hasn't migrated to the
    `crypto_lag.variants:` block yet."""
    cfg = (CONFIG.get("crypto_lag") or {}).get("variants")
    if isinstance(cfg, dict) and cfg:
        return [str(k) for k in cfg.keys()]
    return ["main"]


@app.route("/api/crypto_lag/variants")
def api_crypto_lag_variants():
    """List all configured variants and how many snapshots/closes each one
    has logged. Used by the dashboard JS to know which sections to render.
    """
    d = db()
    out = []
    has_var = _has_crypto_lag_tables(d.conn) and _has_variant_column(d.conn)
    for v in _configured_variants():
        rec = {"name": v, "snapshots": 0, "closes": 0, "fills_24h": 0}
        if has_var:
            try:
                cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
                rec["snapshots"] = d.conn.execute(
                    "SELECT COUNT(*) AS n FROM crypto_lag_state_snapshots WHERE variant = ?",
                    (v,),
                ).fetchone()["n"] or 0
                rec["closes"] = d.conn.execute(
                    "SELECT COUNT(*) AS n FROM crypto_lag_closes WHERE variant = ?",
                    (v,),
                ).fetchone()["n"] or 0
                rec["fills_24h"] = d.conn.execute(
                    """SELECT COUNT(*) AS n FROM crypto_lag_quotes
                       WHERE variant = ? AND status IN ('filled','partially_filled')
                       AND ts >= ?""",
                    (v, cutoff_24h),
                ).fetchone()["n"] or 0
            except Exception:
                pass
        out.append(rec)
    d.close()
    return jsonify({"variants": out})


@app.route("/api/crypto_lag/health")
def api_crypto_lag_health():
    variant = _variant_arg()
    enabled = bool((CONFIG.get("crypto_lag") or {}).get("enabled"))
    symbols = [s.get("binance") for s in (CONFIG.get("crypto_lag") or {}).get("symbols", [])]
    capital_pct = float((CONFIG.get("crypto_lag") or {}).get("capital_pct", 0.30))
    halt_path = ((CONFIG.get("crypto_lag") or {}).get("circuit_breakers") or {}).get(
        "halt_file", "data/crypto_lag.halt"
    )
    halted = Path(halt_path).exists()
    d = db()
    has_tables = _has_crypto_lag_tables(d.conn)
    has_var = has_tables and _has_variant_column(d.conn)
    last_snap_ts = None
    last_decision = None
    recent_only_heartbeat = False
    if has_tables:
        if has_var:
            row = d.conn.execute(
                "SELECT MAX(ts) AS ts FROM crypto_lag_state_snapshots WHERE variant = ?",
                (variant,),
            ).fetchone()
        else:
            row = d.conn.execute(
                "SELECT MAX(ts) AS ts FROM crypto_lag_state_snapshots"
            ).fetchone()
        last_snap_ts = row["ts"] if row and row["ts"] else None
        if last_snap_ts is not None:
            if has_var:
                last_row = d.conn.execute(
                    "SELECT decision FROM crypto_lag_state_snapshots WHERE variant = ? ORDER BY ts DESC LIMIT 1",
                    (variant,),
                ).fetchone()
            else:
                last_row = d.conn.execute(
                    "SELECT decision FROM crypto_lag_state_snapshots ORDER BY ts DESC LIMIT 1"
                ).fetchone()
            last_decision = last_row["decision"] if last_row else None
            cutoff = last_snap_ts - 300
            if has_var:
                mix = d.conn.execute(
                    """SELECT COUNT(*) AS n_quote
                       FROM crypto_lag_state_snapshots
                       WHERE variant = ? AND ts >= ? AND decision IN ('BID','ASK','BOTH','NONE')""",
                    (variant, cutoff),
                ).fetchone()
            else:
                mix = d.conn.execute(
                    """SELECT COUNT(*) AS n_quote
                       FROM crypto_lag_state_snapshots
                       WHERE ts >= ? AND decision IN ('BID','ASK','BOTH','NONE')""",
                    (cutoff,),
                ).fetchone()
            recent_only_heartbeat = (mix["n_quote"] or 0) == 0
    d.close()
    return jsonify({
        "enabled": enabled,
        "symbols": symbols,
        "capital_pct": capital_pct,
        "halted": halted,
        "tables_present": has_tables,
        "variant": variant,
        "last_snapshot_ts": last_snap_ts,
        "last_decision": last_decision,
        "recent_only_heartbeat": recent_only_heartbeat,
        "now": datetime.now(timezone.utc).timestamp(),
    })


@app.route("/api/crypto_lag/kpis")
def api_crypto_lag_kpis():
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False})
    has_var = _has_variant_column(d.conn)
    c = d.conn.cursor()
    cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
    cutoff_1h = datetime.now(timezone.utc).timestamp() - 3600
    # Build variant filter clauses inline so the queries stay readable.
    vw = " AND variant = ?" if has_var else ""
    vw_only = " WHERE variant = ?" if has_var else ""
    vparams = (variant,) if has_var else ()
    pnl_total = c.execute(
        f"SELECT COALESCE(SUM(realized_pnl_usdc), 0) AS s FROM crypto_lag_closes{vw_only}",
        vparams,
    ).fetchone()["s"] or 0.0
    pnl_24h = c.execute(
        f"SELECT COALESCE(SUM(realized_pnl_usdc), 0) AS s FROM crypto_lag_closes WHERE ts >= ?{vw}",
        (cutoff_24h, *vparams),
    ).fetchone()["s"] or 0.0
    closes_24h = c.execute(
        f"SELECT COUNT(*) AS n FROM crypto_lag_closes WHERE ts >= ?{vw}",
        (cutoff_24h, *vparams),
    ).fetchone()["n"] or 0
    wins_24h = c.execute(
        f"SELECT COUNT(*) AS n FROM crypto_lag_closes WHERE ts >= ? AND realized_pnl_usdc > 0{vw}",
        (cutoff_24h, *vparams),
    ).fetchone()["n"] or 0
    fills_24h = c.execute(
        f"SELECT COUNT(*) AS n FROM crypto_lag_quotes WHERE ts >= ? AND status IN ('filled','partially_filled'){vw}",
        (cutoff_24h, *vparams),
    ).fetchone()["n"] or 0
    fills_1h = c.execute(
        f"SELECT COUNT(*) AS n FROM crypto_lag_quotes WHERE ts >= ? AND status IN ('filled','partially_filled'){vw}",
        (cutoff_1h, *vparams),
    ).fetchone()["n"] or 0
    adverse_24h = c.execute(
        f"SELECT COUNT(*) AS n FROM crypto_lag_quotes WHERE ts >= ? AND is_adverse = 1{vw}",
        (cutoff_24h, *vparams),
    ).fetchone()["n"] or 0
    capital_used = c.execute(
        f"SELECT COALESCE(SUM(fill_size_usdc), 0) AS s FROM crypto_lag_quotes WHERE status IN ('filled','partially_filled'){vw}",
        vparams,
    ).fetchone()["s"] or 0.0
    d.close()
    win_rate = (wins_24h / closes_24h) if closes_24h > 0 else None
    adverse_rate = (adverse_24h / fills_24h) if fills_24h > 0 else None
    return jsonify({
        "available": True,
        "pnl_total_usdc": round(pnl_total, 4),
        "pnl_24h_usdc": round(pnl_24h, 4),
        "closes_24h": closes_24h,
        "wins_24h": wins_24h,
        "fills_24h": fills_24h,
        "fills_1h": fills_1h,
        "win_rate_24h": round(win_rate, 3) if win_rate is not None else None,
        "adverse_rate_24h": round(adverse_rate, 3) if adverse_rate is not None else None,
        "capital_used_usdc": round(capital_used, 2),
    })


@app.route("/api/crypto_lag/snapshots")
def api_crypto_lag_snapshots():
    """Last N snapshots per symbol for the price/probability charts."""
    from flask import request
    n = int(request.args.get("n", 200))
    minutes = int(request.args.get("minutes", 30))
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False, "by_symbol": {}})
    has_var = _has_variant_column(d.conn)
    cutoff = datetime.now(timezone.utc).timestamp() - minutes * 60
    if has_var:
        rows = d.conn.execute(
            """SELECT ts, symbol, binance_mid, sigma_realized, book_imbalance,
                      p_model, fair_mid, poly_bid, poly_ask, poly_mid,
                      edge_bid, edge_ask, decision
               FROM crypto_lag_state_snapshots
               WHERE ts >= ? AND variant = ?
               ORDER BY ts ASC""",
            (cutoff, variant),
        ).fetchall()
    else:
        rows = d.conn.execute(
            """SELECT ts, symbol, binance_mid, sigma_realized, book_imbalance,
                      p_model, fair_mid, poly_bid, poly_ask, poly_mid,
                      edge_bid, edge_ask, decision
               FROM crypto_lag_state_snapshots
               WHERE ts >= ?
               ORDER BY ts ASC""",
            (cutoff,),
        ).fetchall()
    d.close()
    by_symbol: dict = {}
    for r in rows:
        rec = dict(r)
        by_symbol.setdefault(rec["symbol"], []).append(rec)
    # Cap each symbol's series to the last n points
    for sym in list(by_symbol.keys()):
        if len(by_symbol[sym]) > n:
            by_symbol[sym] = by_symbol[sym][-n:]
    return jsonify({"available": True, "by_symbol": by_symbol, "minutes": minutes})


@app.route("/api/crypto_lag/decisions")
def api_crypto_lag_decisions():
    """Distribution of decisions over the last 24h."""
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False})
    has_var = _has_variant_column(d.conn)
    cutoff = datetime.now(timezone.utc).timestamp() - 86400
    if has_var:
        rows = d.conn.execute(
            """SELECT decision, COUNT(*) AS n FROM crypto_lag_state_snapshots
               WHERE ts >= ? AND variant = ? GROUP BY decision""",
            (cutoff, variant),
        ).fetchall()
    else:
        rows = d.conn.execute(
            """SELECT decision, COUNT(*) AS n FROM crypto_lag_state_snapshots
               WHERE ts >= ? GROUP BY decision""",
            (cutoff,),
        ).fetchall()
    d.close()
    return jsonify({
        "available": True,
        "buckets": {r["decision"] or "NONE": r["n"] for r in rows},
    })


@app.route("/api/crypto_lag/pnl_series")
def api_crypto_lag_pnl_series():
    """Cumulative P&L series from the closes table — last 24h."""
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False, "points": []})
    has_var = _has_variant_column(d.conn)
    cutoff = datetime.now(timezone.utc).timestamp() - 86400
    if has_var:
        rows = d.conn.execute(
            """SELECT ts, realized_pnl_usdc FROM crypto_lag_closes
               WHERE ts >= ? AND variant = ? ORDER BY ts ASC""",
            (cutoff, variant),
        ).fetchall()
    else:
        rows = d.conn.execute(
            """SELECT ts, realized_pnl_usdc FROM crypto_lag_closes
               WHERE ts >= ? ORDER BY ts ASC""",
            (cutoff,),
        ).fetchall()
    d.close()
    points = []
    cum = 0.0
    for r in rows:
        cum += float(r["realized_pnl_usdc"] or 0.0)
        points.append({"ts": r["ts"], "pnl_cum": round(cum, 4)})
    return jsonify({"available": True, "points": points})


@app.route("/api/crypto_lag/fills")
def api_crypto_lag_fills():
    from flask import request
    n = int(request.args.get("n", 25))
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False, "fills": []})
    has_var = _has_variant_column(d.conn)
    if has_var:
        rows = d.conn.execute(
            """SELECT ts, symbol, side, outcome, price, fill_size_usdc, fill_price, is_adverse
               FROM crypto_lag_quotes
               WHERE status IN ('filled','partially_filled') AND variant = ?
               ORDER BY ts DESC LIMIT ?""",
            (variant, n),
        ).fetchall()
    else:
        rows = d.conn.execute(
            """SELECT ts, symbol, side, outcome, price, fill_size_usdc, fill_price, is_adverse
               FROM crypto_lag_quotes
               WHERE status IN ('filled','partially_filled')
               ORDER BY ts DESC LIMIT ?""",
            (n,),
        ).fetchall()
    d.close()
    return jsonify({
        "available": True,
        "fills": [dict(r) for r in rows],
    })


@app.route("/api/crypto_lag/closes")
def api_crypto_lag_closes():
    from flask import request
    n = int(request.args.get("n", 25))
    variant = _variant_arg()
    d = db()
    if not _has_crypto_lag_tables(d.conn):
        d.close()
        return jsonify({"available": False, "closes": []})
    has_var = _has_variant_column(d.conn)
    if has_var:
        rows = d.conn.execute(
            """SELECT ts, symbol, condition_id, realized_pnl_usdc, final_yes_price, reason
               FROM crypto_lag_closes
               WHERE variant = ?
               ORDER BY ts DESC LIMIT ?""",
            (variant, n),
        ).fetchall()
    else:
        rows = d.conn.execute(
            """SELECT ts, symbol, condition_id, realized_pnl_usdc, final_yes_price, reason
               FROM crypto_lag_closes
               ORDER BY ts DESC LIMIT ?""",
            (n,),
        ).fetchall()
    d.close()
    return jsonify({
        "available": True,
        "closes": [dict(r) for r in rows],
    })


# ============================================================
# HTML
# ============================================================
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Weather Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1"></script>
<style>
  *{box-sizing:border-box}
  body{font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:16px 20px}
  h1{color:#58a6ff;font-size:21px;border-bottom:1px solid #21262d;padding-bottom:10px;margin:0 0 14px}
  h2{color:#79c0ff;font-size:13px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin:20px 0 8px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:4px}
  .card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:13px 15px}
  .card.alert{border-color:#f85149;background:#1f0a09}
  .card.warn{border-color:#d29922;background:#1a1509}
  .label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.04em}
  .value{font-size:21px;font-weight:700;margin-top:5px}
  .pos{color:#3fb950}.neg{color:#f85149}.warn-c{color:#d29922}.muted{color:#8b949e}
  table{width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;overflow:hidden;margin-top:6px;font-size:12px}
  th{background:#1c2129;padding:7px 8px;text-align:left;color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:.04em}
  td{padding:6px 8px;border-top:1px solid #21262d;vertical-align:middle}
  tr.row-alert{background:#1f0a0922}
  tr.row-warn{background:#1a150922}
  tr:hover{background:#1c2129}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
  .dot-green{background:#3fb950}.dot-amber{background:#d29922}.dot-red{background:#f85149}.dot-blue{background:#58a6ff}
  .badge{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:600;margin:0 2px;vertical-align:middle}
  .badge-red{background:#5a0f0f;color:#f85149;border:1px solid #f8514933}
  .badge-amber{background:#2d1f00;color:#d29922;border:1px solid #d2992233}
  .badge-green{background:#0d2615;color:#3fb950;border:1px solid #3fb95033}
  .badge-blue{background:#0d1f40;color:#58a6ff;border:1px solid #58a6ff33}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
  @media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
  canvas{background:#161b22;border-radius:8px;padding:8px}
  #feed{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px;max-height:320px;overflow-y:auto;font-size:11px;font-family:monospace}
  .feed-row{padding:3px 0;border-bottom:1px solid #21262d22;display:flex;gap:8px;align-items:flex-start}
  .feed-time{color:#8b949e;white-space:nowrap;min-width:80px}
  .feed-type{min-width:130px;font-weight:600}
  .feed-body{color:#c9d1d9;flex:1;word-break:break-word}
  .type-TRADE_OPEN{color:#3fb950}.type-TRADE_CLOSE{color:#f85149}.type-CYCLE_START,.type-CYCLE_END{color:#8b949e}
  .type-BET_REEVALUATED{color:#58a6ff}.type-OPPORTUNITY_SKIP{color:#d29922}.type-MARKETS_SCANNED{color:#6e7681}
  a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
  .divider{border:none;border-top:1px solid #21262d;margin:18px 0}
  .reason-box{font-size:11px;color:#c9d1d9;background:#0d1117;border-left:3px solid #21262d;padding:4px 8px;margin-top:4px;white-space:pre-wrap;word-break:break-all}

  /* ─── Crypto-Lag section ─────────────────────────────────── */
  .cl-variant-tag{display:inline-block;margin-left:10px;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;vertical-align:middle}
  .cl-variant-tag[data-variant="main"]{background:#0d2615;color:#3fb950;border:1px solid #3fb95044}
  .cl-variant-tag[data-variant="permissive"]{background:#1a1509;color:#d29922;border:1px solid #d2992244}
  .cl-variant-tag[data-variant="aggressive"]{background:#1f0a09;color:#f85149;border:1px solid #f8514944}
  .cl-header{display:flex;flex-direction:column;gap:2px;margin-top:10px}
  .cl-icon{width:18px;height:18px;color:#f7931a;vertical-align:-3px;margin-right:6px}
  .cl-status-badge{margin-left:8px;display:inline-block;vertical-align:middle}
  .cl-sub{font-size:11px;letter-spacing:.04em}
  .cl-kpi-cards{margin-bottom:14px}
  .cl-kpi-cards .card{position:relative;padding:14px 16px}
  .cl-kpi-cards .kpi-sub{font-size:10px;color:#8b949e;margin-top:4px;letter-spacing:.02em}
  .cl-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:6px}
  @media(max-width:900px){.cl-grid{grid-template-columns:1fr}}
  .cl-chart-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:12px 14px 8px}
  .cl-chart-title{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  .cl-chart-body{height:230px}
  .cl-donut-body{height:230px;display:flex;align-items:center;justify-content:center}
  .cl-empty-overlay{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#6e7681;font-size:12px;background:linear-gradient(180deg,transparent 30%,#161b22 80%);pointer-events:none}
  .cl-empty{color:#8b949e;font-size:12px;text-align:center;padding:18px 0}
  #cl-fills-table table, #cl-closes-table table{font-size:11px}
  #cl-fills-table td, #cl-closes-table td{padding:5px 8px}
  /* Reduced-motion: disable fancy chart fills/animations */
  @media (prefers-reduced-motion: reduce){
    .cl-chart-card{transition:none}
  }
</style>
</head>
<body>
<h1>⛅ Weather Bot &mdash; <span id="mode" class="warn-c">loading…</span>
  <span style="font-size:13px;font-weight:400;color:#8b949e;margin-left:12px" id="last-refresh"></span>
</h1>

<div id="alerts-bar"></div>

<h2>Estado del Bot</h2>
<div class="cards" id="health-cards"></div>

<h2>Portfolio</h2>
<div class="cards" id="stats-cards"></div>

<hr class="divider">

<h2>Posiciones Abiertas <span id="open-count" style="color:#8b949e;font-weight:400"></span></h2>
<div id="open-table"></div>

<h2>Historial de Apuestas Cerradas</h2>
<div id="closed-table"></div>

<hr class="divider">

<div class="grid2">
  <div>
    <h2>Feed en Tiempo Real (últimos eventos)</h2>
    <div id="feed">Cargando…</div>
  </div>
  <div>
    <h2>Oportunidades Rechazadas (24h)</h2>
    <div id="skips-table"></div>
  </div>
</div>

<hr class="divider">

<div class="grid2">
  <div><h2>Calibración (predicho vs real)</h2><canvas id="calib-chart" height="220"></canvas></div>
  <div><h2>Razones de rechazo (24h)</h2><canvas id="skips-chart" height="220"></canvas></div>
</div>

<h2>Fuentes Meteorológicas</h2>
<div id="sources-table"></div>

<!-- ============================================================
     CRYPTO-LAG MAKER BOT SECTION
     Hidden by default; revealed when /api/crypto_lag/health says
     tables_present=true. The first <section> (data-variant="main")
     is the visible template; on load the JS clones it for every
     extra variant configured in config.yaml under crypto_lag.variants
     (e.g. "permissive") and stacks them below.

     IDs inside the section have been replaced by data-cl="<role>"
     attributes so the JS can resolve elements per-variant via
     `section.querySelector('[data-cl="role"]')`.
     ============================================================ -->
<div id="cl-sections-container">
<section class="crypto-lag-section" data-variant="main" id="crypto-lag-section" style="display:none">
  <hr class="divider">
  <div class="cl-header">
    <h2 style="margin-top:6px">
      <svg class="cl-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M13 2 3 14h7l-1 8 10-12h-7z" fill="currentColor"/>
      </svg>
      Crypto-Lag MAKER Bot
      <span data-cl="variant-tag" class="cl-variant-tag"></span>
      <span data-cl="status" class="cl-status-badge" aria-live="polite"></span>
    </h2>
    <div data-cl="symbols" class="cl-sub muted">cargando…</div>
  </div>

  <div class="cards cl-kpi-cards">
    <div class="card" data-cl="kpi-pnl"><div class="label">P&amp;L Total</div><div class="value muted">—</div></div>
    <div class="card" data-cl="kpi-fills"><div class="label">Fills 24h</div><div class="value muted">—</div></div>
    <div class="card" data-cl="kpi-wr"><div class="label">Win Rate 24h</div><div class="value muted">—</div></div>
    <div class="card" data-cl="kpi-cap"><div class="label">Capital Usado</div><div class="value muted">—</div></div>
  </div>

  <div class="cl-grid">
    <div class="cl-chart-card">
      <div class="cl-chart-title">Drift de precio: Binance vs Polymarket implícito (30 min)</div>
      <div class="cl-chart-body"><canvas data-cl="chart-price"></canvas></div>
    </div>
    <div class="cl-chart-card">
      <div class="cl-chart-title">Probabilidad: p_model (verde) vs poly_mid (azul punteado)</div>
      <div class="cl-chart-body"><canvas data-cl="chart-prob"></canvas></div>
    </div>
    <div class="cl-chart-card">
      <div class="cl-chart-title">P&amp;L acumulado (24h)</div>
      <div class="cl-chart-body" style="position:relative">
        <canvas data-cl="chart-pnl"></canvas>
        <div data-cl="pnl-empty" class="cl-empty-overlay">Sin closes todavía.</div>
      </div>
    </div>
    <div class="cl-chart-card">
      <div class="cl-chart-title">Distribución de decisiones (24h)</div>
      <div class="cl-chart-body cl-donut-body"><canvas data-cl="chart-decisions"></canvas></div>
    </div>
  </div>

  <div class="grid2" style="margin-top:14px">
    <div>
      <h2>Fills recientes</h2>
      <div data-cl="fills-table"></div>
    </div>
    <div>
      <h2>Closes recientes</h2>
      <div data-cl="closes-table"></div>
    </div>
  </div>
</section>
</div>

<script>
const $ = id => document.getElementById(id);
async function fetchJSON(url){try{const r=await fetch(url);return r.ok?r.json():null}catch{return null}}

function pctClass(v){return v>0?'pos':v<0?'neg':'muted'}
function badge(label,cls){return `<span class="badge badge-${cls}">${label}</span>`}

function riskDot(r){
  if(r>=75)return'dot-green';
  if(r>=50)return'dot-amber';
  return'dot-red';
}

function formatTime(ts){
  if(!ts)return'-';
  const d=new Date(ts);
  return d.toLocaleTimeString('es-ES',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function formatDT(ts){
  if(!ts)return'-';
  return new Date(ts).toLocaleString('es-ES',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
}

let calibChart=null, skipsChart=null;

async function refresh(){
  const [h, portfolio, cal, skips, trades, events] = await Promise.all([
    fetchJSON('/api/health'),
    fetchJSON('/api/portfolio'),
    fetchJSON('/api/calibration'),
    fetchJSON('/api/skips'),
    fetchJSON('/api/bets/open').then(d=>d||{open:[]}),
    fetchJSON('/api/events/recent?n=80'),
  ]);
  const closedData = await fetchJSON('/api/trades');
  if(!h||!portfolio)return;

  $('last-refresh').innerText='Actualizado '+formatTime(new Date().toISOString());

  // ── Bot state ───────────────────────────────────────────────────
  const aliveTxt = h.alive ? `🟢 ACTIVO (PID ${h.pid})` : '🔴 DETENIDO';
  const uptime = h.uptime_seconds
    ? `${Math.floor(h.uptime_seconds/3600)}h ${Math.floor((h.uptime_seconds%3600)/60)}m`
    : '-';
  $('mode').innerText = h.alive ? 'ACTIVO' : 'DETENIDO';
  $('mode').className = h.alive ? 'pos' : 'neg';

  const collapsed = portfolio.collapsed_count || 0;
  const healthCards = [
    {label:'Proceso', value:aliveTxt, cls: h.alive?'':'alert'},
    {label:'Tiempo activo', value:uptime},
    {label:'Lock file', value:h.lock_file_exists?'presente':'ausente'},
  ];
  $('health-cards').innerHTML = healthCards.map(c =>
    `<div class="card ${c.cls||''}"><div class="label">${c.label}</div><div class="value">${c.value}</div></div>`
  ).join('');

  // ── Alerts bar ──────────────────────────────────────────────────
  const alerts = [];
  if(collapsed>0) alerts.push(`<span class="badge badge-red">⚠ ${collapsed} posición${collapsed>1?'es':''} COLAPSADA${collapsed>1?'S':''} (precio &lt; 10% entrada)</span>`);
  if((portfolio.open_count||0)>0 && (portfolio.unrealized_pnl||0)<-20)
    alerts.push(`<span class="badge badge-amber">Pérdida no realizada: $${(portfolio.unrealized_pnl||0).toFixed(2)}</span>`);
  $('alerts-bar').innerHTML = alerts.length
    ? `<div style="background:#1a1509;border:1px solid #d2992244;border-radius:8px;padding:10px 14px;margin-bottom:14px;display:flex;gap:8px;flex-wrap:wrap">${alerts.join('')}</div>`
    : '';

  // ── Portfolio ───────────────────────────────────────────────────
  const roi = portfolio.roi_pct||0;
  const totalPnl = portfolio.total_pnl||0;
  const unrealized = portfolio.unrealized_pnl||0;
  const totalWithU = portfolio.total_pnl_including_unrealized||0;
  $('stats-cards').innerHTML = [
    {label:'Capital total',    value:'$'+(portfolio.current_portfolio_value||0).toFixed(2)},
    {label:'ROI realizado',    value:roi.toFixed(2)+'%', cls:pctClass(roi)},
    {label:'PnL realizado',    value:(totalPnl>=0?'+':'')+totalPnl.toFixed(2)+'$', cls:pctClass(totalPnl)},
    {label:'PnL no realizado', value:(unrealized>=0?'+':'')+unrealized.toFixed(2)+'$', cls:pctClass(unrealized)},
    {label:'PnL total (incl.unrealized)', value:(totalWithU>=0?'+':'')+totalWithU.toFixed(2)+'$', cls:pctClass(totalWithU)},
    {label:'Win rate',         value:(portfolio.win_rate||0).toFixed(1)+'%', cls:(portfolio.win_rate||0)>=50?'pos':'neg'},
    {label:'Trades cerrados',  value:portfolio.closed_trades||0},
    {label:'Posiciones abiertas', value:(portfolio.open_count||0)+(collapsed>0?` <span class="badge badge-red">${collapsed} COLLAPSED</span>`:'')},
    {label:'Exposición total', value:'$'+(portfolio.total_exposure||0).toFixed(2), cls:'warn-c'},
    {label:'Conf. media',      value:(portfolio.avg_confidence||0).toFixed(1)},
  ].map(c=>`<div class="card"><div class="label">${c.label}</div><div class="value ${c.cls||''}">${c.value}</div></div>`).join('');

  // ── Open positions ──────────────────────────────────────────────
  const open = trades.open||[];
  $('open-count').innerText = open.length ? `(${open.length})` : '';
  if(!open.length){
    $('open-table').innerHTML='<p style="color:#8b949e;font-size:13px">No hay posiciones abiertas.</p>';
  } else {
    const rows = open.map(t => {
      const entry = t.price_entry||0;
      const bid   = t.best_bid_current;
      const upnl  = t.unrealized_pnl||0;
      const risk  = t.risk_score||t.confidence_score||0;
      const isCollapsed = bid!=null && entry>0 && (bid/entry)<0.10;
      const isDegraded  = !!t.confidence_degraded;
      const divPct = bid!=null && entry>0 ? ((bid-entry)/entry*100) : null;

      const flags = [];
      if(isCollapsed) flags.push(badge('COLLAPSED','red'));
      else if(isDegraded) flags.push(badge('DEGRADED','amber'));
      if((t.avg_up_count||0)>0) flags.push(badge('×'+t.avg_up_count+' scale-in','green'));
      if(t.parent_trade_id) flags.push(`<span style="color:#8b949e;font-size:10px">↳#${t.parent_trade_id}</span>`);

      const ensText = t.ensemble_mean!=null
        ? `μ=${Number(t.ensemble_mean).toFixed(1)}, σ=${Number(t.ensemble_std||0).toFixed(2)}, n=${t.ensemble_sources_used||0}`
        : '-';
      const marketPrice = bid!=null ? bid.toFixed(4) : '-';
      const mktVsEnsemble = bid!=null && t.prob_real
        ? ` <span style="font-size:10px;color:${Math.abs((bid||0)-(t.prob_real||0))>0.35?'#f85149':'#8b949e'}">(${((bid-(t.prob_real||0))*100).toFixed(0)}pp vs ensemble)</span>`
        : '';

      const reasonSnippet = (t.reasoning||'').slice(0,120);

      return `<tr class="${isCollapsed?'row-alert':isDegraded?'row-warn':''}">
        <td>#${t.id}</td>
        <td>
          <div style="font-weight:600">${(t.market_question||'').slice(0,70)}</div>
          ${reasonSnippet?`<div class="reason-box">${reasonSnippet}${(t.reasoning||'').length>120?'…':''}</div>`:''}
        </td>
        <td><b>${t.side}</b></td>
        <td>${entry.toFixed(4)}</td>
        <td>${marketPrice}${mktVsEnsemble}</td>
        <td>$${(t.size_usdc||0).toFixed(2)}</td>
        <td><span class="dot ${riskDot(risk)}"></span>${risk.toFixed(0)}%</td>
        <td class="${pctClass(upnl)}"><b>$${upnl.toFixed(2)}</b>${divPct!=null?`<div style="font-size:10px">${divPct.toFixed(0)}%</div>`:''}</td>
        <td>${ensText}</td>
        <td>${flags.join(' ')||'-'}</td>
        <td><a href="/bet/${t.id}">detalle</a></td>
      </tr>`;
    });
    $('open-table').innerHTML = `<table><tr>
      <th>ID</th><th>Mercado + Razón</th><th>Lado</th><th>Entrada</th><th>Precio mkt</th>
      <th>Tamaño</th><th>Confianza</th><th>PnL no realiz.</th><th>Ensemble</th><th>Flags</th><th></th>
    </tr>${rows.join('')}</table>`;
  }

  // ── Closed trades ───────────────────────────────────────────────
  const closed = (closedData&&closedData.closed)||[];
  if(!closed.length){
    $('closed-table').innerHTML='<p style="color:#8b949e;font-size:13px">Sin apuestas cerradas.</p>';
  } else {
    const rows = closed.map(t => {
      const pnl = t.profit_loss||0;
      const isWin = pnl>0;
      const pnlPct = t.size_usdc ? (pnl/t.size_usdc*100) : null;
      const closeReason = t.close_reason||'';
      const badgeCR = closeReason.includes('win')
        ? badge('WIN','green')
        : closeReason.includes('emergency')
        ? badge('EMERGENCY','amber')
        : badge('LOSS','red');
      return `<tr>
        <td>#${t.id}</td>
        <td>${(t.market_question||'').slice(0,75)}</td>
        <td>${t.side}</td>
        <td>${(t.price_entry||0).toFixed(3)} → ${(t.closed_price||0).toFixed(3)}</td>
        <td class="${pctClass(pnl)}"><b>${pnl>=0?'+':''}${pnl.toFixed(2)}$</b>${pnlPct!=null?` (${pnl>=0?'+':''}${pnlPct.toFixed(0)}%)`:''}
          ${badgeCR}</td>
        <td>${(t.confidence_score||0).toFixed(0)}</td>
        <td style="color:#8b949e;font-size:11px">${closeReason}</td>
        <td><a href="/bet/${t.id}">detalle</a></td>
      </tr>`;
    });
    $('closed-table').innerHTML = `<table><tr>
      <th>ID</th><th>Mercado</th><th>Lado</th><th>Entrada → Salida</th>
      <th>PnL</th><th>Conf.</th><th>Razón cierre</th><th></th>
    </tr>${rows.join('')}</table>`;
  }

  // ── Live event feed ─────────────────────────────────────────────
  const evList = (events&&events.events)||[];
  if(!evList.length){
    $('feed').innerHTML='<div style="color:#8b949e">Sin eventos recientes.</div>';
  } else {
    $('feed').innerHTML = evList.map(ev => {
      const t = ev.type||'';
      const d = ev.data||{};
      let body = '';
      if(t==='TRADE_OPEN')
        body=`#${d.trade_id} ${d.side||''} @ ${(d.price||0).toFixed(4)} · $${(d.size_usdc||0).toFixed(2)} · conf=${d.confidence_score||'-'} · ${(d.question||'').slice(0,60)}`;
      else if(t==='TRADE_CLOSE')
        body=`#${d.trade_id} ${d.side||''} ${d.close_reason||''} · entry=${(d.entry_price||0).toFixed(4)} exit=${(d.exit_price||0).toFixed(4)} · PnL $${(d.pnl_usdc||0).toFixed(2)} (${((d.pnl_pct||0)*100).toFixed(0)}%)`;
      else if(t==='BET_REEVALUATED')
        body=`#${d.trade_id} prob=${((d.prob_real||0)*100).toFixed(0)}% mktprice=${(d.price_market||0).toFixed(4)} upnl=$${(d.unrealized_pnl||0).toFixed(2)} ${d.action||'HOLD'} ${(d.notes||'').slice(0,40)}`;
      else if(t==='OPPORTUNITY_SKIP')
        body=`${d.reason_code||''} · ${(d.market_question||'').slice(0,60)}`;
      else if(t==='MARKETS_SCANNED')
        body=`${d.total||0} mercados, ${d.weather||0} weather`;
      else if(t==='CYCLE_START'||t==='CYCLE_END')
        body=`ciclo #${ev.cycle_id||d.cycle_id||'-'}`;
      else body=JSON.stringify(d).slice(0,80);
      return `<div class="feed-row"><span class="feed-time">${formatTime(ev.timestamp)}</span><span class="feed-type type-${t}">${t}</span><span class="feed-body">${body}</span></div>`;
    }).join('');
  }

  // ── Rejected opportunities ──────────────────────────────────────
  const skipList = (skips&&skips.recent)||[];
  $('skips-table').innerHTML = skipList.length
    ? `<table><tr><th>Hora</th><th>Mercado</th><th>Razón</th><th>Conf</th><th>Std</th></tr>`+
      skipList.slice(0,20).map(s=>`<tr>
        <td>${formatDT(s.timestamp)}</td>
        <td>${(s.market_question||'').slice(0,55)}</td>
        <td>${s.reason_code||''}</td>
        <td>${s.confidence_score||'-'}</td>
        <td>${s.ensemble_std||'-'}</td>
      </tr>`).join('')+'</table>'
    : '<p style="color:#8b949e;font-size:13px">Sin rechazos recientes.</p>';

  // ── Charts ──────────────────────────────────────────────────────
  if(cal&&cal.buckets){
    const labels = cal.buckets.map(b=>(b.bin/10).toFixed(1)+'-'+((b.bin+1)/10).toFixed(1));
    if(calibChart){calibChart.destroy();calibChart=null;}
    calibChart = new Chart($('calib-chart'),{
      type:'bar',
      data:{labels,datasets:[
        {label:'Predicho',data:cal.buckets.map(b=>b.predicted_avg),backgroundColor:'#58a6ff88'},
        {label:'WR real', data:cal.buckets.map(b=>b.actual_avg),  backgroundColor:'#3fb95088'},
      ]},
      options:{scales:{y:{beginAtZero:true,max:1}},plugins:{legend:{labels:{color:'#c9d1d9'}}}},
    });
  }
  if(skips&&skips.summary&&skips.summary.length){
    if(skipsChart){skipsChart.destroy();skipsChart=null;}
    skipsChart = new Chart($('skips-chart'),{
      type:'bar',
      data:{
        labels:skips.summary.map(s=>s.reason_code),
        datasets:[{label:'Rechazos',data:skips.summary.map(s=>s.n),backgroundColor:'#d29922aa'}],
      },
      options:{indexAxis:'y',plugins:{legend:{labels:{color:'#c9d1d9'}}}},
    });
  }

  // ── Sources ─────────────────────────────────────────────────────
  const srcData = await fetchJSON('/api/sources');
  const srcRows = (srcData&&srcData.sources)||[];
  $('sources-table').innerHTML = srcRows.length
    ? `<table><tr><th>Fuente</th><th>Métrica</th><th>Trades</th><th>Brier</th><th>MAE</th><th>Fallos consec.</th><th>Estado</th></tr>`+
      srcRows.map(r=>{
        const fails=r.consecutive_failures||0;
        const dot=fails>=3?'dot-red':(r.brier_score>0.3?'dot-amber':'dot-green');
        return `<tr><td>${r.source_name}</td><td>${r.metric}</td><td>${r.trades_used||0}</td>
          <td>${(r.brier_score||0).toFixed(3)}</td><td>${(r.mae||0).toFixed(2)}</td>
          <td>${fails}</td><td><span class="dot ${dot}"></span>${r.last_updated?new Date(r.last_updated).toLocaleString():'-'}</td></tr>`;
      }).join('')+'</table>'
    : '<p style="color:#8b949e;font-size:13px">Sin datos de fuentes todavía.</p>';
}

refresh();
setInterval(refresh, 30000);

/* =============================================================
   CRYPTO-LAG MAKER BOT — sección añadida 30-Apr-2026
   v2 (May 2026): multi-variant. Una sección por variant configurada
   en config.yaml `crypto_lag.variants:`. Cada sección es un clon del
   template y se refresca de forma independiente, filtrando por
   `?variant=<name>` en cada endpoint.
   ============================================================= */
const CL = {
  // charts indexed by variant + role: CL.charts['main']['px']
  charts: {},
  symbolColors: {
    BTCUSDT:  { line:'#f7931a', fill:'rgba(247,147,26,0.15)' },
    ETHUSDT:  { line:'#8b9eff', fill:'rgba(139,158,255,0.15)' },
    SOLUSDT:  { line:'#14f195', fill:'rgba(20,241,149,0.15)' },
    BNBUSDT:  { line:'#f0b90b', fill:'rgba(240,185,11,0.15)' },
    XRPUSDT:  { line:'#23292f', fill:'rgba(120,120,120,0.15)' },
    DOGEUSDT: { line:'#c2a633', fill:'rgba(194,166,51,0.15)' },
  },
  fmtTime: t => new Date(t*1000).toLocaleTimeString(),
  fmtUsd:  v => (v>=0?'+':'') + '$' + Number(v).toFixed(2),
  // discovered variants (ordered) — populated on startup
  variants: ['main'],
  bound: false,  // becomes true after we've cloned + bound all sections
};

const clCommonOpts = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: {
    legend: { labels: { color:'#c9d1d9', font:{size:11}, boxWidth:10 } },
    tooltip: { mode:'index', intersect:false, backgroundColor:'#161b22', titleColor:'#79c0ff', bodyColor:'#e6edf3', borderColor:'#30363d', borderWidth:1 },
  },
  scales: {
    x: { ticks: { color:'#8b949e', font:{size:10}, maxRotation:0, autoSkipPadding:20 }, grid: { color:'#21262d' } },
    y: { ticks: { color:'#8b949e', font:{size:10} }, grid: { color:'#21262d' } },
  },
  interaction: { mode:'index', intersect:false },
};

/** Get the section element for a variant. */
function clSection(variant){
  return document.querySelector(`section.crypto-lag-section[data-variant="${variant}"]`);
}

/** Lookup an element inside a variant's section by its data-cl role. */
function clEl(section, role){
  return section.querySelector(`[data-cl="${role}"]`);
}

/** Store / replace a chart for (variant, role). */
function clEnsureChart(variant, role, ctx, cfg){
  CL.charts[variant] = CL.charts[variant] || {};
  const old = CL.charts[variant][role];
  if(old){ try{ old.destroy(); }catch(_){} }
  CL.charts[variant][role] = new Chart(ctx, cfg);
  return CL.charts[variant][role];
}

/** Clone the main section template for an extra variant and append it. */
function clCloneSectionForVariant(variant){
  if(variant === 'main') return;
  if(clSection(variant)) return;  // already cloned
  const tpl = clSection('main');
  if(!tpl) return;
  const clone = tpl.cloneNode(true);
  clone.setAttribute('data-variant', variant);
  clone.id = `crypto-lag-section-${variant}`;
  // Reset any chart canvases in the clone — Chart.js will (re)attach on next render.
  clone.querySelectorAll('canvas').forEach(c => { c.removeAttribute('id'); });
  // Clear the dynamic table containers so we don't carry main's HTML over.
  clone.querySelectorAll('[data-cl="fills-table"], [data-cl="closes-table"]').forEach(c => {
    c.innerHTML = '';
  });
  // Reset KPI cards to their loading state
  ['kpi-pnl','kpi-fills','kpi-wr','kpi-cap'].forEach(role => {
    const el = clEl(clone, role);
    if(!el) return;
    const labelMap = {
      'kpi-pnl':'P&L Total', 'kpi-fills':'Fills 24h',
      'kpi-wr':'Win Rate 24h', 'kpi-cap':'Capital Usado',
    };
    el.innerHTML =
      `<div class="label">${labelMap[role]}</div><div class="value muted">—</div>`;
  });
  tpl.parentNode.appendChild(clone);
}

/** Decorate the variant tag in the header. */
function clSetVariantTag(section, variant){
  const tag = clEl(section, 'variant-tag');
  if(!tag) return;
  tag.setAttribute('data-variant', variant);
  // Friendly labels
  const labels = { main: 'estricto', permissive: 'permisivo (validación)' };
  tag.textContent = labels[variant] || variant;
}

async function refreshCryptoLag(variant){
  variant = variant || 'main';
  const section = clSection(variant);
  if(!section) return;
  try {
    const h = await fetch(`/api/crypto_lag/health?variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    if(!h || !h.tables_present){ section.style.display='none'; return; }
    section.style.display='block';
    clSetVariantTag(section, variant);

    // Status badge
    const status = clEl(section, 'status');
    if(!h.enabled){
      status.innerHTML = '<span class="badge badge-amber">DISABLED</span>';
    } else if(h.halted){
      status.innerHTML = '<span class="badge badge-red">HALTED</span>';
    } else {
      const ageOk = h.last_snapshot_ts && (h.now - h.last_snapshot_ts) < 60;
      if(!ageOk){
        status.innerHTML = '<span class="badge badge-amber">IDLE</span>';
      } else if(h.last_decision === 'PRE_EVENT' || h.recent_only_heartbeat){
        status.innerHTML = '<span class="badge badge-blue">WAITING</span>'
          + '<span style="margin-left:6px;font-size:10px;color:#8b949e">esperando event window</span>';
      } else {
        status.innerHTML = '<span class="badge badge-green">LIVE</span>';
      }
    }
    clEl(section, 'symbols').textContent =
      `${h.symbols.join(' · ')} · ${(h.capital_pct*100).toFixed(0)}% bankroll`;

    // KPIs
    const k = await fetch(`/api/crypto_lag/kpis?variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    if(k && k.available){
      const pnlClass = k.pnl_total_usdc >= 0 ? 'pos' : 'neg';
      clEl(section, 'kpi-pnl').innerHTML =
        `<div class="label">P&amp;L Total</div><div class="value ${pnlClass}">${CL.fmtUsd(k.pnl_total_usdc)}</div>
         <div class="kpi-sub">24h: <span class="${k.pnl_24h_usdc>=0?'pos':'neg'}">${CL.fmtUsd(k.pnl_24h_usdc)}</span></div>`;
      clEl(section, 'kpi-fills').innerHTML =
        `<div class="label">Fills 24h</div><div class="value">${k.fills_24h}</div>
         <div class="kpi-sub">1h: ${k.fills_1h}${k.adverse_rate_24h!=null?` · adverse ${(k.adverse_rate_24h*100).toFixed(0)}%`:''}</div>`;
      const wr = k.win_rate_24h;
      clEl(section, 'kpi-wr').innerHTML =
        `<div class="label">Win Rate 24h</div>
         <div class="value ${wr==null?'muted':(wr>=0.5?'pos':'neg')}">${wr==null?'—':(wr*100).toFixed(0)+'%'}</div>
         <div class="kpi-sub">${k.wins_24h}W / ${k.closes_24h}T</div>`;
      clEl(section, 'kpi-cap').innerHTML =
        `<div class="label">Capital Usado</div><div class="value">$${k.capital_used_usdc.toFixed(0)}</div>
         <div class="kpi-sub muted">acumulado en fills</div>`;
    }

    // Charts + tables
    const snap = await fetch(`/api/crypto_lag/snapshots?minutes=30&n=300&variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    clRenderPriceChart(variant, section, snap);
    clRenderProbChart(variant, section, snap);

    const pnl = await fetch(`/api/crypto_lag/pnl_series?variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    clRenderPnlChart(variant, section, pnl);

    const dec = await fetch(`/api/crypto_lag/decisions?variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    clRenderDecisionChart(variant, section, dec);

    const fills = await fetch(`/api/crypto_lag/fills?n=12&variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    clRenderFills(section, fills);
    const closes = await fetch(`/api/crypto_lag/closes?n=12&variant=${encodeURIComponent(variant)}`).then(r=>r.json());
    clRenderCloses(section, closes);
  } catch(e){ console.warn(`crypto_lag[${variant}] refresh:`, e); }
}

function clRenderPriceChart(variant, section, snap){
  const canvas = clEl(section, 'chart-price');
  const ctx = canvas.getContext('2d');
  if(!snap || !snap.available || !Object.keys(snap.by_symbol||{}).length){
    const old = (CL.charts[variant]||{}).px;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].px; }
    return;
  }
  const datasets = [];
  Object.entries(snap.by_symbol).forEach(([sym, rows])=>{
    if(!rows.length) return;
    const c = CL.symbolColors[sym] || { line:'#79c0ff', fill:'rgba(121,192,255,0.15)' };
    const base = rows[0].binance_mid || 1;
    datasets.push({
      label: sym + ' (Binance)',
      data: rows.map(r=>({ x: r.ts*1000, y: ((r.binance_mid/base)-1)*100 })),
      borderColor: c.line, backgroundColor: c.fill, borderWidth: 1.6,
      pointRadius: 0, tension: 0.25, fill: false,
    });
    datasets.push({
      label: sym + ' (Polymarket implied)',
      data: rows
        .filter(r => r.poly_mid && r.poly_mid > 0 && r.decision !== 'PRE_EVENT')
        .map(r => ({ x: r.ts*1000, y: (r.poly_mid - 0.5) * 0.4 })),
      borderColor: c.line, borderWidth: 1, borderDash: [4,3],
      pointRadius: 0, tension: 0.25, fill: false,
    });
  });
  clEnsureChart(variant, 'px', ctx, {
    type: 'line',
    data: { datasets },
    options: {
      ...clCommonOpts,
      scales: {
        x: { type:'time', time: { unit:'minute' }, ticks: clCommonOpts.scales.x.ticks, grid: clCommonOpts.scales.x.grid },
        y: { ticks: { color:'#8b949e', font:{size:10}, callback: v => v.toFixed(2)+'%' }, grid: { color:'#21262d' } },
      },
    },
  });
}

function clRenderProbChart(variant, section, snap){
  const canvas = clEl(section, 'chart-prob');
  const ctx = canvas.getContext('2d');
  if(!snap || !snap.available){
    const old = (CL.charts[variant]||{}).prob;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].prob; }
    return;
  }
  const entries = Object.entries(snap.by_symbol).sort((a,b)=>b[1].length-a[1].length);
  if(!entries.length){
    const old = (CL.charts[variant]||{}).prob;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].prob; }
    return;
  }
  const [sym, rows] = entries[0];
  const datasets = [
    {
      label: `p_model (${sym})`,
      data: rows.map(r=>({ x: r.ts*1000, y: r.p_model })),
      borderColor: '#3fb950', backgroundColor:'rgba(63,185,80,0.10)',
      borderWidth: 1.8, pointRadius: 0, tension: 0.2, fill: '+1',
    },
    {
      label: `poly_mid (${sym})`,
      data: rows.map(r=>({ x: r.ts*1000, y: r.poly_mid })),
      borderColor: '#58a6ff', borderWidth: 1.4, borderDash:[3,3],
      pointRadius: 0, tension: 0.2, fill: false,
    },
  ];
  clEnsureChart(variant, 'prob', ctx, {
    type: 'line',
    data: { datasets },
    options: {
      ...clCommonOpts,
      scales: {
        x: { type:'time', time: { unit:'minute' }, ticks: clCommonOpts.scales.x.ticks, grid: clCommonOpts.scales.x.grid },
        y: { min:0, max:1, ticks: { color:'#8b949e', font:{size:10}, callback: v => (v*100).toFixed(0)+'%' }, grid: { color:'#21262d' } },
      },
    },
  });
}

function clRenderPnlChart(variant, section, pnl){
  const canvas = clEl(section, 'chart-pnl');
  const ctx = canvas.getContext('2d');
  const empty = clEl(section, 'pnl-empty');
  if(!pnl || !pnl.available || !pnl.points.length){
    const old = (CL.charts[variant]||{}).pnl;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].pnl; }
    if(empty) empty.style.display='flex';
    return;
  }
  if(empty) empty.style.display='none';
  const lastPnl = pnl.points[pnl.points.length-1].pnl_cum;
  const color = lastPnl >= 0 ? '#3fb950' : '#f85149';
  const fillColor = lastPnl >= 0 ? 'rgba(63,185,80,0.18)' : 'rgba(248,81,73,0.18)';
  clEnsureChart(variant, 'pnl', ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: 'P&L Acumulado (24h)',
        data: pnl.points.map(p=>({ x: p.ts*1000, y: p.pnl_cum })),
        borderColor: color, backgroundColor: fillColor, borderWidth: 2,
        pointRadius: 0, tension: 0.15, fill: 'origin',
      }],
    },
    options: {
      ...clCommonOpts,
      scales: {
        x: { type:'time', time: { unit:'minute' }, ticks: clCommonOpts.scales.x.ticks, grid: clCommonOpts.scales.x.grid },
        y: { ticks: { color:'#8b949e', font:{size:10}, callback: v => '$'+v.toFixed(2) }, grid: { color:'#21262d' } },
      },
    },
  });
}

function clRenderDecisionChart(variant, section, dec){
  const canvas = clEl(section, 'chart-decisions');
  const ctx = canvas.getContext('2d');
  if(!dec || !dec.available){
    const old = (CL.charts[variant]||{}).dec;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].dec; }
    return;
  }
  const buckets = dec.buckets || {};
  const labels  = ['BID','ASK','BOTH','NONE','PRE_EVENT','GATED','NO_BOOK','STRIKE_CAPTURED','RESOLUTION_WINDOW'];
  const colors  = ['#3fb950','#f85149','#79c0ff','#6e7681','#d29922','#a371f7','#39c5cf','#ff9b5e','#bb8009'];
  const present = labels
    .map((l,i)=>({label:l, count: buckets[l] || 0, color: colors[i]}))
    .filter(x => x.count > 0);
  if(!present.length){
    const old = (CL.charts[variant]||{}).dec;
    if(old){ try{ old.destroy(); }catch(_){} delete CL.charts[variant].dec; }
    canvas.parentElement.innerHTML = '<div class="cl-empty">Sin decisiones en 24h.</div>';
    return;
  }
  clEnsureChart(variant, 'dec', ctx, {
    type: 'doughnut',
    data: {
      labels: present.map(x=>x.label),
      datasets: [{ data: present.map(x=>x.count), backgroundColor: present.map(x=>x.color), borderColor:'#0d1117', borderWidth: 2 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '62%',
      plugins: {
        legend: { position:'right', labels: { color:'#c9d1d9', font:{size:11}, boxWidth:10 } },
        tooltip: { backgroundColor:'#161b22', titleColor:'#79c0ff', bodyColor:'#e6edf3' },
      },
    },
  });
}

function clRenderFills(section, payload){
  const t = clEl(section, 'fills-table');
  const rows = (payload && payload.fills) || [];
  if(!rows.length){ t.innerHTML = '<p class="cl-empty">Sin fills todavía.</p>'; return; }
  t.innerHTML = '<table><tr><th>Hora</th><th>Símbolo</th><th>Lado</th><th>Outcome</th><th>Precio</th><th>USDC</th><th>Adverse</th></tr>'
    + rows.map(r=>{
        const sideCls = r.side==='BUY' ? 'badge-green' : 'badge-red';
        const adv = r.is_adverse ? '<span class="badge badge-amber">TOXIC</span>' : '<span class="muted">·</span>';
        return `<tr>
          <td class="muted">${CL.fmtTime(r.ts)}</td>
          <td><b>${r.symbol}</b></td>
          <td><span class="badge ${sideCls}">${r.side}</span></td>
          <td>${r.outcome}</td>
          <td>${(r.fill_price||r.price).toFixed(3)}</td>
          <td>$${(r.fill_size_usdc||0).toFixed(2)}</td>
          <td>${adv}</td>
        </tr>`;
      }).join('')
    + '</table>';
}

function clRenderCloses(section, payload){
  const t = clEl(section, 'closes-table');
  const rows = (payload && payload.closes) || [];
  if(!rows.length){ t.innerHTML = '<p class="cl-empty">Sin closes todavía.</p>'; return; }
  t.innerHTML = '<table><tr><th>Hora</th><th>Símbolo</th><th>P&amp;L</th><th>Final</th><th>Razón</th></tr>'
    + rows.map(r=>{
        const cls = r.realized_pnl_usdc >= 0 ? 'pos' : 'neg';
        return `<tr>
          <td class="muted">${CL.fmtTime(r.ts)}</td>
          <td><b>${r.symbol}</b></td>
          <td class="${cls}">${CL.fmtUsd(r.realized_pnl_usdc)}</td>
          <td>${(r.final_yes_price!=null ? (r.final_yes_price).toFixed(2) : '—')}</td>
          <td class="muted">${r.reason||''}</td>
        </tr>`;
      }).join('')
    + '</table>';
}

/** Discover variants from the server and clone the section for each extra one. */
async function clDiscoverAndBindVariants(){
  if(CL.bound) return;
  try {
    const res = await fetch('/api/crypto_lag/variants').then(r=>r.json());
    const list = (res && res.variants && res.variants.map(v=>v.name)) || ['main'];
    // Always include 'main' first (the visible template).
    const ordered = ['main', ...list.filter(v => v !== 'main')];
    CL.variants = ordered;
    ordered.forEach(v => { if(v !== 'main') clCloneSectionForVariant(v); });
    CL.bound = true;
  } catch(e) {
    console.warn('crypto_lag: variant discovery failed, using main only', e);
    CL.variants = ['main'];
    CL.bound = true;
  }
}

async function refreshAllCryptoLagVariants(){
  await clDiscoverAndBindVariants();
  // Refresh all variants in parallel — they hit different DB rows so there's
  // no contention.
  await Promise.all(CL.variants.map(v => refreshCryptoLag(v)));
}

// Respect prefers-reduced-motion: if true, lower polling rate.
const _clPrefersReduced = window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const _clInterval = _clPrefersReduced ? 15000 : 5000;
refreshAllCryptoLagVariants();
setInterval(refreshAllCryptoLagVariants, _clInterval);
</script>
</body>
</html>
"""


BET_DETAIL_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bet #{{trade_id}} — Weather Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background:#0f1117; color:#e1e4e8; margin:20px; }
  h1,h2 { color:#58a6ff; }
  h1 { font-size:20px; border-bottom:1px solid #2d333b; padding-bottom:8px; }
  h2 { font-size:14px; margin:18px 0 8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:10px; }
  .card { background:#161b22; border:1px solid #2d333b; border-radius:10px; padding:12px; }
  .label { font-size:11px; color:#8b949e; text-transform:uppercase; }
  .value { font-size:18px; font-weight:700; margin-top:4px; }
  table { width:100%; border-collapse:collapse; background:#161b22; font-size:12px; }
  th { background:#1c2129; padding:6px; text-align:left; color:#8b949e; }
  td { padding:6px; border-top:1px solid #2d333b; }
  .pos { color:#3fb950; } .neg { color:#f85149; }
  a { color:#58a6ff; }
  canvas { background:#161b22; border-radius:8px; padding:10px; }
</style>
</head>
<body>
<p><a href="/">&larr; back</a></p>
<h1 id="title">Loading bet #{{trade_id}}…</h1>
<div class="grid" id="summary"></div>

<h2>Latest forecast by source</h2>
<div id="sources-table"></div>

<h2>Risk score over time</h2>
<canvas id="risk-chart" height="220"></canvas>

<h2>Evaluation history</h2>
<div id="evals-table"></div>

<script>
async function load() {
  const r = await fetch('/api/bet/{{trade_id}}');
  if (!r.ok) { document.getElementById('title').innerText = 'Bet not found'; return; }
  const d = await r.json();
  const t = d.trade;
  const degraded = !!t.confidence_degraded;
  const avgUpCount = t.avg_up_count || 0;
  const tagDegraded = degraded ? ` <span style="background:#b34;color:#fff;padding:2px 6px;border-radius:3px;font-size:0.75em">DEGRADED (hold to resolution)</span>` : '';
  const tagScaleIn = avgUpCount > 0 ? ` <span style="background:#2d6;color:#fff;padding:2px 6px;border-radius:3px;font-size:0.75em">SCALED-IN ×${avgUpCount}</span>` : '';
  document.getElementById('title').innerHTML =
    `Bet #${t.id} — ${t.side} @ ${(t.price_entry||0).toFixed(4)} on ${t.location||'?'}${tagDegraded}${tagScaleIn}`;
  const upnl = t.unrealized_pnl || 0;
  const cls = upnl > 0 ? 'pos' : upnl < 0 ? 'neg' : '';
  const markValue = ((t.best_bid_current||0) * (t.shares||0)).toFixed(2);
  document.getElementById('summary').innerHTML =
    `<div class="card"><div class="label">Market</div><div class="value">${(t.market_question||'').slice(0,80)}</div></div>
     <div class="card"><div class="label">Location</div><div class="value">${t.location||'?'}  (${t.lat||'-'}, ${t.lon||'-'})</div></div>
     <div class="card"><div class="label">Target</div><div class="value">${t.target_date||'?'}</div></div>
     <div class="card"><div class="label">Condition</div><div class="value">${t.condition_type||'?'} / ${t.metric||'?'}</div></div>
     <div class="card"><div class="label">Entry → Mark</div><div class="value">${(t.price_entry||0).toFixed(4)} → ${(t.best_bid_current||0).toFixed(4)}</div></div>
     <div class="card"><div class="label">Size</div><div class="value">$${(t.size_usdc||0).toFixed(2)} (${(t.shares||0).toFixed(2)} sh)</div></div>
     <div class="card"><div class="label">Risk score entry</div><div class="value">${(t.risk_score||t.confidence_score||0).toFixed(0)}</div></div>
     <div class="card"><div class="label">Prob real entry</div><div class="value">${((t.prob_real||0)*100).toFixed(1)}%</div></div>
     <div class="card"><div class="label">Min prob required</div><div class="value">${((t.min_prob_required||0)*100).toFixed(1)}%</div></div>
     <div class="card"><div class="label">Days ahead</div><div class="value">${t.days_ahead||'-'}</div></div>
     <div class="card"><div class="label">Scale-ins</div><div class="value">${avgUpCount}${degraded ? ' (blocked: degraded)' : ''}</div></div>
     <div class="card"><div class="label">Unrealized PnL</div><div class="value ${cls}">$${upnl.toFixed(2)}</div></div>
     <div class="card"><div class="label">Mark-to-market</div><div class="value">$${markValue}</div></div>`;

  // Latest forecast by source (group most recent timestamp)
  const snaps = d.forecast_snapshots || [];
  const bySrc = {};
  for (const s of snaps) {
    if (!bySrc[s.source] || s.timestamp > bySrc[s.source].timestamp) bySrc[s.source] = s;
  }
  const rows = Object.values(bySrc).sort((a,b) => a.source.localeCompare(b.source));
  const nums = rows.map(r => Number(r.forecast_value)).filter(v => Number.isFinite(v));
  const mean = nums.length ? (nums.reduce((a,b)=>a+b,0) / nums.length) : null;
  const variance = nums.length ? (nums.reduce((a,b)=>a+(b-mean)*(b-mean),0) / nums.length) : 0;
  const std = Math.sqrt(variance);
  const consensusRow = mean === null
    ? ''
    : `<tr style="background:#1a2a3a"><td><b>Consensus (mean / std)</b></td>
         <td><b>${mean.toFixed(2)}</b></td><td>${std.toFixed(2)}</td>
         <td>${nums.length} sources</td></tr>`;
  document.getElementById('sources-table').innerHTML =
    `<table><tr><th>Source</th><th>Forecast</th><th>Std</th><th>When</th></tr>` +
    (rows.length ? rows.map(r =>
      `<tr><td>${r.source}</td><td>${r.forecast_value ?? '-'}</td>
           <td>${r.forecast_std ?? '-'}</td>
           <td>${r.timestamp ? new Date(r.timestamp).toLocaleString() : '-'}</td></tr>`
    ).join('') : `<tr><td colspan="4">No snapshots yet</td></tr>`) + consensusRow + `</table>`;

  // Risk chart from evaluations
  const evs = (d.evaluations || []).slice().reverse();
  new Chart(document.getElementById('risk-chart'), {
    type: 'line',
    data: {
      labels: evs.map(e => new Date(e.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'Risk score',
        data: evs.map(e => e.risk_score),
        borderColor:'#58a6ff', fill:false, tension:0.2,
      }],
    },
    options: { scales: { y: { min: 0, max: 100 } } },
  });

  // Evaluation table
  document.getElementById('evals-table').innerHTML =
    `<table><tr><th>Time</th><th>Prob real</th><th>Risk</th><th>Action</th><th>Price mkt</th><th>uPnL</th><th>Notes</th></tr>` +
    (evs.length ? evs.slice().reverse().map(e =>
      `<tr><td>${new Date(e.timestamp).toLocaleString()}</td>
           <td>${((e.prob_real||0)*100).toFixed(1)}%</td>
           <td>${(e.risk_score||0).toFixed(0)}</td>
           <td><b>${e.action||'-'}</b></td>
           <td>${(e.price_market||0).toFixed(4)}</td>
           <td>${(e.unrealized_pnl||0).toFixed(2)}</td>
           <td>${(e.notes||'').slice(0,80)}</td></tr>`
    ).join('') : `<tr><td colspan="7">No evaluations yet</td></tr>`) + `</table>`;
}
load();
setInterval(load, 30000);
</script>
</body>
</html>
"""


@app.route("/bet/<int:trade_id>")
def bet_detail(trade_id: int):
    return render_template_string(BET_DETAIL_HTML, trade_id=trade_id)


@app.route("/")
@app.route("/index.html")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    port = CONFIG.get("dashboard", {}).get("port", 5000)
    app.run(host="0.0.0.0", port=port, debug=False)
