"""
Database - SQLite persistence for trades, cycles, confidence, sources and memory.
Schema enriched for Weather Bot v2:
  - trades:    + confidence_score, confidence_breakdown, ensemble_*, sources_json, vetos, side_wr
  - source_reliability:  rolling Brier / MAE per source
  - opportunity_skips:   every rejected opportunity with structured reason_code
  - market_resolutions:  actual outcome once a market closes (used for calibration)
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("polymarket_bot.database")


class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads + a single writer — needed because
        # the crypto_lag thread shares this Database with the weather scheduler.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
        self._create_tables()
        self._create_crypto_lag_tables()
        logger.info(f"Database initialized at {db_path}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                mode TEXT NOT NULL DEFAULT 'DEMO',
                action TEXT NOT NULL,
                market_id TEXT,
                market_question TEXT,
                strategy TEXT DEFAULT 'weather',
                side TEXT,
                price_entry REAL,
                size_usdc REAL,
                shares REAL,
                prob_real_estimated REAL,
                prob_market REAL,
                ev_calculated REAL,
                reasoning TEXT,
                status TEXT NOT NULL DEFAULT 'OPEN',
                profit_loss REAL DEFAULT 0.0,
                closed_at TEXT,
                closed_price REAL,
                confidence TEXT,                 -- human label HIGH/MED/LOW (kept for back-compat)
                confidence_score REAL,           -- numeric 0-100
                confidence_breakdown TEXT,       -- JSON: {component: score}
                ensemble_mean REAL,
                ensemble_std REAL,
                ensemble_sources_used INTEGER,
                sources_json TEXT,               -- JSON: {source_name: value}
                vetos_triggered TEXT,            -- JSON array of strings (empty for accepted)
                side_wr_at_entry REAL,           -- rolling WR of this side at entry time
                module TEXT DEFAULT 'weather',
                is_shadow INTEGER DEFAULT 0      -- 1 if trade not executed (shadow mode)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mode TEXT NOT NULL,
                markets_scanned INTEGER DEFAULT 0,
                weather_markets_found INTEGER DEFAULT 0,
                opportunities_found INTEGER DEFAULT 0,
                opportunities_skipped INTEGER DEFAULT 0,
                trades_executed INTEGER DEFAULT 0,
                portfolio_value REAL DEFAULT 0.0,
                pnl_cycle REAL DEFAULT 0.0,
                pnl_total REAL DEFAULT 0.0,
                capital_initial REAL DEFAULT 0.0,
                sources_available INTEGER DEFAULT 0,
                duration_seconds REAL DEFAULT 0.0,
                notes TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                analysis_text TEXT,
                prob_estimated REAL,
                prob_actual REAL,
                estimation_error REAL,
                bias_identified TEXT,
                lesson_extracted TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS learned_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_text TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                times_applied INTEGER DEFAULT 0,
                times_helpful INTEGER DEFAULT 0,
                effectiveness_pct REAL DEFAULT 0.0,
                active INTEGER DEFAULT 1
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS parameter_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                reason TEXT,
                applied INTEGER DEFAULT 0,
                performance_before REAL,
                performance_after REAL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS source_reliability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                metric TEXT NOT NULL,           -- temperature_c | precipitation_mm | ...
                trades_used INTEGER DEFAULT 0,
                brier_score REAL DEFAULT 0.0,
                mae REAL DEFAULT 0.0,
                mean_latency_ms REAL DEFAULT 0.0,
                consecutive_failures INTEGER DEFAULT 0,
                last_updated TEXT,
                UNIQUE(source_name, metric)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS opportunity_skips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                market_id TEXT,
                market_question TEXT,
                metric TEXT,
                reason_code TEXT NOT NULL,     -- enum: LOW_CONF, HIGH_STD, FEW_SOURCES, ...
                reason_detail TEXT,
                confidence_score REAL,
                ensemble_std REAL,
                sources_used INTEGER,
                prob_market REAL,
                prob_estimated REAL,
                edge REAL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS market_resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL UNIQUE,
                market_question TEXT,
                resolved_at TEXT,
                actual_outcome TEXT,            -- 'YES' | 'NO'
                actual_value REAL,              -- numeric value if applicable (e.g. actual temp)
                metric TEXT,
                threshold REAL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS process_lock (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pid INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                mode TEXT NOT NULL
            )
        """)

        # Per-cycle re-evaluation of open bets
        c.execute("""
            CREATE TABLE IF NOT EXISTS bet_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                cycle_id INTEGER,
                timestamp TEXT NOT NULL,
                prob_real REAL,
                risk_score REAL,
                action TEXT,                      -- HOLD | AVERAGE_UP | CLOSE
                price_market REAL,
                best_bid REAL,
                best_ask REAL,
                unrealized_pnl REAL,
                value_if_closed_now REAL,
                value_if_averaged_up_10 REAL,
                vetos TEXT,
                notes TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # Forecast snapshots per source — trazabilidad completa
        c.execute("""
            CREATE TABLE IF NOT EXISTS forecast_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                eval_id INTEGER,
                market_id TEXT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                metric TEXT,
                forecast_value REAL,
                forecast_std REAL,
                latency_ms INTEGER,
                FOREIGN KEY (trade_id) REFERENCES trades(id),
                FOREIGN KEY (eval_id) REFERENCES bet_evaluations(id)
            )
        """)

        # Cache of parsed weather market metadata
        c.execute("""
            CREATE TABLE IF NOT EXISTS weather_markets (
                market_id TEXT PRIMARY KEY,
                market_question TEXT,
                location TEXT,
                lat REAL,
                lon REAL,
                target_date TEXT,
                target_dt TEXT,
                condition_type TEXT,
                metric TEXT,
                threshold REAL,
                last_seen TEXT
            )
        """)

        # Indexes (idempotent)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_bet_eval_trade ON bet_evaluations(trade_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_trade ON forecast_snapshots(trade_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_opp_skips_time ON opportunity_skips(timestamp)")

        # Additive columns on trades — ignore if already present
        self._ensure_trade_columns()

        self.conn.commit()

    def _create_crypto_lag_tables(self):
        """Tables specific to the crypto-lag MAKER bot (Fase B). Idempotent.

        v3 (May 2026 — shadow-mode): added `variant TEXT NOT NULL DEFAULT 'main'`
        column to all three tables so we can run multiple instances of the bot
        side-by-side (e.g. main = strict simulator, permissive = optimistic
        simulator) without conflating their stats. Legacy DBs are migrated
        in `_ensure_crypto_lag_variant_column()` below.
        """
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_lag_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                symbol TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                market_slug TEXT,
                side TEXT NOT NULL,                  -- BUY | SELL
                outcome TEXT NOT NULL,               -- YES | NO
                price REAL NOT NULL,
                size_usdc REAL NOT NULL,
                status TEXT NOT NULL,                -- placed | canceled | filled | partially_filled
                fill_price REAL,
                fill_size_usdc REAL DEFAULT 0,
                is_adverse INTEGER DEFAULT 0,
                external_order_id TEXT,
                local_order_id TEXT,
                variant TEXT NOT NULL DEFAULT 'main'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_lag_state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                symbol TEXT NOT NULL,
                binance_mid REAL,
                sigma_realized REAL,
                book_imbalance REAL,
                p_model REAL,
                fair_mid REAL,
                poly_bid REAL,
                poly_ask REAL,
                poly_mid REAL,
                edge_bid REAL,
                edge_ask REAL,
                decision TEXT,
                variant TEXT NOT NULL DEFAULT 'main'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS crypto_lag_closes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                condition_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                realized_pnl_usdc REAL NOT NULL,
                final_yes_price REAL,
                reason TEXT,
                variant TEXT NOT NULL DEFAULT 'main'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_quotes_ts ON crypto_lag_quotes(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_quotes_cond ON crypto_lag_quotes(condition_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_snap_ts ON crypto_lag_state_snapshots(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_close_ts ON crypto_lag_closes(ts)")
        self.conn.commit()
        # Migrate legacy DBs that pre-date the `variant` column.
        self._ensure_crypto_lag_variant_column()
        # New indexes that require the variant column being present.
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_quotes_variant ON crypto_lag_quotes(variant, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_snap_variant ON crypto_lag_state_snapshots(variant, ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_clag_close_variant ON crypto_lag_closes(variant, ts)")
        self.conn.commit()

    def _ensure_crypto_lag_variant_column(self) -> None:
        """Add `variant` column to crypto_lag_* tables on legacy DBs.

        SQLite ALTER TABLE ADD COLUMN is non-destructive and instantaneous,
        so this is safe to run on every startup. We check existence first so
        the migration is idempotent.
        """
        c = self.conn.cursor()
        for table in ("crypto_lag_quotes", "crypto_lag_state_snapshots", "crypto_lag_closes"):
            try:
                c.execute(f"PRAGMA table_info({table})")
                cols = {row["name"] for row in c.fetchall()}
                if "variant" not in cols:
                    # Default 'main' so historical rows are attributed to the
                    # original (now legacy / strict) bot variant.
                    c.execute(
                        f"ALTER TABLE {table} ADD COLUMN variant TEXT NOT NULL DEFAULT 'main'"
                    )
                    logger.info(f"db migration: added 'variant' column to {table}")
            except sqlite3.OperationalError as exc:
                logger.warning(f"db migration {table}: {exc}")
        self.conn.commit()

    def log_crypto_lag_fill(self, fill, variant: str = "main") -> None:
        c = self.conn.cursor()
        c.execute(
            """INSERT INTO crypto_lag_quotes
               (ts, symbol, condition_id, side, outcome, price, size_usdc,
                status, fill_price, fill_size_usdc, is_adverse, local_order_id, variant)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                float(fill.ts), fill.symbol, fill.condition_id,
                fill.side, fill.outcome, float(fill.fill_price),
                float(fill.fill_size_usdc), "filled",
                float(fill.fill_price), float(fill.fill_size_usdc),
                1 if fill.is_adverse else 0, fill.order_id, str(variant),
            ),
        )
        self.conn.commit()

    def log_crypto_lag_snapshot(self, snap, variant: str = "main") -> None:
        c = self.conn.cursor()
        c.execute(
            """INSERT INTO crypto_lag_state_snapshots
               (ts, symbol, binance_mid, sigma_realized, book_imbalance,
                p_model, fair_mid, poly_bid, poly_ask, poly_mid,
                edge_bid, edge_ask, decision, variant)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                float(snap.ts), snap.symbol, float(snap.binance_mid),
                float(snap.sigma_realized), float(snap.book_imbalance),
                float(snap.p_model), float(snap.fair_mid),
                float(snap.poly_bid), float(snap.poly_ask), float(snap.poly_mid),
                float(snap.edge_bid), float(snap.edge_ask), snap.decision,
                str(variant),
            ),
        )
        self.conn.commit()

    def log_crypto_lag_close(self, ev, variant: str = "main") -> None:
        c = self.conn.cursor()
        c.execute(
            """INSERT INTO crypto_lag_closes
               (ts, condition_id, symbol, realized_pnl_usdc, final_yes_price, reason, variant)
               VALUES (?,?,?,?,?,?,?)""",
            (
                float(ev.ts), ev.condition_id, ev.symbol,
                float(ev.realized_pnl_usdc), float(ev.final_yes_price), ev.reason,
                str(variant),
            ),
        )
        self.conn.commit()

    def _ensure_trade_columns(self):
        """Add new trade columns in place without destroying data."""
        c = self.conn.cursor()
        c.execute("PRAGMA table_info(trades)")
        existing = {row["name"] for row in c.fetchall()}
        to_add = [
            ("risk_score", "REAL"),
            ("prob_real", "REAL"),
            ("min_prob_required", "REAL"),
            ("days_ahead", "INTEGER"),
            ("location", "TEXT"),
            ("lat", "REAL"),
            ("lon", "REAL"),
            ("target_date", "TEXT"),
            ("target_dt", "TEXT"),
            ("condition_type", "TEXT"),
            ("metric", "TEXT"),
            ("unrealized_pnl", "REAL"),
            ("best_bid_current", "REAL"),
            ("best_ask_current", "REAL"),
            ("last_mark_at", "TEXT"),
            ("parent_trade_id", "INTEGER"),       # for averaging-up entries
            ("condition_params", "TEXT"),         # JSON blob for reeval
            ("clob_token_ids", "TEXT"),           # JSON list of yes/no CLOB token ids
            ("token_yes", "TEXT"),                # resolved YES token id (outcomes-aware)
            ("token_no",  "TEXT"),                # resolved NO  token id (outcomes-aware)
            ("confidence_degraded", "INTEGER"),   # 1 once prob_real dipped below degrade_threshold
            ("degraded_at", "TEXT"),              # ISO-8601 timestamp of first degradation
            ("avg_up_count", "INTEGER"),          # scale-in entries taken on this parent trade
        ]
        for name, typ in to_add:
            if name not in existing:
                try:
                    c.execute(f"ALTER TABLE trades ADD COLUMN {name} {typ}")
                except sqlite3.OperationalError:
                    pass
        self.conn.commit()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def log_trade(self, trade_data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        trade_data.setdefault("timestamp", now)
        trade_data.setdefault("status", "OPEN")
        trade_data.setdefault("profit_loss", 0.0)
        trade_data.setdefault("module", "weather")
        trade_data.setdefault("strategy", "weather")
        trade_data.setdefault("action", "BUY")
        trade_data.setdefault("is_shadow", 0)

        # Serialize JSON-ish fields if dict/list provided
        for k in ("confidence_breakdown", "sources_json", "vetos_triggered"):
            if k in trade_data and not isinstance(trade_data[k], str) and trade_data[k] is not None:
                trade_data[k] = json.dumps(trade_data[k])

        c = self.conn.cursor()
        cols = ", ".join(trade_data.keys())
        placeholders = ", ".join(["?"] * len(trade_data))
        c.execute(f"INSERT INTO trades ({cols}) VALUES ({placeholders})", list(trade_data.values()))
        self.conn.commit()
        trade_id = c.lastrowid
        logger.info(
            f"Trade #{trade_id} logged: {trade_data.get('action')} {trade_data.get('side')} "
            f"'{(trade_data.get('market_question') or '')[:50]}' "
            f"confidence={trade_data.get('confidence_score')}"
        )
        return trade_id

    def get_open_positions(self) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE status = 'OPEN' AND is_shadow = 0 ORDER BY timestamp DESC")
        return [dict(r) for r in c.fetchall()]

    def get_open_positions_for_market(self, market_id: str) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' AND is_shadow = 0 AND market_id = ? ORDER BY timestamp DESC",
            (market_id,),
        )
        return [dict(r) for r in c.fetchall()]

    def close_position(self, trade_id: int, closed_price: float, profit_loss: float, status: str = "CLOSED"):
        now = datetime.now(timezone.utc).isoformat()
        c = self.conn.cursor()
        # Freeze mark-to-market fields at 0 so the dashboard/accounting doesn't
        # keep adding unrealized_pnl from a stale bid snapshot after close.
        c.execute(
            """UPDATE trades
                  SET status=?, profit_loss=?, closed_at=?, closed_price=?,
                      unrealized_pnl=0, best_bid_current=NULL, best_ask_current=NULL
                WHERE id=?""",
            (status, profit_loss, now, closed_price, trade_id),
        )
        self.conn.commit()
        logger.info(f"Trade #{trade_id} closed: pnl={profit_loss:.4f} status={status}")

    def get_trade_by_id(self, trade_id: int) -> Optional[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        r = c.fetchone()
        return dict(r) if r else None

    def has_recent_losing_close(self, market_id: str, side: str, hours: float) -> bool:
        """True if there is a CLOSED trade with pnl<0 for the same market+side
        whose closed_at is within the last `hours` hours. Used as anti-whipsaw gate."""
        if not market_id or not side or hours <= 0:
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        c = self.conn.cursor()
        c.execute(
            """SELECT 1 FROM trades
               WHERE market_id = ? AND side = ?
                 AND status IN ('CLOSED','SIMULATED')
                 AND profit_loss < 0
                 AND closed_at >= ?
               LIMIT 1""",
            (market_id, side, cutoff),
        )
        return c.fetchone() is not None

    def get_recent_closed_trades(self, n: int = 20) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            "SELECT * FROM trades WHERE status IN ('CLOSED','SIMULATED') ORDER BY closed_at DESC LIMIT ?",
            (n,),
        )
        return [dict(r) for r in c.fetchall()]

    def get_trades_paginated(self, page: int = 1, per_page: int = 20, shadow_only: bool = False) -> dict:
        c = self.conn.cursor()
        offset = (page - 1) * per_page
        where = "WHERE is_shadow = 1" if shadow_only else ""
        c.execute(f"SELECT COUNT(*) as total FROM trades {where}")
        total = c.fetchone()["total"]
        c.execute(
            f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (per_page, offset),
        )
        trades = [dict(r) for r in c.fetchall()]
        return {"trades": trades, "total": total, "page": page, "per_page": per_page,
                "pages": (total + per_page - 1) // per_page}

    # ------------------------------------------------------------------
    # Rolling WR by side - used by confidence engine veto
    # ------------------------------------------------------------------
    def get_side_rolling_winrate(self, side: str, last_n: int = 20) -> float:
        c = self.conn.cursor()
        c.execute(
            """SELECT profit_loss FROM trades
               WHERE side = ? AND status IN ('CLOSED','SIMULATED') AND is_shadow = 0
               ORDER BY closed_at DESC LIMIT ?""",
            (side, last_n),
        )
        rows = c.fetchall()
        if not rows:
            return 0.5  # neutral default
        wins = sum(1 for r in rows if (r["profit_loss"] or 0) > 0)
        return wins / len(rows)

    # ------------------------------------------------------------------
    # Cycles
    # ------------------------------------------------------------------
    def log_cycle(self, cycle_data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cycle_data.setdefault("timestamp", now)
        c = self.conn.cursor()
        cols = ", ".join(cycle_data.keys())
        placeholders = ", ".join(["?"] * len(cycle_data))
        c.execute(f"INSERT INTO cycles ({cols}) VALUES ({placeholders})", list(cycle_data.values()))
        self.conn.commit()
        return c.lastrowid

    def get_cycle_count(self) -> int:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM cycles")
        return c.fetchone()["cnt"]

    def get_cycles_history(self, limit: int = 100) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM cycles ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    def get_portfolio_value(self, initial_capital: float) -> float:
        c = self.conn.cursor()
        c.execute("SELECT COALESCE(SUM(profit_loss),0) as tp FROM trades WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0")
        total_pnl = c.fetchone()["tp"]
        c.execute("SELECT COALESCE(SUM(size_usdc),0) as exp FROM trades WHERE status='OPEN' AND is_shadow = 0")
        open_exposure = c.fetchone()["exp"]
        return initial_capital + total_pnl - open_exposure

    def get_total_exposure(self) -> float:
        c = self.conn.cursor()
        c.execute("SELECT COALESCE(SUM(size_usdc),0) as exp FROM trades WHERE status='OPEN' AND is_shadow = 0")
        return c.fetchone()["exp"]

    def get_exposure_for_market(self, market_id: str) -> float:
        c = self.conn.cursor()
        c.execute(
            "SELECT COALESCE(SUM(size_usdc),0) as exp FROM trades WHERE status='OPEN' AND is_shadow = 0 AND market_id = ?",
            (market_id,),
        )
        return c.fetchone()["exp"]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_statistics(self, initial_capital: float) -> dict:
        c = self.conn.cursor()

        c.execute("SELECT COUNT(*) as total FROM trades WHERE action='BUY' AND is_shadow = 0")
        total_trades = c.fetchone()["total"]

        c.execute("SELECT COUNT(*) as wins FROM trades WHERE status IN ('CLOSED','SIMULATED') AND profit_loss > 0 AND is_shadow = 0")
        wins = c.fetchone()["wins"]

        c.execute("SELECT COUNT(*) as closed FROM trades WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0")
        closed = c.fetchone()["closed"]

        win_rate = (wins / closed * 100) if closed > 0 else 0.0

        c.execute("SELECT COALESCE(SUM(profit_loss),0) as tp FROM trades WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0")
        total_pnl = c.fetchone()["tp"]

        c.execute("""SELECT side, COALESCE(SUM(profit_loss),0) as pnl, COUNT(*) as count,
                            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                     FROM trades WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0
                     GROUP BY side""")
        pnl_by_side = {
            row["side"]: {
                "pnl": row["pnl"],
                "count": row["count"],
                "win_rate": (row["wins"] / row["count"] * 100) if row["count"] > 0 else 0,
            }
            for row in c.fetchall()
        }

        c.execute("SELECT MAX(profit_loss) as best, MIN(profit_loss) as worst FROM trades WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0")
        row = c.fetchone()
        best_trade = row["best"] or 0.0
        worst_trade = row["worst"] or 0.0

        # Max drawdown
        c.execute("SELECT portfolio_value FROM cycles ORDER BY timestamp ASC")
        values = [r["portfolio_value"] for r in c.fetchall()]
        max_dd, peak = 0.0, initial_capital
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        portfolio_value = self.get_portfolio_value(initial_capital)
        roi_pct = ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0

        c.execute("SELECT AVG(confidence_score) as avgc FROM trades WHERE confidence_score IS NOT NULL AND is_shadow = 0")
        row = c.fetchone()
        avg_confidence = row["avgc"] if row and row["avgc"] is not None else 0.0

        return {
            "total_trades": total_trades,
            "closed_trades": closed,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 4),
            "pnl_by_side": pnl_by_side,
            "max_drawdown": round(max_dd * 100, 2),
            "best_trade": round(best_trade, 4),
            "worst_trade": round(worst_trade, 4),
            "current_portfolio_value": round(portfolio_value, 2),
            "initial_capital": initial_capital,
            "roi_pct": round(roi_pct, 2),
            "open_positions": len(self.get_open_positions()),
            "avg_confidence": round(avg_confidence, 1),
        }

    def get_calibration_buckets(self) -> list[dict]:
        """Return 10 buckets of prob_real_estimated vs actual outcome (for calibration chart)."""
        c = self.conn.cursor()
        c.execute(
            """SELECT prob_real_estimated, profit_loss FROM trades
               WHERE status IN ('CLOSED','SIMULATED') AND is_shadow = 0 AND prob_real_estimated IS NOT NULL"""
        )
        rows = c.fetchall()
        buckets = [{"bin": i, "predicted_avg": 0.0, "actual_avg": 0.0, "count": 0} for i in range(10)]
        bins = [[] for _ in range(10)]
        for r in rows:
            p = max(0.0, min(0.999999, r["prob_real_estimated"]))
            idx = int(p * 10)
            outcome = 1.0 if r["profit_loss"] > 0 else 0.0
            bins[idx].append((p, outcome))
        for i, b in enumerate(bins):
            if b:
                buckets[i]["count"] = len(b)
                buckets[i]["predicted_avg"] = round(sum(p for p, _ in b) / len(b), 3)
                buckets[i]["actual_avg"] = round(sum(o for _, o in b) / len(b), 3)
        return buckets

    # ------------------------------------------------------------------
    # Opportunity skips
    # ------------------------------------------------------------------
    def log_opportunity_skip(self, data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("timestamp", now)
        c = self.conn.cursor()
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        c.execute(f"INSERT INTO opportunity_skips ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()
        return c.lastrowid

    def get_recent_skips(self, limit: int = 50) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM opportunity_skips ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

    def get_skip_reasons_summary(self, hours: int = 24) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            """SELECT reason_code, COUNT(*) as n FROM opportunity_skips
               WHERE timestamp >= datetime('now', ?) GROUP BY reason_code ORDER BY n DESC""",
            (f"-{hours} hours",),
        )
        return [dict(r) for r in c.fetchall()]

    # ------------------------------------------------------------------
    # Source reliability
    # ------------------------------------------------------------------
    def upsert_source_reliability(self, source_name: str, metric: str, updates: dict):
        now = datetime.now(timezone.utc).isoformat()
        c = self.conn.cursor()
        c.execute(
            "SELECT id FROM source_reliability WHERE source_name = ? AND metric = ?",
            (source_name, metric),
        )
        row = c.fetchone()
        updates["last_updated"] = now
        if row:
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [row["id"]]
            c.execute(f"UPDATE source_reliability SET {set_clause} WHERE id = ?", values)
        else:
            updates["source_name"] = source_name
            updates["metric"] = metric
            cols = ", ".join(updates.keys())
            placeholders = ", ".join(["?"] * len(updates))
            c.execute(f"INSERT INTO source_reliability ({cols}) VALUES ({placeholders})", list(updates.values()))
        self.conn.commit()

    def get_source_reliability(self) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM source_reliability ORDER BY source_name, metric")
        return [dict(r) for r in c.fetchall()]

    def get_source_reliability_weights(self, metric: str) -> dict[str, float]:
        """Return {source_name: weight} where weight = 1 / (1 + brier_score).
        Sources without data use weight 1.0 implicitly (ensemble_forecast default)."""
        c = self.conn.cursor()
        c.execute(
            "SELECT source_name, brier_score, trades_used FROM source_reliability WHERE metric = ?",
            (metric,),
        )
        out: dict[str, float] = {}
        for r in c.fetchall():
            if (r["trades_used"] or 0) < 5:
                continue
            b = max(0.0, r["brier_score"] or 0.0)
            out[r["source_name"]] = 1.0 / (1.0 + b)
        return out

    # ------------------------------------------------------------------
    # Market resolutions
    # ------------------------------------------------------------------
    def log_market_resolution(self, market_id: str, data: dict):
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("resolved_at", now)
        data["market_id"] = market_id
        c = self.conn.cursor()
        c.execute("SELECT id FROM market_resolutions WHERE market_id = ?", (market_id,))
        row = c.fetchone()
        if row:
            set_clause = ", ".join([f"{k} = ?" for k in data.keys() if k != "market_id"])
            values = [v for k, v in data.items() if k != "market_id"] + [market_id]
            c.execute(f"UPDATE market_resolutions SET {set_clause} WHERE market_id = ?", values)
        else:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            c.execute(f"INSERT INTO market_resolutions ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Memory - analyses, rules, parameter adjustments
    # ------------------------------------------------------------------
    def log_analysis(self, data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("timestamp", now)
        c = self.conn.cursor()
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        c.execute(f"INSERT INTO trade_analyses ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()
        return c.lastrowid

    def get_recent_analyses(self, n: int = 20) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM trade_analyses ORDER BY timestamp DESC LIMIT ?", (n,))
        return [dict(r) for r in c.fetchall()]

    def get_unanalyzed_closed_trades(self) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            """SELECT t.* FROM trades t
               LEFT JOIN trade_analyses ta ON t.id = ta.trade_id
               WHERE t.status IN ('CLOSED','SIMULATED') AND ta.id IS NULL AND t.action='BUY' AND t.is_shadow = 0
               ORDER BY t.closed_at DESC"""
        )
        return [dict(r) for r in c.fetchall()]

    def get_learned_rules(self, active_only: bool = True) -> list[dict]:
        c = self.conn.cursor()
        if active_only:
            c.execute("SELECT * FROM learned_rules WHERE active = 1 ORDER BY effectiveness_pct DESC")
        else:
            c.execute("SELECT * FROM learned_rules ORDER BY created_at DESC")
        return [dict(r) for r in c.fetchall()]

    def add_learned_rule(self, rule_data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        rule_data.setdefault("created_at", now)
        rule_data.setdefault("updated_at", now)
        c = self.conn.cursor()
        cols = ", ".join(rule_data.keys())
        placeholders = ", ".join(["?"] * len(rule_data))
        c.execute(f"INSERT INTO learned_rules ({cols}) VALUES ({placeholders})", list(rule_data.values()))
        self.conn.commit()
        return c.lastrowid

    def update_rule(self, rule_id: int, updates: dict):
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [rule_id]
        c = self.conn.cursor()
        c.execute(f"UPDATE learned_rules SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

    def deactivate_rule(self, rule_id: int):
        self.update_rule(rule_id, {"active": 0})

    def log_parameter_adjustment(self, data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("timestamp", now)
        c = self.conn.cursor()
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        c.execute(f"INSERT INTO parameter_adjustments ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()
        return c.lastrowid

    def get_parameter_adjustments(self, limit: int = 20) -> list[dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM parameter_adjustments ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

    # ------------------------------------------------------------------
    # Bet evaluations (per-cycle monitoring of open bets)
    # ------------------------------------------------------------------
    def log_bet_evaluation(self, data: dict) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("timestamp", now)
        if "vetos" in data and not isinstance(data["vetos"], str) and data["vetos"] is not None:
            data["vetos"] = json.dumps(data["vetos"])
        c = self.conn.cursor()
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        c.execute(f"INSERT INTO bet_evaluations ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()
        return c.lastrowid

    def get_bet_evaluations(self, trade_id: int, limit: int = 100) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            "SELECT * FROM bet_evaluations WHERE trade_id = ? ORDER BY timestamp DESC LIMIT ?",
            (trade_id, limit),
        )
        return [dict(r) for r in c.fetchall()]

    # ------------------------------------------------------------------
    # Forecast snapshots
    # ------------------------------------------------------------------
    def log_forecast_snapshots(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        c = self.conn.cursor()
        n = 0
        for r in rows:
            r.setdefault("timestamp", now)
            cols = ", ".join(r.keys())
            placeholders = ", ".join(["?"] * len(r))
            c.execute(f"INSERT INTO forecast_snapshots ({cols}) VALUES ({placeholders})", list(r.values()))
            n += 1
        self.conn.commit()
        return n

    def get_forecast_snapshots(self, trade_id: int, limit: int = 200) -> list[dict]:
        c = self.conn.cursor()
        c.execute(
            "SELECT * FROM forecast_snapshots WHERE trade_id = ? ORDER BY timestamp DESC, source LIMIT ?",
            (trade_id, limit),
        )
        return [dict(r) for r in c.fetchall()]

    # ------------------------------------------------------------------
    # Weather markets metadata cache
    # ------------------------------------------------------------------
    def upsert_weather_market(self, market_id: str, data: dict):
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("last_seen", now)
        data["market_id"] = market_id
        c = self.conn.cursor()
        c.execute("SELECT 1 FROM weather_markets WHERE market_id = ?", (market_id,))
        if c.fetchone():
            sets = ", ".join(f"{k} = ?" for k in data.keys() if k != "market_id")
            vals = [v for k, v in data.items() if k != "market_id"] + [market_id]
            c.execute(f"UPDATE weather_markets SET {sets} WHERE market_id = ?", vals)
        else:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            c.execute(f"INSERT INTO weather_markets ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Mark-to-market updates for an open trade
    # ------------------------------------------------------------------
    def update_trade_mark(self, trade_id: int, updates: dict):
        if not updates:
            return
        updates["last_mark_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        c = self.conn.cursor()
        c.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", list(updates.values()) + [trade_id])
        self.conn.commit()

    def partial_close(self, trade_id: int, closed_price: float, shares_sold: float,
                      pnl_realized: float, reason: str = "partial_close"):
        """Reduce shares/size of a trade (partial sell) and accumulate realized PnL."""
        c = self.conn.cursor()
        c.execute("SELECT shares, size_usdc, profit_loss FROM trades WHERE id = ?", (trade_id,))
        row = c.fetchone()
        if not row:
            return
        new_shares = max(0.0, (row["shares"] or 0) - shares_sold)
        new_size = max(0.0, (row["size_usdc"] or 0) * (new_shares / max(row["shares"] or 1e-9, 1e-9)))
        new_pnl = (row["profit_loss"] or 0.0) + pnl_realized
        now = datetime.now(timezone.utc).isoformat()
        if new_shares <= 1e-6:
            c.execute(
                "UPDATE trades SET shares = 0, size_usdc = 0, status = 'CLOSED', "
                "profit_loss = ?, closed_at = ?, closed_price = ? WHERE id = ?",
                (new_pnl, now, closed_price, trade_id),
            )
        else:
            c.execute(
                "UPDATE trades SET shares = ?, size_usdc = ?, profit_loss = ? WHERE id = ?",
                (new_shares, new_size, new_pnl, trade_id),
            )
        self.conn.commit()
        logger.info(f"Trade #{trade_id} partial_close: shares sold={shares_sold} pnl+={pnl_realized:+.4f} ({reason})")

    # ------------------------------------------------------------------
    # Destructive reset helpers (used by state_reset on startup)
    # ------------------------------------------------------------------
    def wipe_runtime_tables(self, reset_rules: bool = False) -> dict:
        """Delete all rows from tables that represent a single bot run.
        Keeps schema, learned_rules (unless reset_rules), parameter_adjustments, source_reliability."""
        c = self.conn.cursor()
        counts: dict[str, int] = {}
        tables = [
            "trades", "cycles", "trade_analyses", "opportunity_skips",
            "bet_evaluations", "forecast_snapshots", "weather_markets",
            "market_resolutions",
        ]
        if reset_rules:
            tables.extend(["learned_rules", "parameter_adjustments", "source_reliability"])
        for t in tables:
            try:
                c.execute(f"SELECT COUNT(*) as n FROM {t}")
                counts[t] = c.fetchone()["n"]
                c.execute(f"DELETE FROM {t}")
            except sqlite3.OperationalError:
                counts[t] = 0
        try:
            c.execute("DELETE FROM sqlite_sequence WHERE name IN "
                      "('trades','cycles','trade_analyses','opportunity_skips',"
                      "'bet_evaluations','forecast_snapshots')")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()
        return counts

    # alias expected by memory module
    def get_unanalyzed_trades(self) -> list[dict]:
        return self.get_unanalyzed_closed_trades()

    # ------------------------------------------------------------------
    def close(self):
        self.conn.close()
