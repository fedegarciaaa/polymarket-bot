"""
Database module - SQLite persistence for trades, cycles, memory and analytics.
"""

import sqlite3
import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("polymarket_bot.database")


class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                mode TEXT NOT NULL DEFAULT 'DEMO',
                action TEXT NOT NULL,
                market_id TEXT,
                market_question TEXT,
                strategy TEXT,
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
                confidence TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mode TEXT NOT NULL,
                markets_scanned INTEGER DEFAULT 0,
                opportunities_found INTEGER DEFAULT 0,
                trades_executed INTEGER DEFAULT 0,
                portfolio_value REAL DEFAULT 0.0,
                pnl_cycle REAL DEFAULT 0.0,
                pnl_total REAL DEFAULT 0.0,
                capital_initial REAL DEFAULT 0.0,
                notes TEXT
            )
        """)

        cursor.execute("""
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

        cursor.execute("""
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

        cursor.execute("""
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

        self.conn.commit()

    # ---- Trades ----

    def log_trade(self, trade_data: dict) -> int:
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        trade_data.setdefault("timestamp", now)
        trade_data.setdefault("status", "OPEN")
        trade_data.setdefault("profit_loss", 0.0)

        columns = ", ".join(trade_data.keys())
        placeholders = ", ".join(["?"] * len(trade_data))
        values = list(trade_data.values())

        cursor.execute(
            f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        trade_id = cursor.lastrowid
        logger.info(f"Trade logged: id={trade_id} action={trade_data.get('action')} market={trade_data.get('market_question', '')[:50]}")
        return trade_id

    def get_open_positions(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY timestamp DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def close_position(self, trade_id: int, closed_price: float, profit_loss: float, status: str = "CLOSED"):
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE trades SET status = ?, profit_loss = ?, closed_at = ?, closed_price = ?
               WHERE id = ?""",
            (status, profit_loss, now, closed_price, trade_id),
        )
        self.conn.commit()
        logger.info(f"Position closed: id={trade_id} pnl={profit_loss:.4f} status={status}")

    def get_trade_by_id(self, trade_id: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_recent_closed_trades(self, n: int = 20) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE status IN ('CLOSED', 'SIMULATED') ORDER BY closed_at DESC LIMIT ?",
            (n,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trades_paginated(self, page: int = 1, per_page: int = 20) -> dict:
        cursor = self.conn.cursor()
        offset = (page - 1) * per_page

        cursor.execute("SELECT COUNT(*) as total FROM trades")
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (per_page, offset),
        )
        trades = [dict(row) for row in cursor.fetchall()]

        return {
            "trades": trades,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
        }

    # ---- Cycles ----

    def log_cycle(self, cycle_data: dict) -> int:
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cycle_data.setdefault("timestamp", now)

        columns = ", ".join(cycle_data.keys())
        placeholders = ", ".join(["?"] * len(cycle_data))
        values = list(cycle_data.values())

        cursor.execute(
            f"INSERT INTO cycles ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        cycle_id = cursor.lastrowid
        logger.info(f"Cycle logged: id={cycle_id} trades={cycle_data.get('trades_executed', 0)}")
        return cycle_id

    def get_cycle_count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM cycles")
        return cursor.fetchone()["cnt"]

    def get_cycles_history(self, limit: int = 100) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM cycles ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ---- Portfolio ----

    def get_portfolio_value(self, initial_capital: float) -> float:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(profit_loss), 0) as total_pnl FROM trades WHERE status IN ('CLOSED', 'SIMULATED')"
        )
        total_pnl = cursor.fetchone()["total_pnl"]

        cursor.execute(
            "SELECT COALESCE(SUM(size_usdc), 0) as exposure FROM trades WHERE status = 'OPEN'"
        )
        open_exposure = cursor.fetchone()["exposure"]

        return initial_capital + total_pnl - open_exposure

    def get_total_exposure(self) -> float:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(size_usdc), 0) as exposure FROM trades WHERE status = 'OPEN'"
        )
        return cursor.fetchone()["exposure"]

    # ---- Statistics ----

    def get_statistics(self, initial_capital: float) -> dict:
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM trades WHERE action = 'BUY'")
        total_trades = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) as wins FROM trades WHERE status IN ('CLOSED', 'SIMULATED') AND profit_loss > 0"
        )
        wins = cursor.fetchone()["wins"]

        cursor.execute(
            "SELECT COUNT(*) as closed FROM trades WHERE status IN ('CLOSED', 'SIMULATED')"
        )
        closed = cursor.fetchone()["closed"]

        win_rate = (wins / closed * 100) if closed > 0 else 0.0

        cursor.execute(
            "SELECT COALESCE(SUM(profit_loss), 0) as total_pnl FROM trades WHERE status IN ('CLOSED', 'SIMULATED')"
        )
        total_pnl = cursor.fetchone()["total_pnl"]

        cursor.execute(
            """SELECT strategy, COALESCE(SUM(profit_loss), 0) as pnl, COUNT(*) as count
               FROM trades WHERE status IN ('CLOSED', 'SIMULATED') AND action = 'BUY'
               GROUP BY strategy"""
        )
        pnl_by_strategy = {row["strategy"]: {"pnl": row["pnl"], "count": row["count"]} for row in cursor.fetchall()}

        cursor.execute(
            "SELECT MAX(profit_loss) as best FROM trades WHERE status IN ('CLOSED', 'SIMULATED')"
        )
        best_row = cursor.fetchone()
        best_trade = best_row["best"] if best_row and best_row["best"] is not None else 0.0

        cursor.execute(
            "SELECT MIN(profit_loss) as worst FROM trades WHERE status IN ('CLOSED', 'SIMULATED')"
        )
        worst_row = cursor.fetchone()
        worst_trade = worst_row["worst"] if worst_row and worst_row["worst"] is not None else 0.0

        # Max drawdown calculation from cycle history
        cursor.execute("SELECT portfolio_value FROM cycles ORDER BY timestamp ASC")
        values = [row["portfolio_value"] for row in cursor.fetchall()]
        max_drawdown = 0.0
        peak = initial_capital
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        portfolio_value = self.get_portfolio_value(initial_capital)
        roi_pct = ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0

        return {
            "total_trades": total_trades,
            "closed_trades": closed,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 4),
            "pnl_by_strategy": pnl_by_strategy,
            "max_drawdown": round(max_drawdown * 100, 2),
            "best_trade": round(best_trade, 4),
            "worst_trade": round(worst_trade, 4),
            "current_portfolio_value": round(portfolio_value, 2),
            "initial_capital": initial_capital,
            "roi_pct": round(roi_pct, 2),
            "open_positions": len(self.get_open_positions()),
        }

    def get_performance_summary(self, last_n_trades: int = 50) -> dict:
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT strategy, action, side, price_entry, profit_loss, ev_calculated,
                      prob_real_estimated, prob_market, confidence, status
               FROM trades WHERE status IN ('CLOSED', 'SIMULATED')
               ORDER BY closed_at DESC LIMIT ?""",
            (last_n_trades,),
        )
        trades = [dict(row) for row in cursor.fetchall()]

        if not trades:
            return {"trades": [], "avg_ev": 0, "avg_error": 0, "win_rate": 0}

        total = len(trades)
        wins = sum(1 for t in trades if t["profit_loss"] > 0)
        avg_ev = sum(t["ev_calculated"] or 0 for t in trades) / total

        errors = []
        for t in trades:
            if t["prob_real_estimated"] and t["profit_loss"] is not None:
                actual = 1.0 if t["profit_loss"] > 0 else 0.0
                errors.append(abs(t["prob_real_estimated"] - actual))
        avg_error = sum(errors) / len(errors) if errors else 0

        return {
            "trades": trades,
            "avg_ev": round(avg_ev, 4),
            "avg_estimation_error": round(avg_error, 4),
            "win_rate": round(wins / total * 100, 2) if total > 0 else 0,
            "total_closed": total,
        }

    # ---- Memory: Trade Analyses ----

    def log_analysis(self, analysis_data: dict) -> int:
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        analysis_data.setdefault("timestamp", now)

        columns = ", ".join(analysis_data.keys())
        placeholders = ", ".join(["?"] * len(analysis_data))
        values = list(analysis_data.values())

        cursor.execute(
            f"INSERT INTO trade_analyses ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_recent_analyses(self, n: int = 20) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trade_analyses ORDER BY timestamp DESC LIMIT ?", (n,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_unanalyzed_closed_trades(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT t.* FROM trades t
               LEFT JOIN trade_analyses ta ON t.id = ta.trade_id
               WHERE t.status IN ('CLOSED', 'SIMULATED') AND ta.id IS NULL AND t.action = 'BUY'
               ORDER BY t.closed_at DESC"""
        )
        return [dict(row) for row in cursor.fetchall()]

    # ---- Memory: Learned Rules ----

    def get_learned_rules(self, active_only: bool = True) -> list[dict]:
        cursor = self.conn.cursor()
        if active_only:
            cursor.execute(
                "SELECT * FROM learned_rules WHERE active = 1 ORDER BY effectiveness_pct DESC"
            )
        else:
            cursor.execute("SELECT * FROM learned_rules ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def add_learned_rule(self, rule_data: dict) -> int:
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        rule_data.setdefault("created_at", now)
        rule_data.setdefault("updated_at", now)

        columns = ", ".join(rule_data.keys())
        placeholders = ", ".join(["?"] * len(rule_data))
        values = list(rule_data.values())

        cursor.execute(
            f"INSERT INTO learned_rules ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        rule_id = cursor.lastrowid
        logger.info(f"New rule learned: id={rule_id} - {rule_data.get('rule_text', '')[:60]}")
        return rule_id

    def update_rule(self, rule_id: int, updates: dict):
        now = datetime.now(timezone.utc).isoformat()
        updates["updated_at"] = now
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [rule_id]

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE learned_rules SET {set_clause} WHERE id = ?",
            values,
        )
        self.conn.commit()

    def deactivate_rule(self, rule_id: int):
        self.update_rule(rule_id, {"active": 0})
        logger.info(f"Rule deactivated: id={rule_id}")

    # ---- Memory: Parameter Adjustments ----

    def log_parameter_adjustment(self, adjustment_data: dict) -> int:
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        adjustment_data.setdefault("timestamp", now)

        columns = ", ".join(adjustment_data.keys())
        placeholders = ", ".join(["?"] * len(adjustment_data))
        values = list(adjustment_data.values())

        cursor.execute(
            f"INSERT INTO parameter_adjustments ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        adj_id = cursor.lastrowid
        logger.info(f"Parameter adjustment logged: {adjustment_data.get('parameter_name')} {adjustment_data.get('old_value')} -> {adjustment_data.get('new_value')}")
        return adj_id

    def get_parameter_adjustments(self, limit: int = 20) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM parameter_adjustments ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()
        logger.info("Database connection closed")
