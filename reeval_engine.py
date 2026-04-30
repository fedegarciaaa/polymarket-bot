"""
Re-evaluation engine for open weather positions.

Política: HOLD-TO-RESOLUTION. Cada apuesta se mantiene hasta que Polymarket
resuelva el mercado — nunca se vende anticipadamente, independientemente de
caídas de prob_real, vetos duros o proximidad al cierre.

Runs every cycle (10 min). For each OPEN trade:
  1. Re-fetch ensemble from all 7 weather sources at stored (lat, lon, target_dt).
  2. Recompute prob_real and hard vetos (using confidence_engine).
  3. Mark-to-market from CLOB /book (best bid / best ask).
  4. Decide action (sin CLOSE):
        AVERAGE_UP — prob_real_new ≥ prob_real_entry + avg_up_prob_delta, la
                     posición NO está marcada como DEGRADED, el número de
                     re-apuestas previas < avg_up_max_count, y el exposure cap
                     lo permite. Añade `avg_up_size_fraction` (1/3) del tamaño
                     inicial.
        HOLD       — caso por defecto. Si prob_real_new < degrade_threshold,
                     además marcamos la posición con `confidence_degraded=1`
                     para bloquear futuros scale-ins en este mercado.
  5. Persistimos evaluaciones en bet_evaluations + forecast_snapshots para
     auditoría y dashboard (sparkline de risk_score, historial por fuente).

The engine is self-contained: given (db, api, weather_bot_strategy, risk_manager,
config) it operates on trades already persisted in the DB.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from strategies.weather_ensemble import ensemble_forecast
from strategies.confidence_engine import ConfidenceEngine
from structured_logger import get_logger

logger = logging.getLogger("polymarket_bot.reeval")


@dataclass
class ReevalDecision:
    action: str                # HOLD | AVERAGE_UP | CLOSE_EMERGENCY
    reason: str = ""
    prob_real_new: float = 0.0
    risk_score: float = 0.0
    vetos: list = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    unrealized_pnl: float = 0.0
    add_size_usdc: float = 0.0
    degraded: bool = False
    market_collapse: bool = False


class ReevalEngine:
    def __init__(self, config: dict, db, api, weather_strategy, risk_manager):
        self.config = config
        self.db = db
        self.api = api
        self.weather = weather_strategy
        self.risk = risk_manager
        self.confidence_engine = ConfidenceEngine(config)

        wb = dict(config.get("weather", {}) or {})
        # HOLD-TO-RESOLUTION policy: never sell. degrade_threshold marks the
        # position as DEGRADED (blocks future scale-ins) but the position is
        # held to Polymarket resolution.
        self.degrade_threshold = float(wb.get("degrade_threshold", 0.50))
        self.redegrade_clear_threshold = float(wb.get("redegrade_clear_threshold", 0.70))
        self.avg_up_prob_delta = float(wb.get("avg_up_prob_delta", 0.05))
        self.avg_up_size_fraction = float(wb.get("avg_up_size_fraction", 0.333))
        self.avg_up_max_count = int(wb.get("avg_up_max_count", 2))
        self.ensemble_weight = float(wb.get("ensemble_weight", 0.50))

        # Circuit-breaker: cierre anticipado cuando el precio colapsa a casi cero
        # y quedan pocas horas para resolución (el residual es recuperable).
        self.cb_enabled = bool(wb.get("circuit_breaker_enabled", False))
        self.cb_price_pct = float(wb.get("circuit_breaker_price_pct", 0.05))
        self.cb_hours_left = float(wb.get("circuit_breaker_hours_left", 24.0))

        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY") or ""
        self.mode = (config.get("bot", {}) or {}).get("mode", "DEMO").upper()
        self.initial_capital = float(
            (config.get("bot", {}) or {}).get("demo_capital", 1000.0)
        )

    # ── Entry point ────────────────────────────────────────────────
    async def run_cycle(self, cycle_id: Optional[int] = None) -> dict:
        try:
            open_trades = self.db.get_open_positions() or []
        except Exception as e:
            logger.error(f"get_open_positions failed: {e}")
            return {"evaluated": 0, "averaged_up": 0, "held": 0, "degraded": 0}

        weather_trades = [t for t in open_trades if (t.get("strategy") or "") == "weather_bot"]

        summary = {
            "evaluated": 0, "averaged_up": 0, "held": 0,
            "degraded": 0, "collapsed": 0, "emergency_closed": 0, "errors": 0,
        }
        tasks = [self._reevaluate_trade(t, cycle_id) for t in weather_trades]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                summary["errors"] += 1
                logger.warning(f"reeval exception: {r}")
                continue
            summary["evaluated"] += 1
            if r.action == "AVERAGE_UP":
                summary["averaged_up"] += 1
            elif r.action == "CLOSE_EMERGENCY":
                summary["emergency_closed"] += 1
            else:
                summary["held"] += 1
            if r.degraded:
                summary["degraded"] += 1
            if r.market_collapse:
                summary["collapsed"] += 1
        logger.info(
            f"Reeval cycle: {summary['evaluated']} open trades "
            f"(HOLD={summary['held']} AVG_UP={summary['averaged_up']} "
            f"COLLAPSED={summary['collapsed']} EMERGENCY_CLOSED={summary['emergency_closed']} "
            f"DEGRADED={summary['degraded']} ERR={summary['errors']})"
        )
        return summary

    # ── Single-trade reeval ────────────────────────────────────────
    async def _reevaluate_trade(self, trade: dict, cycle_id: Optional[int]) -> ReevalDecision:
        trade_id = trade.get("id")
        market_id = trade.get("market_id")
        side = (trade.get("side") or "YES").upper()
        price_entry = float(trade.get("price_entry") or 0.0)
        shares = float(trade.get("shares") or 0.0)
        prob_real_entry = float(trade.get("prob_real") or 0.0)
        size_usdc_initial = float(trade.get("size_usdc") or 0.0)
        trace_id = trade.get("trace_id") or get_logger().new_trace_id()

        # Fields stored at open time (populated by weather_bot enrichment)
        lat = trade.get("lat")
        lon = trade.get("lon")
        target_dt_str = trade.get("target_dt") or ""
        metric = trade.get("metric") or ""
        condition_type = trade.get("condition_type") or ""

        condition_params: dict = {}
        raw_params = trade.get("condition_params")
        if isinstance(raw_params, str) and raw_params:
            try:
                condition_params = json.loads(raw_params)
            except Exception:
                condition_params = {}
        elif isinstance(raw_params, dict):
            condition_params = raw_params

        decision = ReevalDecision(action="HOLD", vetos=[])

        # ── Mark-to-market from CLOB ───────────────────────────────
        token_ids = self._resolve_token_ids(trade)
        token_id = token_ids.get(side)
        book = self.api.get_orderbook(token_id) if token_id else None
        best_bid = book.get("best_bid") if book else None
        best_ask = book.get("best_ask") if book else None
        decision.best_bid = best_bid
        decision.best_ask = best_ask

        price_mark = best_bid if best_bid is not None else price_entry
        unrealized_pnl = (price_mark - price_entry) * shares if price_entry > 0 else 0.0
        decision.unrealized_pnl = unrealized_pnl

        try:
            self.db.update_trade_mark(trade_id, {
                "best_bid_current": best_bid,
                "best_ask_current": best_ask,
                "unrealized_pnl": unrealized_pnl,
            })
        except Exception as e:
            logger.debug(f"update_trade_mark failed ({trade_id}): {e}")

        try:
            get_logger().log_position_mark(
                trade_id=trade_id,
                market_id=market_id,
                best_bid=best_bid,
                best_ask=best_ask,
                unrealized_pnl=unrealized_pnl,
                value_if_closed_now=(best_bid or 0.0) * shares,
                trace_id=trace_id,
            )
        except Exception:
            pass

        # ── Re-run ensemble + vetos ─────────────────────────────────
        if not (lat and lon and target_dt_str and metric):
            decision.reason = "missing_stored_fields"
            decision.action = "HOLD"
            self._log_reeval(trace_id, trade, decision, price_mark)
            return decision

        try:
            target_dt = datetime.fromisoformat(target_dt_str.replace("Z", "+00:00"))
        except Exception:
            decision.reason = "bad_target_dt"
            self._log_reeval(trace_id, trade, decision, price_mark)
            return decision

        try:
            weights = self.db.get_source_reliability_weights(metric)
        except Exception:
            weights = None

        try:
            ensemble = await ensemble_forecast(
                self.weather.sources, float(lat), float(lon), target_dt, metric,
                trace_id=trace_id, market_id=market_id, source_weights=weights,
            )
        except Exception as e:
            logger.warning(f"reeval ensemble_forecast failed trade={trade_id}: {e}")
            decision.reason = "ensemble_failed"
            self._log_reeval(trace_id, trade, decision, price_mark)
            return decision

        prob_ensemble = self.weather._condition_probability(ensemble, condition_type, condition_params)
        if prob_ensemble is None:
            # HOLD-TO-RESOLUTION: never sell. Missing ensemble data just blocks
            # future scale-ins by marking the position as DEGRADED.
            decision.vetos = ["NO_ENSEMBLE_DATA"]
            decision.action = "HOLD"
            decision.reason = "NO_ENSEMBLE_DATA"
            decision.degraded = True
            self._mark_degraded(trade_id)
            self._log_reeval(trace_id, trade, decision, price_mark)
            return decision

        # Blend with market; pick prob for the side actually held.
        price_yes_now = price_mark if side == "YES" else (1.0 - (best_ask or price_entry))
        from strategies.weather_ensemble import blend_with_market
        prob_yes = blend_with_market(prob_ensemble, price_yes_now, self.ensemble_weight)
        our_prob_new = prob_yes if side == "YES" else (1.0 - prob_yes)
        risk_score_new = round(our_prob_new * 100, 1)
        decision.prob_real_new = our_prob_new
        decision.risk_score = risk_score_new

        hours_left = max(0.0, (target_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0)
        days_ahead = max(0, int(hours_left // 24))

        prob_market_now = price_mark
        br = self.confidence_engine.evaluate(
            ensemble_result=ensemble,
            prob_estimated=our_prob_new,
            prob_market=prob_market_now,
            liquidity=float(trade.get("liquidity") or 0),
            volume_24h=float(trade.get("volume_24h") or 0),
            hours_to_resolution=hours_left,
            side=side,
            is_pyramid=False,
            days_ahead=days_ahead,
        )
        decision.vetos = list(br.vetos)

        # ── Persist evaluation ──────────────────────────────────────
        try:
            snapshot_rows = []
            for src_name, src_val in (ensemble.per_source or {}).items():
                snapshot_rows.append({
                    "trade_id": trade_id,
                    "source": src_name,
                    "forecast_value": src_val,
                })
            if snapshot_rows:
                self.db.log_forecast_snapshots(snapshot_rows)
        except Exception as e:
            logger.debug(f"log_forecast_snapshots failed: {e}")

        # ── Detect market collapse ──────────────────────────────────
        # Si el precio de mercado cae por debajo de `cb_price_pct` × precio_entrada
        # la posición ha perdido >95% de su valor — loggear como MARKET_COLLAPSE.
        market_collapse = (
            price_entry > 0
            and price_mark is not None
            and price_mark < price_entry * self.cb_price_pct
        )
        if market_collapse:
            decision.market_collapse = True
            decision.vetos = list(decision.vetos or []) + ["MARKET_COLLAPSE"]
            logger.warning(
                f"MARKET_COLLAPSE trade={trade_id}: bid={price_mark:.4f} "
                f"entry={price_entry:.4f} ({price_mark/price_entry:.1%} of entry) "
                f"upnl=${unrealized_pnl:.2f} hours_left={hours_left:.1f}h"
            )

        # ── Decide ──────────────────────────────────────────────────
        # HOLD-TO-RESOLUTION política: nunca vendemos. Sólo decidimos si
        # añadir (AVERAGE_UP) o mantener (HOLD). Si prob_real_new cae bajo
        # degrade_threshold o aparece un hard veto marcamos degraded=True
        # para bloquear futuros scale-ins en este mercado.
        #
        # Excepción: circuit_breaker_enabled=true + colapso de precio +
        # pocas horas para resolución → CLOSE_EMERGENCY para recuperar el
        # residual antes de que el mercado resuelva a 0.
        hard_veto = any(v in br.vetos for v in (
            "NO_ENSEMBLE_DATA", "FEW_SOURCES", "HIGH_ENSEMBLE_STD", "EDGE_SUSPICIOUS",
        ))

        already_degraded = bool(trade.get("confidence_degraded"))
        newly_degraded = (our_prob_new < self.degrade_threshold) or hard_veto
        decision.degraded = already_degraded or newly_degraded

        # Circuit-breaker check (highest priority override)
        cb_trigger = (
            self.cb_enabled
            and market_collapse
            and hours_left <= self.cb_hours_left
            and decision.degraded
        )
        if cb_trigger:
            decision.action = "CLOSE_EMERGENCY"
            decision.reason = (
                f"circuit_breaker: price {price_mark:.4f} < {self.cb_price_pct:.0%}×entry "
                f"({hours_left:.1f}h left)"
            )
        else:
            can_scale_in = (
                not decision.degraded
                and int(trade.get("avg_up_count") or 0) < self.avg_up_max_count
                and our_prob_new >= prob_real_entry + self.avg_up_prob_delta
            )

            if can_scale_in:
                add_size = round(size_usdc_initial * self.avg_up_size_fraction, 2)
                if self._can_average_up(trade, add_size):
                    decision.action = "AVERAGE_UP"
                    decision.reason = f"prob_delta +{(our_prob_new - prob_real_entry):.2f}"
                    decision.add_size_usdc = add_size
                else:
                    decision.action = "HOLD"
                    decision.reason = "avg_up_capped_by_exposure"
            elif decision.degraded and not already_degraded:
                decision.action = "HOLD"
                if hard_veto:
                    decision.reason = f"degraded:hard_veto:{br.vetos}"
                else:
                    decision.reason = f"degraded:prob_below ({our_prob_new:.2f}<{self.degrade_threshold:.2f})"
            elif decision.degraded:
                decision.action = "HOLD"
                decision.reason = "held_degraded"
            else:
                decision.action = "HOLD"
                decision.reason = "within_band"

        # Persist degraded flag to trades row the first time it trips.
        if newly_degraded and not already_degraded:
            self._mark_degraded(trade_id)

        if decision.action == "AVERAGE_UP":
            await self._do_average_up(trade, decision, trace_id, prob_real_entry, price_mark)
        elif decision.action == "CLOSE_EMERGENCY":
            self._do_emergency_close(trade, decision, price_mark, price_entry, shares, trace_id)

        try:
            self.db.log_bet_evaluation({
                "trade_id": trade_id,
                "prob_real": our_prob_new,
                "confidence": risk_score_new,
                "risk_score": risk_score_new,
                "action": decision.action,
                "price_market": price_mark,
                "unrealized_pnl": unrealized_pnl,
                "notes": decision.reason[:500],
            })
        except Exception as e:
            logger.debug(f"log_bet_evaluation failed: {e}")

        self._log_reeval(trace_id, trade, decision, price_mark, hours_left)
        return decision

    def _do_emergency_close(
        self, trade: dict, decision: ReevalDecision,
        price_mark: float, price_entry: float, shares: float, trace_id: str,
    ) -> None:
        """Cerrar posición en modo DEMO cuando circuit_breaker dispara."""
        if self.mode != "DEMO":
            logger.warning(
                f"CLOSE_EMERGENCY trade={trade.get('id')} — only implemented in DEMO. "
                "In LIVE you would need to submit a sell order manually."
            )
            return
        trade_id = trade.get("id")
        pnl = (price_mark - price_entry) * shares
        try:
            self.db.close_position(trade_id, price_mark, pnl, "SIMULATED")
        except Exception as e:
            logger.error(f"emergency close db.close_position failed trade={trade_id}: {e}")
            return
        try:
            size_usdc = float(trade.get("size_usdc") or 0)
            pnl_pct = pnl / size_usdc if size_usdc else 0.0
            slog = get_logger()
            slog.log_trade_close(
                trace_id=trace_id,
                trade_id=trade_id,
                market_id=trade.get("market_id", ""),
                side=trade.get("side", ""),
                entry_price=price_entry,
                exit_price=price_mark,
                pnl_usdc=pnl,
                pnl_pct=pnl_pct,
                close_reason="emergency_close",
            )
        except Exception as e:
            logger.debug(f"emergency close slog failed trade={trade_id}: {e}")
        logger.warning(
            f"EMERGENCY CLOSE executed trade={trade_id}: "
            f"price={price_mark:.4f} pnl=${pnl:.2f} reason={decision.reason}"
        )

    # ── Helpers ────────────────────────────────────────────────────
    def _resolve_token_ids(self, trade: dict) -> dict:
        # Prefer the explicit token_yes / token_no that scan_markets now
        # populates from the market's `outcomes` array (correctly aligned even
        # if Gamma returns ["No","Yes"]). Fall back to positional parsing only
        # if those fields are absent (legacy trades opened before the fix).
        ty = trade.get("token_yes")
        tn = trade.get("token_no")
        if ty and tn:
            return {"YES": str(ty), "NO": str(tn)}

        raw = trade.get("clob_token_ids") or ""
        try:
            if isinstance(raw, str) and raw:
                ids = json.loads(raw)
            elif isinstance(raw, list):
                ids = raw
            else:
                return {}
        except Exception:
            return {}
        if not isinstance(ids, list) or len(ids) < 2:
            return {}
        return {"YES": str(ids[0]), "NO": str(ids[1])}

    def _can_average_up(self, trade: dict, add_size: float) -> bool:
        try:
            portfolio_value = float(self.db.get_portfolio_value(self.initial_capital) or 0.0)
        except Exception:
            portfolio_value = self.initial_capital
        if portfolio_value <= 0 or add_size <= 0:
            return False
        cap_pct = float(self.risk.max_exposure_per_market_pct)
        cap = portfolio_value * cap_pct
        try:
            same = self.db.get_open_positions_for_market(trade.get("market_id")) or []
        except Exception:
            same = []
        existing = sum(float(p.get("size_usdc") or 0) for p in same)
        return (existing + add_size) <= cap

    def _mark_degraded(self, trade_id) -> None:
        """Persist confidence_degraded=1 and degraded_at on the trades row."""
        if not trade_id:
            return
        try:
            self.db.update_trade_mark(trade_id, {
                "confidence_degraded": 1,
                "degraded_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.debug(f"mark_degraded failed ({trade_id}): {e}")

    async def _do_average_up(self, trade, decision: ReevalDecision, trace_id, prob_real_entry, price_mark):
        add_size = decision.add_size_usdc
        side = (trade.get("side") or "YES").upper()
        market_id = trade.get("market_id")
        entry_price = float(trade.get("price_entry") or 0.0)
        price = price_mark if price_mark > 0 else entry_price

        # Guard: don't average UP at a price below entry. If the best_bid of our
        # side has collapsed, the market disagrees with our thesis regardless of
        # what our ensemble says — buying more is averaging DOWN into a loser.
        if price < entry_price * 0.92:
            decision.action = "HOLD"
            decision.reason += f"|avg_up_blocked_price_below_entry({price:.3f}<{entry_price:.3f})"
            return

        if self.mode == "LIVE":
            try:
                _ = self.api.place_order_live(
                    market_id=market_id, side=side, size_usdc=add_size, price=price,
                    private_key=self.private_key, market_question=trade.get("market_question", ""),
                )
            except Exception as e:
                logger.warning(f"avg_up LIVE order failed: {e}")
                decision.reason += f"|live_order_failed:{e}"
                return
        else:
            _ = self.api.place_order_demo(
                market_id=market_id, side=side, size_usdc=add_size, price=price,
                market_question=trade.get("market_question", ""),
            )

        shares_added = add_size / price if price > 0 else 0.0
        try:
            new_id = self.db.log_trade({
                "cycle_id": None,
                "strategy": "weather_bot",
                "action": "BUY",
                "market_id": market_id,
                "market_question": trade.get("market_question", ""),
                "side": side,
                "price_entry": price,
                "size_usdc": add_size,
                "shares": shares_added,
                "confidence_score": decision.risk_score,
                "prob_real_estimated": decision.prob_real_new,
                "ev_calculated": 0.0,
                "status": "OPEN",
                "mode": self.mode,
                "risk_score": decision.risk_score,
                "prob_real": decision.prob_real_new,
                "days_ahead": trade.get("days_ahead"),
                "location": trade.get("location"),
                "lat": trade.get("lat"),
                "lon": trade.get("lon"),
                "target_date": trade.get("target_date"),
                "target_dt": trade.get("target_dt"),
                "condition_type": trade.get("condition_type"),
                "metric": trade.get("metric"),
                "parent_trade_id": trade.get("id"),
            })
        except Exception as e:
            logger.warning(f"avg_up DB log_trade failed: {e}")
            new_id = None

        # Bump avg_up_count on the parent so future cycles respect the cap.
        try:
            new_count = int(trade.get("avg_up_count") or 0) + 1
            self.db.update_trade_mark(trade.get("id"), {"avg_up_count": new_count})
        except Exception as e:
            logger.debug(f"avg_up_count bump failed: {e}")

        get_logger().log_bet_averaged_up(
            trace_id=trace_id,
            parent_trade_id=trade.get("id"),
            new_trade_id=new_id,
            market_id=market_id,
            side=side,
            add_price=price,
            add_size_usdc=add_size,
            prob_real_entry=prob_real_entry,
            prob_real_now=decision.prob_real_new,
        )

    def _log_reeval(
        self, trace_id, trade, decision: ReevalDecision, price_mark: float,
        hours_left: Optional[float] = None,
    ):
        try:
            get_logger().log_bet_reevaluated(
                trace_id=trace_id,
                trade_id=trade.get("id"),
                market_id=trade.get("market_id"),
                prob_real=decision.prob_real_new,
                prob_real_entry=float(trade.get("prob_real") or 0.0),
                risk_score=decision.risk_score,
                action=decision.action,
                vetos=decision.vetos,
                price_market=price_mark,
                unrealized_pnl=decision.unrealized_pnl,
                hours_to_resolution=hours_left,
                notes=decision.reason,
            )
        except Exception as e:
            logger.debug(f"log_bet_reevaluated failed: {e}")
