"""
Polymarket Trading Bot - Main Orchestrator
Usage: python main.py --mode demo
"""

import argparse
import logging
import os
import sys
import time
import yaml
import schedule
import colorlog
from datetime import datetime, timezone
from dotenv import load_dotenv

from database import Database
from polymarket_api import PolymarketAPI
from claude_agent import ClaudeAgent
from risk_manager import RiskManager
from memory import MemorySystem
from notifications import TelegramNotifier
from strategies.bonding import BondingStrategy
from strategies.arbitrage import ValueBettingStrategy, MomentumStrategy


def setup_logging(config: dict):
    log_level = config.get("logging", {}).get("level", "INFO")
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y-%m-%d')}.log")

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    ))

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger("polymarket_bot")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner(config: dict, mode: str):
    capital = config["bot"]["demo_capital"]
    interval = config["bot"]["cycle_interval_minutes"]
    model = config.get("claude", {}).get("model", "claude-sonnet-4-20250514")

    banner = f"""
╔══════════════════════════════════════════════════════════╗
║           POLYMARKET TRADING BOT v1.0                    ║
║══════════════════════════════════════════════════════════║
║  Modo:        {mode:<42}║
║  Capital:     ${capital:<41,.2f}║
║  Intervalo:   {interval} minutos{' ' * (35 - len(str(interval)))}║
║  Modelo:      {model:<42}║
║  Estrategias: Bonding + Value Betting + Momentum         ║
║  Memoria:     Auto-mejora activa                         ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


class TradingBot:
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode.upper()
        self.cycle_count = 0
        self.initial_capital = config["bot"]["demo_capital"]

        # Initialize components
        self.db = Database(config.get("database", {}).get("path", "data/bot.db"))
        self.api = PolymarketAPI(config)
        self.risk = RiskManager(config)
        self.notifier = TelegramNotifier()
        self.bonding = BondingStrategy(config)
        self.value_betting = ValueBettingStrategy(config)
        self.momentum = MomentumStrategy(config)
        self.memory = MemorySystem(self.db, config)

        # Claude agent (may fail if no API key)
        try:
            self.claude = ClaudeAgent(config)
        except ValueError as e:
            self.logger = logging.getLogger("polymarket_bot")
            self.logger.error(f"Claude agent initialization failed: {e}")
            self.logger.warning("Bot will run without Claude analysis (scanning only)")
            self.claude = None

        self.logger = logging.getLogger("polymarket_bot")
        self.logger.info(f"TradingBot initialized in {self.mode} mode")

    def get_portfolio_state(self, current_prices: dict = None) -> dict:
        portfolio_value = self.db.get_portfolio_value(self.initial_capital)
        total_exposure = self.db.get_total_exposure()
        stats = self.db.get_statistics(self.initial_capital)

        # Calculate mark-to-market unrealized P&L
        unrealized_pnl = 0.0
        open_positions = self.db.get_open_positions()
        if current_prices:
            for pos in open_positions:
                mid = pos.get("market_id")
                prices = current_prices.get(mid, {})
                side = pos.get("side", "YES")
                current_price = prices.get(f"price_{side.lower()}", pos.get("price_entry", 0))
                entry_price = pos.get("price_entry", 0)
                shares = pos.get("shares", 0)
                unrealized_pnl += (current_price - entry_price) * shares

        # True portfolio value = cash + market value of positions
        mtm_portfolio = portfolio_value + total_exposure + unrealized_pnl

        return {
            "portfolio_value": round(mtm_portfolio, 2),
            "available_capital": round(portfolio_value, 2),
            "total_exposure": total_exposure,
            "unrealized_pnl": round(unrealized_pnl, 4),
            "open_positions": stats["open_positions"],
            "total_pnl": round(stats["total_pnl"] + unrealized_pnl, 4),
            "realized_pnl": stats["total_pnl"],
            "win_rate": stats["win_rate"],
            "initial_capital": self.initial_capital,
        }

    def check_and_close_positions(self, markets_data: list):
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return 0.0

        # Build current prices map
        current_prices = {}
        for market in markets_data:
            current_prices[market["id"]] = {
                "price_yes": market["price_yes"],
                "price_no": market["price_no"],
            }

        # For markets not in scan, try fetching individually
        for pos in open_positions:
            mid = pos.get("market_id")
            if mid and mid not in current_prices:
                prices = self.api.get_market_prices(mid)
                if prices:
                    current_prices[mid] = prices

        pnl_from_closes = 0.0

        # Check stop-losses
        stops = self.risk.check_stop_losses(open_positions, current_prices)
        for stop in stops:
            trade = self.db.get_trade_by_id(stop["trade_id"])
            pnl = stop["profit_loss"]
            self.db.close_position(stop["trade_id"], stop["current_price"], pnl, "CLOSED")
            pnl_from_closes += pnl
            self.logger.warning(f"Stop-loss closed trade {stop['trade_id']}: P&L=${pnl:+.4f}")
            if trade:
                self.notifier.notify_trade_closed(trade, pnl, "stop_loss")
                # Analyze the closed trade
                if self.claude and self.memory.analysis_enabled:
                    closed_trade = self.db.get_trade_by_id(stop["trade_id"])
                    if closed_trade:
                        self.memory.analyze_closed_trade(closed_trade, self.claude)

        # Check take-profits
        takes = self.risk.check_take_profits(open_positions, current_prices)
        for take in takes:
            pnl = take["profit_loss"]
            self.db.close_position(take["trade_id"], take["current_price"], pnl, "CLOSED")
            pnl_from_closes += pnl
            self.logger.info(f"Take-profit closed trade {take['trade_id']}: P&L=${pnl:+.4f}")
            trade = self.db.get_trade_by_id(take["trade_id"])
            if trade:
                self.notifier.notify_trade_closed(trade, pnl, "take_profit")

        # DEMO: Check for auto-resolution
        if self.mode == "DEMO":
            already_closed = [s["trade_id"] for s in stops + takes]
            for pos in open_positions:
                if pos["id"] in already_closed:
                    continue

                mid = pos.get("market_id")
                prices = current_prices.get(mid, {})
                side = pos.get("side", "YES")
                current_price = prices.get(f"price_{side.lower()}", pos.get("price_entry", 0))
                entry_price = pos.get("price_entry", 0)

                # CHECK 1: Market expired (end_date passed)
                # If the market has ended, resolve based on final prices
                end_date_str = None
                for m in markets_data:
                    if m["id"] == mid:
                        end_date_str = m.get("end_date", "")
                        break

                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        if now > end_date:
                            # Market expired - resolve based on which side is winning
                            price_yes = prices.get("price_yes", 0.5)
                            price_no = prices.get("price_no", 0.5)
                            shares = pos.get("shares", 0)

                            if side == "YES":
                                won = price_yes > 0.5
                            else:
                                won = price_no > 0.5

                            if won:
                                pnl = (1.0 - entry_price) * shares
                                status_msg = "EXPIRED-WIN"
                            else:
                                pnl = -entry_price * shares
                                status_msg = "EXPIRED-LOSS"

                            self.db.close_position(pos["id"], current_price, pnl, "SIMULATED")
                            pnl_from_closes += pnl
                            self.logger.info(
                                f"[DEMO] {status_msg} trade {pos['id']}: "
                                f"market expired, P&L=${pnl:+.4f}"
                            )
                            trade = self.db.get_trade_by_id(pos["id"])
                            if trade:
                                self.notifier.notify_trade_closed(trade, pnl, status_msg.lower())
                                if self.claude and self.memory.analysis_enabled:
                                    self.memory.analyze_closed_trade(trade, self.claude)
                            continue
                    except (ValueError, TypeError):
                        pass

                # CHECK 2: Win condition - price > 0.97
                if current_price > 0.97:
                    shares = pos.get("shares", 0)
                    pnl = (1.0 - entry_price) * shares
                    self.db.close_position(pos["id"], current_price, pnl, "SIMULATED")
                    pnl_from_closes += pnl
                    self.logger.info(f"[DEMO] Auto-win trade {pos['id']}: price={current_price:.4f} P&L=${pnl:+.4f}")
                    trade = self.db.get_trade_by_id(pos["id"])
                    if trade:
                        self.notifier.notify_trade_closed(trade, pnl, "auto_win")
                        if self.claude and self.memory.analysis_enabled:
                            self.memory.analyze_closed_trade(trade, self.claude)

                # CHECK 3: Loss condition - price dropped > 40% from entry
                elif entry_price > 0 and (entry_price - current_price) / entry_price >= 0.40:
                    shares = pos.get("shares", 0)
                    pnl = (current_price - entry_price) * shares
                    self.db.close_position(pos["id"], current_price, pnl, "SIMULATED")
                    pnl_from_closes += pnl
                    self.logger.warning(f"[DEMO] Auto-loss trade {pos['id']}: price={current_price:.4f} P&L=${pnl:+.4f}")
                    trade = self.db.get_trade_by_id(pos["id"])
                    if trade:
                        self.notifier.notify_trade_closed(trade, pnl, "auto_loss")
                        if self.claude and self.memory.analysis_enabled:
                            self.memory.analyze_closed_trade(trade, self.claude)

        return pnl_from_closes

    def execute_trade(self, decision: dict, cycle_id: int) -> bool:
        action = decision.get("action", "SKIP")
        if action != "BUY":
            return False

        side = decision.get("side", "YES")
        price = decision.get("price_entry", 0)
        size = decision.get("suggested_size_usdc", 0)
        market_id = decision.get("market_id", "")
        market_question = decision.get("market_question", "")

        if price <= 0 or size <= 0:
            self.logger.warning(f"Invalid trade params: price={price}, size={size}")
            return False

        # Check for duplicate market position
        open_positions = self.db.get_open_positions()
        for existing in open_positions:
            if existing.get("market_id") == market_id:
                self.logger.warning(
                    f"Duplicate position blocked: already have trade #{existing['id']} "
                    f"on {market_question[:50]}"
                )
                return False

        # Validate with risk manager
        portfolio = self.get_portfolio_state()
        ev = decision.get("ev_calculated", 0)
        valid, reason = self.risk.validate_trade(
            ev=ev, price=price,
            liquidity=5000,  # already filtered
            min_liquidity=self.config["risk"]["min_liquidity_usdc"],
            volume_24h=10000,
            min_volume=self.config["risk"]["min_volume_24h"],
        )

        if not valid:
            self.logger.warning(f"Trade rejected by risk: {reason}")
            return False

        if not self.risk.can_open_new_position(portfolio["portfolio_value"], portfolio["total_exposure"]):
            return False

        # Cap position size
        max_size = portfolio["portfolio_value"] * self.config["risk"]["max_position_pct"]
        size = min(size, max_size, portfolio["available_capital"])

        if size < 2.0:
            self.logger.warning(f"Trade size too small: ${size:.2f}")
            return False

        shares = size / price if price > 0 else 0

        if self.mode == "DEMO":
            order = self.api.place_order_demo(market_id, side, size, price, market_question)
            status = "OPEN"
        else:
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            order_type = "FOK" if decision.get("strategy") == "sum_to_one" else "GTC"
            order = self.api.place_order_live(
                market_id, side, size, price, private_key, order_type, market_question
            )
            if not order:
                self.logger.error(f"Live order failed for {market_question[:50]}")
                return False
            status = "OPEN"

        trade_data = {
            "cycle_id": cycle_id,
            "mode": self.mode,
            "action": "BUY",
            "market_id": market_id,
            "market_question": market_question,
            "strategy": decision.get("strategy", "unknown"),
            "side": side,
            "price_entry": price,
            "size_usdc": size,
            "shares": round(shares, 6),
            "prob_real_estimated": decision.get("prob_real_estimated", 0),
            "prob_market": decision.get("prob_market", 0),
            "ev_calculated": ev,
            "reasoning": decision.get("reasoning", ""),
            "status": status,
            "confidence": decision.get("confidence", "MEDIUM"),
        }

        trade_id = self.db.log_trade(trade_data)
        self.logger.info(
            f"Trade executed: #{trade_id} {side} @ {price:.4f} ${size:.2f} "
            f"EV={ev:.4f} [{decision.get('strategy')}] {market_question[:50]}"
        )

        trade_data["id"] = trade_id
        self.notifier.notify_trade(trade_data)
        return True

    def run_memory_tasks(self):
        """Run memory-related tasks: analyze unanalyzed trades, extract rules, adjust params."""
        if not self.claude:
            return

        # Analyze any unanalyzed closed trades
        unanalyzed = self.memory.get_unanalyzed_trades()
        for trade in unanalyzed[:3]:  # Limit to 3 per cycle to save API calls
            self.logger.info(f"Analyzing closed trade #{trade['id']}...")
            self.memory.analyze_closed_trade(trade, self.claude)

        # Extract rules periodically
        if self.memory.should_extract_rules(self.cycle_count):
            self.logger.info("Extracting new rules from recent analyses...")
            new_rules = self.memory.extract_rules(self.claude)
            if new_rules:
                rules_text = "; ".join([r["rule_text"][:60] for r in new_rules])
                self.notifier.notify_memory_update("New Rules", rules_text)

        # Adjust parameters periodically
        if self.memory.should_adjust_parameters(self.cycle_count):
            self.logger.info("Evaluating parameter adjustments...")
            stats = self.db.get_statistics(self.initial_capital)
            suggestions = self.memory.suggest_parameter_adjustments(
                self.claude, stats, self.config
            )
            for suggestion in suggestions:
                self.memory.apply_adjustment(
                    suggestion["parameter_name"],
                    suggestion["new_value"],
                    self.config,
                    performance_before=stats.get("roi_pct"),
                )
                # Update risk manager with new values
                self.risk = RiskManager(self.config)
                self.notifier.notify_memory_update(
                    "Parameter Adjusted",
                    f"{suggestion['parameter_name']}: {suggestion['old_value']} -> {suggestion['new_value']} ({suggestion['reason']})"
                )

    def run_cycle(self):
        self.cycle_count += 1
        cycle_start = time.time()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"CYCLE #{self.cycle_count} starting...")
        self.logger.info(f"{'='*60}")

        trades_executed = 0
        pnl_cycle = 0.0
        opportunities_found = 0

        try:
            # 1. Get portfolio state
            portfolio = self.get_portfolio_state()
            self.logger.info(
                f"Portfolio: ${portfolio['portfolio_value']:.2f} | "
                f"Exposure: ${portfolio['total_exposure']:.2f} | "
                f"P&L: ${portfolio['total_pnl']:+.4f}"
            )

            # 2. Scan markets
            min_volume = self.config["risk"]["min_volume_24h"]
            min_liquidity = self.config["risk"]["min_liquidity_usdc"]
            markets = self.api.scan_markets(min_volume, min_liquidity)

            if not markets:
                self.logger.warning("No markets found in scan")
                self._log_cycle(0, 0, 0, portfolio, pnl_cycle, "No markets found")
                return

            # 3. Check and close existing positions
            pnl_from_closes = self.check_and_close_positions(markets)
            pnl_cycle += pnl_from_closes

            # Refresh portfolio after closes
            portfolio = self.get_portfolio_state()

            # 4. Find opportunities (3 strategies)
            bonding_opps = self.bonding.find_opportunities(markets, self.config)
            value_opps = self.value_betting.find_opportunities(markets, self.config)
            momentum_opps = self.momentum.find_opportunities(markets, self.config)
            opportunities_found = len(bonding_opps) + len(value_opps) + len(momentum_opps)

            self.logger.info(
                f"Opportunities: {len(bonding_opps)} bonding, "
                f"{len(value_opps)} value, {len(momentum_opps)} momentum"
            )

            # 5. Analyze with Claude if there are opportunities
            if opportunities_found > 0 and self.claude:
                if self.risk.can_open_new_position(portfolio["portfolio_value"], portfolio["total_exposure"]):
                    # Get memory context
                    memory_context = self.memory.get_memory_prompt_section()

                    result = self.claude.analyze_opportunities(
                        bonding_opps, value_opps, momentum_opps, portfolio, memory_context
                    )

                    if result and "decisions" in result:
                        max_trades = self.config["bot"].get("max_trades_per_cycle", 3)
                        min_ev = self.config["risk"]["min_ev_threshold"]

                        # Filter and sort decisions
                        buy_decisions = [
                            d for d in result["decisions"]
                            if d.get("action") == "BUY" and d.get("ev_calculated", 0) >= min_ev
                        ]
                        buy_decisions.sort(key=lambda x: x.get("ev_calculated", 0), reverse=True)

                        cycle_id = self.cycle_count
                        for decision in buy_decisions[:max_trades]:
                            if self.execute_trade(decision, cycle_id):
                                trades_executed += 1

                        # Log skipped decisions
                        skipped = [d for d in result["decisions"] if d.get("action") == "SKIP"]
                        for s in skipped:
                            self.logger.info(
                                f"SKIPPED: {s.get('market_question', '?')[:50]} - {s.get('reasoning', 'no reason')}"
                            )

                        if result.get("self_assessment"):
                            self.logger.info(f"Claude self-assessment: {result['self_assessment'][:200]}")
                    else:
                        self.logger.warning("Claude returned no valid decisions")
                else:
                    self.logger.info("Max exposure reached, skipping new trades")
            elif not self.claude:
                self.logger.info("No Claude agent available, skipping analysis")

            # 6. Run memory tasks (analysis, rules, params)
            self.run_memory_tasks()

            # 7. Log cycle
            portfolio = self.get_portfolio_state()
            self._log_cycle(
                len(markets), opportunities_found, trades_executed,
                portfolio, pnl_cycle, ""
            )

            elapsed = time.time() - cycle_start
            self.logger.info(
                f"Cycle #{self.cycle_count} complete in {elapsed:.1f}s | "
                f"Trades: {trades_executed} | Cycle P&L: ${pnl_cycle:+.4f} | "
                f"Portfolio: ${portfolio['portfolio_value']:.2f}"
            )

            # Active rules info
            active_rules = self.memory.get_active_rules()
            if active_rules:
                self.logger.info(f"Active learned rules: {len(active_rules)}")

        except Exception as e:
            self.logger.error(f"Cycle #{self.cycle_count} failed: {e}", exc_info=True)
            self.notifier.notify_error(str(e), f"Cycle #{self.cycle_count}")
            portfolio = self.get_portfolio_state()
            self._log_cycle(0, 0, 0, portfolio, 0, f"Error: {str(e)[:200]}")

    def _log_cycle(self, markets_scanned, opportunities, trades, portfolio, pnl_cycle, notes):
        cycle_data = {
            "mode": self.mode,
            "markets_scanned": markets_scanned,
            "opportunities_found": opportunities,
            "trades_executed": trades,
            "portfolio_value": portfolio["portfolio_value"],
            "pnl_cycle": pnl_cycle,
            "pnl_total": portfolio["total_pnl"],
            "capital_initial": self.initial_capital,
            "notes": notes,
        }
        self.db.log_cycle(cycle_data)
        self.notifier.notify_cycle_summary(cycle_data)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "live"],
                        help="Trading mode: demo (simulated) or live (real orders)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config)

    mode = args.mode.upper()
    print_banner(config, mode)

    if mode == "LIVE":
        logger.warning("=" * 60)
        logger.warning("  LIVE MODE - REAL MONEY WILL BE USED!")
        logger.warning("  Make sure POLYMARKET_PRIVATE_KEY is set correctly")
        logger.warning("=" * 60)
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            logger.error("POLYMARKET_PRIVATE_KEY not set. Aborting.")
            sys.exit(1)

    # Initialize bot
    bot = TradingBot(config, mode)

    # Run first cycle immediately
    logger.info("Running first cycle...")
    bot.run_cycle()

    # Schedule subsequent cycles
    interval = config["bot"]["cycle_interval_minutes"]
    schedule.every(interval).minutes.do(bot.run_cycle)
    logger.info(f"Scheduled: cycle every {interval} minutes. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        bot.db.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
