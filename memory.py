"""
Memory System - Self-improving bot via trade analysis, rule extraction, and parameter tuning.

Components:
1. Trade history context for Claude (last N trades as context)
2. Post-trade auto-analysis (why it won/lost, bias detection)
3. Learned rules extraction (accumulated wisdom)
4. Automatic parameter adjustment (EV threshold, Kelly fraction, etc.)
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("polymarket_bot.memory")


class MemorySystem:
    def __init__(self, db, config: dict):
        self.db = db
        mem_cfg = config.get("memory", {})
        self.history_size = mem_cfg.get("history_context_size", 20)
        self.analysis_enabled = mem_cfg.get("analysis_after_close", True)
        self.rule_extraction_interval = mem_cfg.get("rule_extraction_every_n_cycles", 20)
        self.param_adjustment_interval = mem_cfg.get("parameter_adjustment_every_n_cycles", 20)
        self.min_trades_for_adjustments = mem_cfg.get("min_trades_for_adjustments", 20)
        self.max_active_rules = mem_cfg.get("max_active_rules", 15)
        logger.info(
            f"MemorySystem initialized: history_size={self.history_size}, "
            f"rule_interval={self.rule_extraction_interval} cycles"
        )

    # ---- Component 1: Trade History Context ----

    def get_trade_history_context(self, n: int = None) -> str:
        n = n or self.history_size
        trades = self.db.get_recent_closed_trades(n)

        if not trades:
            return "No hay trades cerrados todavia. Este es el inicio del historial."

        lines = ["=== HISTORIAL DE TRADES RECIENTES ==="]
        wins = 0
        losses = 0
        total_pnl = 0.0

        for t in trades:
            pnl = t.get("profit_loss", 0) or 0
            total_pnl += pnl
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "NEUTRAL"
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

            est_prob = t.get("prob_real_estimated", 0) or 0
            mkt_prob = t.get("prob_market", 0) or 0
            error = abs(est_prob - (1.0 if pnl > 0 else 0.0)) if est_prob > 0 else None

            line = (
                f"- [{result}] {t.get('strategy', '?')} | {t.get('side', '?')} @ {t.get('price_entry', 0):.4f} | "
                f"EV={t.get('ev_calculated', 0):.4f} | P&L=${pnl:+.4f} | "
                f"Est.prob={est_prob:.2f} vs Market={mkt_prob:.2f}"
            )
            if error is not None:
                line += f" | Error={error:.2f}"
            line += f" | {t.get('market_question', '')[:50]}"
            lines.append(line)

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        lines.append(f"\nResumen: {wins}W/{losses}L ({win_rate:.0f}% win rate) | P&L total: ${total_pnl:+.4f}")

        return "\n".join(lines)

    # ---- Component 2: Post-Trade Auto-Analysis ----

    def analyze_closed_trade(self, trade: dict, claude_agent=None) -> dict:
        return {}

    def _analyze_closed_trade_unused(self, trade: dict, claude_agent) -> dict:
        if not self.analysis_enabled:
            return {}

        pnl = trade.get("profit_loss", 0) or 0
        result = "GANANCIA" if pnl > 0 else "PERDIDA"
        actual_prob = 1.0 if pnl > 0 else 0.0
        est_prob = trade.get("prob_real_estimated", 0) or 0
        estimation_error = abs(est_prob - actual_prob)

        prompt = f"""Analiza este trade cerrado y extrae lecciones:

TRADE:
- Mercado: {trade.get('market_question', '?')}
- Estrategia: {trade.get('strategy', '?')}
- Lado: {trade.get('side', '?')} @ {trade.get('price_entry', 0):.4f}
- Size: ${trade.get('size_usdc', 0):.2f}
- EV calculado: {trade.get('ev_calculated', 0):.4f}
- Prob. estimada: {est_prob:.4f}
- Prob. mercado: {trade.get('prob_market', 0):.4f}
- Resultado: {result} (P&L: ${pnl:+.4f})
- Error de estimacion: {estimation_error:.4f}
- Razonamiento original: {trade.get('reasoning', 'N/A')}

RESPONDE EN JSON:
{{
  "analysis": "analisis detallado de por que el trade tuvo este resultado",
  "bias_identified": "sesgo identificado o 'ninguno'",
  "lesson": "leccion concreta y accionable para futuros trades",
  "calibration_note": "nota sobre si la estimacion de probabilidad fue buena o mala y por que"
}}"""

        try:
            response = claude_agent.call_claude(prompt, max_tokens=1024)
            parsed = json.loads(response)

            analysis_data = {
                "trade_id": trade["id"],
                "analysis_text": parsed.get("analysis", ""),
                "prob_estimated": est_prob,
                "prob_actual": actual_prob,
                "estimation_error": estimation_error,
                "bias_identified": parsed.get("bias_identified", ""),
                "lesson_extracted": parsed.get("lesson", ""),
            }

            self.db.log_analysis(analysis_data)
            logger.info(f"Trade {trade['id']} analyzed. Bias: {parsed.get('bias_identified', 'none')}")
            return parsed

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from Claude during trade analysis for trade {trade['id']}")
            return {}
        except Exception as e:
            logger.error(f"Error analyzing trade {trade['id']}: {e}")
            return {}

    def get_unanalyzed_trades(self) -> list[dict]:
        return self.db.get_unanalyzed_closed_trades()

    # ---- Component 3: Rule Extraction ----

    def should_extract_rules(self, cycle_count: int) -> bool:
        return cycle_count > 0 and cycle_count % self.rule_extraction_interval == 0

    def extract_rules(self, claude_agent=None, force: bool = False) -> list[dict]:
        return []

    def _extract_rules_unused(self, claude_agent, force: bool = False) -> list[dict]:
        analyses = self.db.get_recent_analyses(30)
        if not analyses and not force:
            logger.info("No analyses available for rule extraction")
            return []

        current_rules = self.db.get_learned_rules(active_only=True)
        current_rules_text = "\n".join(
            [f"- [{r['id']}] (eff:{r['effectiveness_pct']:.0f}%) {r['rule_text']}" for r in current_rules]
        ) or "Ninguna regla activa todavia."

        analyses_text = ""
        for a in analyses:
            analyses_text += (
                f"- Trade #{a['trade_id']}: error={a.get('estimation_error', 0):.2f}, "
                f"bias={a.get('bias_identified', 'none')}, "
                f"leccion={a.get('lesson_extracted', 'N/A')}\n"
            )

        performance = self.db.get_performance_summary(50)

        prompt = f"""Eres el sistema de memoria de un bot de trading en Polymarket.

ANALISIS RECIENTES:
{analyses_text}

REGLAS ACTIVAS ACTUALES:
{current_rules_text}

PERFORMANCE:
- Win rate: {performance.get('win_rate', 0):.1f}%
- Error promedio estimacion: {performance.get('avg_estimation_error', 0):.3f}
- EV promedio: {performance.get('avg_ev', 0):.4f}
- Trades cerrados: {performance.get('total_closed', 0)}

TAREA: Basandote en los analisis recientes y el performance:
1. Propone NUEVAS reglas que el bot deberia seguir (max 3)
2. Identifica reglas existentes que deberian desactivarse si ya no son utiles
3. Cada regla debe ser concreta, accionable y verificable

RESPONDE EN JSON:
{{
  "new_rules": [
    {{"rule_text": "texto de la regla", "category": "categoria", "confidence": 0.0-1.0}}
  ],
  "deactivate_rule_ids": [lista de IDs a desactivar],
  "reasoning": "explicacion de por que estas reglas"
}}"""

        try:
            response = claude_agent.call_claude(prompt, max_tokens=2048)
            parsed = json.loads(response)

            new_rules = []
            for rule_data in parsed.get("new_rules", []):
                # Check we haven't exceeded max active rules
                active_count = len(self.db.get_learned_rules(active_only=True))
                if active_count >= self.max_active_rules:
                    logger.warning(f"Max active rules ({self.max_active_rules}) reached, skipping new rules")
                    break

                rule_id = self.db.add_learned_rule({
                    "rule_text": rule_data["rule_text"],
                    "category": rule_data.get("category", "general"),
                    "confidence": rule_data.get("confidence", 0.5),
                })
                new_rules.append({"id": rule_id, **rule_data})

            for rule_id in parsed.get("deactivate_rule_ids", []):
                try:
                    self.db.deactivate_rule(int(rule_id))
                except (ValueError, TypeError):
                    pass

            logger.info(
                f"Rule extraction: {len(new_rules)} new, "
                f"{len(parsed.get('deactivate_rule_ids', []))} deactivated"
            )
            return new_rules

        except json.JSONDecodeError:
            logger.warning("Invalid JSON from Claude during rule extraction")
            return []
        except Exception as e:
            logger.error(f"Error extracting rules: {e}")
            return []

    def get_active_rules(self) -> list[dict]:
        return self.db.get_learned_rules(active_only=True)

    # ---- Component 4: Parameter Adjustment ----

    def should_adjust_parameters(self, cycle_count: int) -> bool:
        if cycle_count <= 0 or cycle_count % self.param_adjustment_interval != 0:
            return False
        stats = self.db.get_performance_summary()
        return stats.get("total_closed", 0) >= self.min_trades_for_adjustments

    def suggest_parameter_adjustments(self, claude_agent=None, stats: dict = None, current_config: dict = None) -> list[dict]:
        return []

    def _suggest_parameter_adjustments_unused(self, claude_agent, stats: dict, current_config: dict) -> list[dict]:
        performance = self.db.get_performance_summary(50)
        adjustments_history = self.db.get_parameter_adjustments(10)

        history_text = ""
        for adj in adjustments_history:
            history_text += (
                f"- {adj['parameter_name']}: {adj['old_value']} -> {adj['new_value']} "
                f"(reason: {adj.get('reason', 'N/A')})\n"
            )
        history_text = history_text or "Sin ajustes previos."

        risk_cfg = current_config.get("risk", {})

        prompt = f"""Eres el optimizador de parametros de un bot de trading Polymarket.

PERFORMANCE ACTUAL:
- Win rate: {performance.get('win_rate', 0):.1f}%
- Error promedio estimacion: {performance.get('avg_estimation_error', 0):.3f}
- EV promedio: {performance.get('avg_ev', 0):.4f}
- Trades cerrados: {performance.get('total_closed', 0)}
- Max drawdown: {stats.get('max_drawdown', 0):.1f}%
- ROI: {stats.get('roi_pct', 0):.1f}%

PARAMETROS ACTUALES:
- min_ev_threshold: {risk_cfg.get('min_ev_threshold', 0.05)}
- kelly_fraction: {risk_cfg.get('kelly_fraction', 0.25)}
- stop_loss_pct: {risk_cfg.get('stop_loss_pct', 0.40)}
- take_profit_partial_pct: {risk_cfg.get('take_profit_partial_pct', 0.80)}
- max_position_pct: {risk_cfg.get('max_position_pct', 0.05)}

HISTORIAL DE AJUSTES:
{history_text}

TAREA: Sugiere ajustes de parametros para mejorar el rendimiento.
- Si el win rate es bajo, considera subir min_ev_threshold
- Si hay muchos stop-losses, considera reducir kelly_fraction
- Si se pierden oportunidades buenas, considera bajar min_ev_threshold
- NO cambies mas de 2 parametros a la vez
- Los cambios deben ser graduales (max 20% del valor actual)

RESPONDE EN JSON:
{{
  "adjustments": [
    {{
      "parameter_name": "nombre del parametro en config.risk",
      "new_value": valor_numerico,
      "reason": "razon del cambio"
    }}
  ],
  "no_change_reason": "si no recomiendas cambios, explica por que"
}}"""

        try:
            response = claude_agent.call_claude(prompt, max_tokens=1024)
            parsed = json.loads(response)

            suggestions = []
            for adj in parsed.get("adjustments", []):
                param_name = adj["parameter_name"]
                new_value = float(adj["new_value"])
                old_value = risk_cfg.get(param_name)

                if old_value is None:
                    logger.warning(f"Unknown parameter: {param_name}, skipping")
                    continue

                # Validate gradual change (max 20% shift)
                if old_value > 0:
                    change_pct = abs(new_value - old_value) / old_value
                    if change_pct > 0.25:
                        logger.warning(
                            f"Adjustment too large for {param_name}: "
                            f"{old_value} -> {new_value} ({change_pct:.0%}), capping"
                        )
                        direction = 1 if new_value > old_value else -1
                        new_value = old_value * (1 + direction * 0.20)

                suggestions.append({
                    "parameter_name": param_name,
                    "old_value": old_value,
                    "new_value": round(new_value, 6),
                    "reason": adj.get("reason", ""),
                })

            if not suggestions and parsed.get("no_change_reason"):
                logger.info(f"No parameter adjustments suggested: {parsed['no_change_reason']}")

            return suggestions

        except json.JSONDecodeError:
            logger.warning("Invalid JSON from Claude during parameter adjustment")
            return []
        except Exception as e:
            logger.error(f"Error suggesting parameter adjustments: {e}")
            return []

    def apply_adjustment(self, param_name: str, new_value: float, config: dict,
                         performance_before: float = None) -> bool:
        old_value = config["risk"].get(param_name)
        if old_value is None:
            return False

        config["risk"][param_name] = new_value

        self.db.log_parameter_adjustment({
            "parameter_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": f"Auto-adjusted based on performance analysis",
            "applied": 1,
            "performance_before": performance_before,
        })

        logger.info(f"Parameter adjusted: {param_name} = {old_value} -> {new_value}")
        return True

    # ---- Combined Memory Prompt ----

    def get_memory_prompt_section(self) -> str:
        sections = []

        # Trade history
        history = self.get_trade_history_context()
        if history:
            sections.append(history)

        # Active rules
        rules = self.get_active_rules()
        if rules:
            rules_text = "=== REGLAS APRENDIDAS (SEGUIR ESTRICTAMENTE) ==="
            for r in rules:
                rules_text += (
                    f"\n- [{r['category']}] (confianza: {r['confidence']:.0%}, "
                    f"efectividad: {r['effectiveness_pct']:.0f}%) {r['rule_text']}"
                )
            sections.append(rules_text)

        # Recent parameter adjustments
        adjustments = self.db.get_parameter_adjustments(5)
        if adjustments:
            adj_text = "=== AJUSTES DE PARAMETROS RECIENTES ==="
            for a in adjustments:
                adj_text += (
                    f"\n- {a['parameter_name']}: {a['old_value']} -> {a['new_value']} "
                    f"({a.get('reason', '')})"
                )
            sections.append(adj_text)

        return "\n\n".join(sections) if sections else ""

    def update_rule_effectiveness(self, rule_id: int, was_helpful: bool):
        rule = None
        for r in self.db.get_learned_rules(active_only=False):
            if r["id"] == rule_id:
                rule = r
                break

        if not rule:
            return

        times_applied = rule["times_applied"] + 1
        times_helpful = rule["times_helpful"] + (1 if was_helpful else 0)
        effectiveness = (times_helpful / times_applied * 100) if times_applied > 0 else 0

        self.db.update_rule(rule_id, {
            "times_applied": times_applied,
            "times_helpful": times_helpful,
            "effectiveness_pct": round(effectiveness, 1),
        })
