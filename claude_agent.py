"""
Claude Agent - Weather-only trade review.

The WeatherBotStrategy produces opportunities already vetted by the ensemble +
confidence engine. Claude acts as a final sanity check: catch obvious parser
errors (wrong city, wrong date, off-by-thousand prices), reject duplicates
against the current portfolio, and optionally reduce size on uncertain trades.
"""

import json
import os
import re
import logging
from datetime import datetime, timezone
from anthropic import Anthropic

logger = logging.getLogger("polymarket_bot.claude")

SYSTEM_PROMPT_BASE = """Eres un revisor final de trades de un bot Weather en Polymarket.

Cada oportunidad llega YA validada por:
  - Ensemble de 3-8 fuentes meteorologicas con acuerdo alto (std baja).
  - Confidence score >= 75 (0-100) con componentes desglosados.
  - Veto por rolling win-rate del lado (YES/NO).
  - Edge minimo aplicado sobre prob_blended vs precio de mercado.

Tu trabajo NO es recalcular el forecast; ya esta hecho. Tu trabajo es:
  1. Detectar errores de parseo: ciudad equivocada, fecha fuera del periodo de
     forecast, umbral mal convertido (F vs C), unidad incorrecta.
  2. Detectar duplicados contra posiciones abiertas (mismo mercado + mismo lado).
  3. Marcar SKIP si la pregunta contiene condiciones especiales que el parser
     no maneja (ej. "highest between X and Y hours", "average over", etc).
  4. Aprobar BUY con size = size_suggested. Solo reduce size a la mitad si ves
     una senal concreta de incertidumbre adicional; nunca la aumentes.

Limites duros:
  - Nunca > 6 aprobaciones por ciclo.
  - Nunca aprobar un mercado con posicion abierta del mismo lado salvo que
     la confidence haya subido >= 5 puntos respecto a la apertura previa.
  - Nunca aprobar si hours_to_resolution < 2.

Responde SOLO en JSON valido (sin markdown, sin backticks):
{
  "analysis": "razonamiento breve del ciclo",
  "decisions": [
    {
      "market_id": "string",
      "market_question": "string",
      "action": "BUY" | "SKIP",
      "side": "YES" | "NO",
      "price_entry": 0.0,
      "suggested_size_usdc": 0.0,
      "prob_real_estimated": 0.0,
      "prob_market": 0.0,
      "ev_calculated": 0.0,
      "confidence_score": 0.0,
      "reasoning": "1 frase"
    }
  ],
  "self_assessment": "sesgos o dudas"
}
"""


class ClaudeAgent:
    def __init__(self, config: dict):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = Anthropic(api_key=api_key)
        claude_cfg = config.get("claude", {})
        self.model = claude_cfg.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = claude_cfg.get("max_tokens", 4096)
        self.temperature = claude_cfg.get("temperature", 0.2)
        logger.info(f"ClaudeAgent (weather) initialized: model={self.model}")

    def _build_system_prompt(self, memory_context: str = "") -> str:
        prompt = SYSTEM_PROMPT_BASE
        if memory_context:
            prompt += f"\n\n=== MEMORIA DEL BOT ===\n{memory_context}"
        return prompt

    def call_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def analyze_weather_opportunities(
        self,
        opportunities: list[dict],
        portfolio_state: dict,
        memory_context: str = "",
    ) -> dict:
        if not opportunities:
            return {"analysis": "no opportunities", "decisions": [], "self_assessment": ""}

        msg = self._format_message(opportunities, portfolio_state or {})
        system_prompt = self._build_system_prompt(memory_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": msg}],
            )
            raw = response.content[0].text.strip()
            parsed = self._parse_json_response(raw)
            if parsed is None:
                logger.error(f"Failed to parse Claude response: {raw[:300]}")
                return {"analysis": "parse_failed", "decisions": [], "self_assessment": raw[:200]}
            return parsed
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
            return {"analysis": f"error: {e}", "decisions": [], "self_assessment": ""}

    def _format_message(self, opps: list[dict], portfolio: dict) -> str:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"FECHA_HOY: {today_str}",
            "",
            "=== PORTFOLIO ===",
            f"Capital disponible: ${portfolio.get('available_capital', 0):.2f}",
            f"Valor total: ${portfolio.get('portfolio_value', 0):.2f}",
            f"Exposicion: ${portfolio.get('total_exposure', 0):.2f}",
            f"Posiciones abiertas: {portfolio.get('open_positions', 0)}",
            f"PnL total: ${portfolio.get('total_pnl', 0):+.4f}",
            f"Win rate: {portfolio.get('win_rate', 0):.1f}%",
            f"Side rolling WR: YES={portfolio.get('yes_wr', 0.5):.2f} NO={portfolio.get('no_wr', 0.5):.2f}",
            "",
            "=== OPORTUNIDADES WEATHER (ya validadas por ensemble+confidence) ===",
        ]
        for i, opp in enumerate(opps, 1):
            lines.append(f"\n--- #{i} ---")
            lines.append(f"market_id: {opp.get('market_id','')}")
            lines.append(f"Pregunta: {opp.get('market_question','')}")
            lines.append(f"Location: {opp.get('location','?')}  Fecha: {opp.get('target_date','?')}  "
                         f"({opp.get('hours_to_resolution',0):.1f}h a resolucion)")
            lines.append(f"Metrica: {opp.get('weather_type','?')}")
            lines.append(f"Lado: {opp.get('side')} @ {opp.get('price',0):.4f}  "
                         f"(YES={opp.get('price_yes',0):.3f} / NO={opp.get('price_no',0):.3f})")
            lines.append(f"Ensemble: mean={opp.get('ensemble_mean')} std={opp.get('ensemble_std')} "
                         f"sources_used={opp.get('ensemble_sources_used')}")
            lines.append(f"P(YES) ensemble={opp.get('prob_ensemble',0):.3f} "
                         f"blended={opp.get('prob_blended',0):.3f}  edge={opp.get('edge',0):+.3f}")
            lines.append(f"Confidence: {opp.get('confidence_score',0):.1f} "
                         f"componentes={opp.get('confidence_breakdown',{}).get('components',{})}")
            lines.append(f"Size sugerida: ${opp.get('suggested_size_usdc', 0):.2f}")
        lines.append("\nDecide BUY/SKIP para cada mercado. JSON ONLY.")
        return "\n".join(lines)

    def analyze_trade_result(self, trade: dict) -> dict:
        """Used by MemorySystem for post-trade analysis."""
        pnl = trade.get("profit_loss", 0) or 0
        result = "GANANCIA" if pnl > 0 else "PERDIDA"
        prompt = f"""Analiza brevemente este trade weather cerrado:

- Mercado: {trade.get('market_question', '?')}
- Lado: {trade.get('side', '?')} @ {trade.get('price_entry', 0):.4f}
- Confidence: {trade.get('confidence_score', '?')}
- Ensemble: mean={trade.get('ensemble_mean','?')} std={trade.get('ensemble_std','?')} sources={trade.get('ensemble_sources_used','?')}
- EV: {trade.get('ev_calculated', 0):.4f}
- Resultado: {result} (${pnl:+.4f})

RESPONDE EN JSON:
{{"analysis": "breve", "bias_identified": "sesgo o ninguno", "lesson": "leccion accionable", "calibration_note": "nota"}}"""
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=512, temperature=self.temperature,
                system="Eres un analista de trades weather. Responde solo JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_json_response(resp.content[0].text.strip()) or {}
        except Exception as e:
            logger.error(f"analyze_trade_result failed: {e}")
            return {}

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None
