"""
Claude Agent - Uses Anthropic API to analyze trading opportunities.
Includes dynamic memory context (learned rules + trade history).
"""

import json
import os
import logging
import re
from anthropic import Anthropic

logger = logging.getLogger("polymarket_bot.claude")

SYSTEM_PROMPT_BASE = """Eres un agente de trading autonomo en Polymarket. Tu objetivo es maximizar el retorno ajustado por riesgo con enfoque CONSERVADOR.

RESTRICCIONES FIJAS (NO NEGOCIABLES):
- EV minimo: 0.02 para bonding, 0.05 para value_betting y momentum
- Maximo 5% del portfolio por trade
- Liquidez minima del mercado: $5,000 USDC
- No operar en precios > 0.97 o < 0.03
- Maximo 3 trades por ciclo
- NUNCA apostar en el mismo mercado dos veces
- Si no tienes informacion clara sobre un mercado, SKIP obligatorio
- Prefiere NO operar a operar con duda. Es mejor perder una oportunidad que perder capital.

ESTRATEGIAS (en orden de prioridad):

1. BONDING (PRIORIDAD ALTA - mas seguro):
   Compra outcomes de alta probabilidad (0.80-0.97) que estan a punto de resolverse.
   - SOLO comprar cuando estes >90% seguro de que el outcome ocurrira
   - Retorno anualizado debe ser > 15%
   - Preferir mercados que resuelven en < 7 dias (menos riesgo)
   - Evitar bonding en mercados con fechas lejanas (>30 dias) - mucho puede cambiar
   - NUNCA hacer bonding en deportes en curso (resultado incierto)
   - IDEAL: eventos geopoliticos donde el status quo es muy improbable que cambie rapido

2. VALUE BETTING (PRIORIDAD MEDIA - solo con edge real):
   Mercados inciertos donde TIENES conocimiento real que el mercado no ha incorporado.
   - SOLO si tu estimacion difiere del mercado en >10 puntos porcentuales
   - SKIP si no sabes nada especifico sobre el tema
   - SKIP en deportes individuales (partidos) - no tienes edge sobre las casas de apuestas
   - SKIP en esports - no tienes informacion en tiempo real
   - Buenos candidatos: politica (elecciones con datos de encuestas), geopolitica, economia
   - SE MUY CONSERVADOR: asume que el mercado es 80% eficiente

3. MOMENTUM (PRIORIDAD BAJA - solo con contexto claro):
   SOLO si el alto volumen tiene una explicacion logica (noticia, evento) Y puedes
   razonar sobre la direccion. No seguir volumen ciegamente.
   - Requiere ratio vol/liq > 10x Y una tesis clara
   - SKIP si no sabes POR QUE hay volumen alto

REGLAS DE SUPERVIVENCIA:
- Es mejor hacer 0 trades que 1 trade malo
- Si solo hay bonding disponible, solo haz bonding
- Nunca asumas que "es probable" sin evidencia concreta
- Los mercados deportivos en curso son CASINO - no tenemos edge
- Si la liquidez es baja (<$20K), reduce el size a la mitad
- Cuando dudes: SKIP

PROCESO DE ANALISIS: Para cada oportunidad:
1. Precio de mercado y probabilidad implicita
2. Que se CONCRETAMENTE sobre este evento? (no suposiciones)
3. Mi estimacion de probabilidad con justificacion
4. EV = prob_real * (1 - precio) - (1 - prob_real) * precio
5. Verificacion de sesgos: estoy anclado al precio? Tengo sobreconfianza?
6. Size via Kelly 25%, NUNCA mas del 5% del portfolio
7. Decision: BUY solo si estoy seguro. SKIP si hay cualquier duda.

RESPONDE SIEMPRE EN JSON VALIDO (sin markdown, sin backticks):
{
  "analysis": "razonamiento completo paso a paso",
  "decisions": [
    {
      "market_id": "string",
      "market_question": "string",
      "action": "BUY o SKIP",
      "side": "YES o NO o null",
      "strategy": "bonding o value_betting o momentum o skip",
      "prob_real_estimated": 0.0,
      "prob_market": 0.0,
      "ev_calculated": 0.0,
      "suggested_size_usdc": 0.0,
      "price_entry": 0.0,
      "reasoning": "razon concisa de la decision",
      "confidence": "HIGH o MEDIUM o LOW"
    }
  ],
  "market_summary": "resumen breve del estado del mercado",
  "self_assessment": "evaluacion de tu nivel de confianza y posibles sesgos en este ciclo"
}"""


class ClaudeAgent:
    def __init__(self, config: dict):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set!")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = Anthropic(api_key=api_key)
        claude_cfg = config.get("claude", {})
        self.model = claude_cfg.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = claude_cfg.get("max_tokens", 4096)
        self.temperature = claude_cfg.get("temperature", 0.3)
        logger.info(f"ClaudeAgent initialized: model={self.model}")

    def _build_system_prompt(self, memory_context: str = "") -> str:
        prompt = SYSTEM_PROMPT_BASE
        if memory_context:
            prompt += f"\n\n=== MEMORIA DEL BOT (CONTEXTO DE EXPERIENCIA PREVIA) ===\n{memory_context}"
        return prompt

    def call_claude(self, user_message: str, max_tokens: int = None,
                    system_override: str = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                system=system_override or SYSTEM_PROMPT_BASE,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text.strip()
            logger.debug(f"Claude response ({len(text)} chars)")
            return text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

    def analyze_opportunities(self, bonding_opps: list, value_opps: list,
                              momentum_opps: list = None, portfolio_state: dict = None,
                              memory_context: str = "") -> dict:
        if portfolio_state is None:
            portfolio_state = {}
        if momentum_opps is None:
            momentum_opps = []
        user_message = self.format_opportunities_message(
            bonding_opps, value_opps, momentum_opps, portfolio_state
        )

        system_prompt = self._build_system_prompt(memory_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text.strip()
            logger.info(f"Claude analyzed {len(bonding_opps)} bonding + {len(value_opps)} value + {len(momentum_opps)} momentum opportunities")

            # Try to parse JSON, handling potential markdown wrapping
            parsed = self._parse_json_response(raw_text)
            if parsed is None:
                logger.error("Failed to parse Claude response as JSON")
                logger.debug(f"Raw response: {raw_text[:500]}")
                return {}

            return parsed

        except Exception as e:
            logger.error(f"Error in analyze_opportunities: {e}")
            return {}

    def format_opportunities_message(self, bonding_opps: list, value_opps: list,
                                     momentum_opps: list, portfolio_state: dict) -> str:
        msg = "=== ESTADO DEL PORTFOLIO ===\n"
        msg += f"Capital disponible: ${portfolio_state.get('available_capital', 0):.2f}\n"
        msg += f"Valor total portfolio: ${portfolio_state.get('portfolio_value', 0):.2f}\n"
        msg += f"Exposicion actual: ${portfolio_state.get('total_exposure', 0):.2f}\n"
        msg += f"Posiciones abiertas: {portfolio_state.get('open_positions', 0)}\n"
        msg += f"P&L total: ${portfolio_state.get('total_pnl', 0):+.4f}\n"
        msg += f"Win rate: {portfolio_state.get('win_rate', 0):.1f}%\n"

        if bonding_opps:
            msg += "\n=== OPORTUNIDADES BONDING (retorno seguro) ===\n"
            for i, opp in enumerate(bonding_opps, 1):
                msg += f"\n--- Bonding #{i} ---\n"
                msg += f"Mercado: {opp['market_question']}\n"
                msg += f"Market ID: {opp['market_id']}\n"
                msg += f"Lado: {opp['side']} @ {opp['price']:.4f}\n"
                msg += f"Retorno anualizado: {opp.get('annualized_return', 0):.1%}\n"
                msg += f"Retorno por dolar: {opp.get('return_per_dollar', 0):.4f}\n"
                msg += f"Dias a resolucion: {opp.get('days_to_resolution', '?')}\n"
                msg += f"Volumen 24h: ${opp['volume_24h']:,.0f}\n"
                msg += f"Liquidez: ${opp['liquidity']:,.0f}\n"
                msg += f"Categoria: {opp.get('category', 'N/A')}\n"

        if value_opps:
            msg += "\n=== OPORTUNIDADES VALUE BETTING (analiza tu edge) ===\n"
            for i, opp in enumerate(value_opps, 1):
                msg += f"\n--- Value #{i} ---\n"
                msg += f"Mercado: {opp['market_question']}\n"
                msg += f"Market ID: {opp['market_id']}\n"
                msg += f"YES: {opp['price_yes']:.4f} | NO: {opp['price_no']:.4f}\n"
                msg += f"Incertidumbre: {opp.get('uncertainty_score', 0):.2f}\n"
                msg += f"Volumen 24h: ${opp['volume_24h']:,.0f}\n"
                msg += f"Liquidez: ${opp['liquidity']:,.0f}\n"
                msg += f"Categoria: {opp.get('category', 'N/A')}\n"
                msg += "IMPORTANTE: Estima TU probabilidad real. No te ancles al precio.\n"

        if momentum_opps:
            msg += "\n=== OPORTUNIDADES MOMENTUM (volumen inusual) ===\n"
            for i, opp in enumerate(momentum_opps, 1):
                msg += f"\n--- Momentum #{i} ---\n"
                msg += f"Mercado: {opp['market_question']}\n"
                msg += f"Market ID: {opp['market_id']}\n"
                msg += f"YES: {opp['price_yes']:.4f} | NO: {opp['price_no']:.4f}\n"
                msg += f"Ratio vol/liq: {opp.get('vol_liq_ratio', 0):.1f}x\n"
                msg += f"Volumen 24h: ${opp['volume_24h']:,.0f}\n"
                msg += f"Liquidez: ${opp['liquidity']:,.0f}\n"
                msg += f"Categoria: {opp.get('category', 'N/A')}\n"

        if not bonding_opps and not value_opps and not momentum_opps:
            msg += "\nNo se encontraron oportunidades en este ciclo.\n"

        msg += "\nAnaliza cada oportunidad y devuelve tu decision en JSON."
        return msg

    def _parse_json_response(self, text: str) -> dict:
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try removing markdown code blocks
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try finding JSON object in text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def analyze_trade_result(self, trade: dict) -> dict:
        """Used by MemorySystem for post-trade analysis."""
        pnl = trade.get("profit_loss", 0) or 0
        result = "GANANCIA" if pnl > 0 else "PERDIDA"

        prompt = f"""Analiza brevemente este trade cerrado:

- Mercado: {trade.get('market_question', '?')}
- Estrategia: {trade.get('strategy', '?')}
- {trade.get('side', '?')} @ {trade.get('price_entry', 0):.4f}
- EV: {trade.get('ev_calculated', 0):.4f}
- Resultado: {result} (${pnl:+.4f})

RESPONDE EN JSON:
{{"analysis": "breve", "bias_identified": "sesgo o ninguno", "lesson": "leccion accionable", "calibration_note": "nota"}}"""

        try:
            resp = self.call_claude(prompt, max_tokens=512)
            return self._parse_json_response(resp) or {}
        except Exception as e:
            logger.error(f"Trade result analysis failed: {e}")
            return {}
