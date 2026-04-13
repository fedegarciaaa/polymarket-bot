# MEMORY - Estado del Proyecto Polymarket Bot

> Documento para continuar el desarrollo desde cualquier ordenador.
> Ultima actualizacion: 2026-04-13

## Estado Actual: v1.1 - Funcional en DEMO

El bot esta operativo en modo DEMO. Se ha ejecutado durante varios ciclos y ha realizado trades simulados exitosamente.

## Arquitectura

```
polymarket-bot/
├── main.py              # Orquestador: ciclos cada 10min, gestiona trades
├── config.yaml          # Configuracion central (estrategias, riesgo, claude, memoria)
├── polymarket_api.py    # Cliente API: Gamma (lectura) + CLOB (trading)
├── claude_agent.py      # Agente Claude Sonnet 4: analiza oportunidades, decide BUY/SKIP
├── risk_manager.py      # Kelly 25%, stop-loss 40%, take-profit 80%, limites exposicion
├── database.py          # SQLite: trades, cycles, trade_analyses, learned_rules, parameter_adjustments
├── memory.py            # Sistema auto-mejora: historial, analisis post-trade, reglas, ajuste params
├── dashboard.py         # Flask :5000 standalone con Chart.js
├── notifications.py     # Telegram: trades, errores, resumenes de ciclo
├── strategies/
│   ├── bonding.py       # Outcomes alta probabilidad (0.80-0.97) con retorno anualizado >15%
│   └── arbitrage.py     # ValueBettingStrategy (0.10-0.90) + MomentumStrategy (vol/liq ratio)
├── start.bat            # Lanza bot + dashboard en Windows
├── .env                 # SECRETO - no en git. Contiene API keys
└── data/bot.db          # SQLite - no en git. Se crea automaticamente
```

## Decisiones de Diseno Tomadas

1. **Procesos separados**: `python main.py --mode demo` + `python dashboard.py` en terminales distintas
2. **Modelo Claude**: Sonnet 4 (`claude-sonnet-4-20250514`) - mas economico que Opus, suficiente para trading
3. **Simulacion DEMO realista**: ganancias (precio > 0.97) Y perdidas (caida > 40% desde entry)
4. **Telegram**: notificaciones opcionales (no-op si no hay token)
5. **SQLite**: sin dependencias externas, portable

## Problemas Resueltos (Historial)

### v1.0 -> v1.1 (2026-04-13)

**Problema 1: Arbitraje sum-to-one muerto**
- Causa: API Gamma devuelve precios normalizados (YES + NO = 1.0 siempre)
- Fix: Eliminada estrategia arbitraje. Reemplazada por Value Betting + Momentum
- Archivos: `strategies/arbitrage.py`, `strategies/__init__.py`, `config.yaml`

**Problema 2: Bonding no encontraba oportunidades**
- Causa: EV crudo de bonding (~0.03) nunca pasaba umbral de 0.05
- Fix: Metrica cambiada a retorno anualizado (>15%). Rango ampliado 0.80-0.97. EV threshold reducido a 0.02 para bonding
- Archivos: `strategies/bonding.py`, `config.yaml`

**Problema 3: Mercados expirados no se cerraban**
- Causa: Solo se cerraba por precio > 0.97 o caida > 40%. Mercados pasados de fecha quedaban abiertos indefinidamente
- Fix: Nuevo check en `check_and_close_positions()` que detecta `end_date` pasada y resuelve segun precio final
- Archivos: `main.py`

**Problema 4: Posiciones duplicadas**
- Causa: Claude podia comprar el mismo mercado varias veces (ej: 2 trades en Fujimori)
- Fix: `execute_trade()` ahora verifica posiciones abiertas antes de ejecutar
- Archivos: `main.py`

**Problema 5: Portfolio mostraba valor incorrecto**
- Causa: Solo contaba cash libre (capital - exposicion), no mark-to-market
- Fix: `get_portfolio_state()` ahora calcula valor real incluyendo unrealized P&L
- Archivos: `main.py`

**Problema 6: Claude apostaba en deportes/esports sin edge**
- Causa: Prompt permisivo, sin guardrails sobre areas de conocimiento
- Fix: Prompt mejorado con enfoque conservador, SKIP obligatorio en deportes, prioridad bonding > value > momentum
- Archivos: `claude_agent.py`

## Sistema de Memoria (Auto-Mejora)

4 componentes implementados en `memory.py`:

1. **Historial como contexto**: Ultimos 20 trades cerrados se inyectan en el prompt de Claude
2. **Auto-analisis post-trade**: Cada trade cerrado se analiza (sesgos, lecciones). Tabla `trade_analyses`
3. **Reglas aprendidas**: Cada 20 ciclos, Claude extrae reglas de sus analisis. Tabla `learned_rules`
4. **Ajuste de parametros**: Cada 20 ciclos, sugiere cambios a EV threshold, Kelly, etc. Tabla `parameter_adjustments`

El sistema necesita ~20+ trades cerrados para empezar a generar reglas y ajustes.

## Configuracion Actual (config.yaml)

- Capital DEMO: $1,000
- Ciclo: cada 10 minutos
- Max 3 trades por ciclo
- Max 20% exposicion total
- Stop-loss: 40% caida
- Take-profit: 80% ganancia (vende 50%)
- Kelly: 25%
- Bonding: precio 0.80-0.97, retorno anualizado >15%, max 90 dias
- Value Betting: precio 0.10-0.90, volumen >$20K, liquidez >$10K
- Momentum: volumen >$50K, ratio vol/liq >1x

## Para Continuar en Otro Ordenador

```bash
git clone https://github.com/TU_USUARIO/polymarket-bot.git
cd polymarket-bot
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Crear .env con tus claves
cp .env.example .env
# Editar .env: ANTHROPIC_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Ejecutar
start.bat                      # Windows: lanza bot + dashboard
# O manualmente:
python main.py --mode demo     # Terminal 1
python dashboard.py            # Terminal 2 -> http://localhost:5000
```

## Proximos Pasos Sugeridos

1. **Dejar correr 24-48h en DEMO** para acumular trades y activar el sistema de memoria
2. **Revisar dashboard** despues de 20+ ciclos para ver reglas aprendidas
3. **Añadir mas fuentes de datos**: noticias via API, datos de encuestas para politica
4. **Web scraping de Polymarket**: obtener precios del orderbook real (CLOB) para encontrar arbitraje real entre bid/ask
5. **Backtesting**: modulo para simular estrategias contra datos historicos
6. **Modo LIVE**: configurar wallet Polygon, empezar con $50 de capital real
7. **Multi-market arbitrage**: buscar arbitraje entre mercados relacionados (ej: "Trump gana" vs "Republicano gana")
