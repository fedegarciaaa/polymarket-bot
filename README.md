# Polymarket Weather Bot v2

Bot autonomo enfocado exclusivamente en mercados meteorologicos de Polymarket.
El nucleo del sistema es un **ensemble multi-fuente** de forecasts combinado con
un **motor de confidence** que decide si la oportunidad merece ser apostada.

Claude AI actua como revisor final (sanity check, no recalcula el forecast).

## Caracteristicas

- **Ensemble multi-fuente**: 4 APIs gratuitas (Open-Meteo, NOAA GFS, ECMWF Open Data, MET Norway)
  + 4 APIs opcionales con key (Visual Crossing, OpenWeatherMap, WeatherAPI, Tomorrow.io).
  El bot funciona en modo degradado con las fuentes disponibles.
- **Confidence engine 0-100** con componentes ponderados (acuerdo del ensemble,
  edge, liquidez, tiempo a resolucion, n. de fuentes) y vetos duros
  (pocas fuentes, std alta, edge sospechoso, rolling WR malo del lado).
- **Pyramiding adaptativo**: solo permitido con `confidence >= 80` y condiciones
  de mercado cambiadas.
- **PID lock**: garantia de un solo proceso (psutil). Al arrancar, si hay otro
  bot corriendo aborta; si el lock es huerfano lo limpia.
- **Shadow mode**: ejecuta todo el pipeline sin enviar ordenes; ideal para
  backtest en vivo.
- **Logging estructurado** (`logs/events.jsonl`) con `trace_id` por trade y
  codigos de skip enumerados. `log_analyzer.py` genera reportes markdown.
- **Dashboard web** con: health panel, fiabilidad por fuente (Brier), chart
  de calibracion, motivos de skip, trades abiertos/cerrados.
- **Notificaciones Telegram** para trades, errores, fuente caida, anomalias
  de confidence y violaciones de lock.

## Arquitectura

```
main.py
  +-- PolymarketAPI           (scan_weather_markets)
  +-- WeatherBotStrategy       (async find_opportunities)
  |     +-- weather_sources/   (cada fuente implementa .forecast())
  |     +-- weather_ensemble   (media ponderada + std + prob)
  |     +-- confidence_engine  (score + vetos)
  +-- RiskManager              (can_add_to_market + pyramiding + Kelly)
  +-- ClaudeAgent              (sanity check final)
  +-- Database / StructuredLogger / TelegramNotifier / MemorySystem
```

## Requisitos

- Python 3.10+
- `pip install -r requirements.txt`
- `ANTHROPIC_API_KEY` (obligatoria)
- API keys opcionales para fuentes premium (ver `.env.example`)

## Quick start

```bash
# 1. Instalar deps
pip install -r requirements.txt

# 2. Configurar .env con ANTHROPIC_API_KEY
cp .env.example .env
# edita .env

# 3. Lanzar
start.bat
```

Menu de `start.bat`:

| Opc. | Modo |
|------|------|
| 1 | DEMO - simulacion con ordenes fake |
| 2 | LIVE - ordenes reales (pide `CONFIRMO`) |
| 3 | Solo dashboard |
| 4 | Shadow mode (EXECUTE_TRADES=false) |
| 5 | Analizar logs ultimas 24h (markdown) |
| 6 | Backup + wipe datos (a `data/archive/YYYYMMDD_HHMM/`) |

## Ficheros clave

| Fichero | Descripcion |
|---------|-------------|
| [main.py](main.py) | Loop principal + PID lock + shadow mode |
| [config.yaml](config.yaml) | Thresholds (min_confidence, ventanas horarias, Kelly, etc.) |
| [strategies/weather_bot.py](strategies/weather_bot.py) | Parser de preguntas + pipeline async |
| [strategies/weather_ensemble.py](strategies/weather_ensemble.py) | Agregador de forecasts |
| [strategies/confidence_engine.py](strategies/confidence_engine.py) | Score + vetos |
| [strategies/weather_sources/](strategies/weather_sources/) | Adaptadores por proveedor |
| [risk_manager.py](risk_manager.py) | Kelly + pyramiding + limites |
| [database.py](database.py) | SQLite: trades, skips, source_reliability, cycles |
| [structured_logger.py](structured_logger.py) | JSONL event log |
| [dashboard.py](dashboard.py) | Flask + Chart.js |
| [log_analyzer.py](log_analyzer.py) | Reporte markdown desde DB+JSONL |
| [claude_agent.py](claude_agent.py) | Revisor final weather-only |

## Umbrales relevantes (config.yaml)

```yaml
weather:
  min_edge: 0.12
  kelly_fraction: 0.15
  max_position_pct: 0.03
  min_confidence: 75
  min_confidence_pyramid: 80
  ensemble_std_max_c: 3.0
  ensemble_std_max_mm: 8.0
  max_trades_per_cycle: 2
  max_concurrent_positions: 4
  max_total_exposure_pct: 0.20
  max_exposure_per_market_pct: 0.15
  trading_windows_utc: [[0, 2], [4, 7]]
```

## Criterio para pasar de shadow -> LIVE

72h acumuladas (shadow + smoke) con:
- WR >= 55% sobre >= 10 trades
- PnL positivo
- Dashboard health verde (>=3 fuentes disponibles)

## Limitaciones conocidas

- Parser usa `15:00 UTC` como aproximacion para mercados daily-max-temp.
- ECMWF Open Data llega como GRIB2: se cachea por ciclo.
- Sin API keys premium el ensemble corre con 3-4 fuentes (funcional pero con
  mayor std y menos sensibilidad a outliers).

## Logs y analisis

- Logs legibles: `logs/bot_YYYY-MM-DD.log` (rotacion diaria, sin borrado)
- Eventos estructurados: `logs/events.jsonl`
- Reporte manual: `python log_analyzer.py --last-hours 24 --output reports/`
- Dashboard en `http://localhost:5000` con auto-refresh 30s

## Soporte

- `/help` - no disponible, este es un proyecto personal
- Feedback / bugs: issues en el repo local
