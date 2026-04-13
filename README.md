# Polymarket Trading Bot

Bot de trading autonomo para Polymarket con analisis mediante Claude AI, sistema de auto-mejora, dashboard web y notificaciones Telegram.

## Caracteristicas

- **Estrategia Bonding**: Compra outcomes de alta probabilidad (0.88-0.97) para capturar retornos seguros
- **Arbitraje Sum-to-One**: Detecta cuando YES + NO < 1.0 para profit garantizado
- **Agente Claude**: Analiza oportunidades con razonamiento paso a paso y calculo de EV
- **Sistema de Memoria**: Auto-mejora continua con reglas aprendidas y ajuste de parametros
- **Risk Management**: Kelly Criterion al 25%, stop-losses, take-profits, limites de exposicion
- **Dashboard Web**: Monitoreo en tiempo real con graficos y estadisticas
- **Notificaciones Telegram**: Alertas de trades y errores criticos
- **Modo DEMO**: Simulacion completa sin riesgo de capital real

## Requisitos

- Python 3.10+
- Cuenta en Anthropic (API key para Claude)
- (Opcional) Token de Telegram para notificaciones
- (Solo LIVE) Wallet de Polymarket con fondos en Polygon

## Instalacion

```bash
# Clonar o copiar el proyecto
cd polymarket-bot

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu ANTHROPIC_API_KEY
```

## Configuracion

### .env
```
ANTHROPIC_API_KEY=sk-ant-tu-clave-aqui
TELEGRAM_TOKEN=tu-bot-token        # opcional
TELEGRAM_CHAT_ID=tu-chat-id        # opcional
```

### config.yaml
El archivo `config.yaml` contiene toda la configuracion del bot:

- `bot.mode`: DEMO o LIVE
- `bot.cycle_interval_minutes`: Intervalo entre ciclos (default: 10)
- `bot.demo_capital`: Capital inicial simulado (default: 1000)
- `risk.*`: Parametros de riesgo (EV minimo, Kelly fraction, stop-loss, etc.)
- `strategies.*`: Configuracion de cada estrategia
- `claude.model`: Modelo de Claude a usar
- `memory.*`: Configuracion del sistema de auto-mejora

## Ejecucion

### Modo DEMO (recomendado para empezar)
```bash
python main.py --mode demo
```

### Dashboard (en otra terminal)
```bash
python dashboard.py
# Abrir http://localhost:5000
```

### Modo LIVE (dinero real)
```bash
# Asegurate de tener POLYMARKET_PRIVATE_KEY en .env
python main.py --mode live
```

## Como funciona

### Ciclo de Trading
1. Escanea mercados activos via Gamma API
2. Revisa stop-losses y take-profits de posiciones abiertas
3. Busca oportunidades con estrategias bonding y arbitraje
4. Claude analiza las oportunidades y decide BUY o SKIP
5. Ejecuta trades (max 3 por ciclo) con validacion de riesgo
6. Sistema de memoria analiza resultados y aprende

### Sistema de Memoria (Auto-Mejora)
El bot mejora continuamente a traves de 4 mecanismos:

1. **Historial como Contexto**: Claude recibe los ultimos 20 trades cerrados para calibrar estimaciones
2. **Auto-Analisis Post-Trade**: Cada trade cerrado se analiza para identificar sesgos y extraer lecciones
3. **Reglas Aprendidas**: Cada 20 ciclos, el bot extrae reglas de sus analisis (ej: "evitar mercados politicos a >30 dias")
4. **Ajuste de Parametros**: Cada 20 ciclos, revisa y ajusta EV threshold, Kelly fraction, etc.

### Estrategias

**Bonding**: Compra outcomes con probabilidad alta (88-97%) que estan cerca de resolverse. El retorno es pequeno pero casi seguro, similar a un bono.

**Arbitraje Sum-to-One**: Cuando la suma de precios YES + NO es menor a ~0.985, comprar ambos lados garantiza un profit al resolver el mercado.

## Estructura del Proyecto

```
polymarket-bot/
├── main.py              # Orquestador principal
├── config.yaml          # Configuracion
├── polymarket_api.py    # Cliente API Gamma/CLOB
├── claude_agent.py      # Agente Claude AI
├── risk_manager.py      # Gestion de riesgo
├── database.py          # SQLite persistencia
├── memory.py            # Sistema de auto-mejora
├── dashboard.py         # Dashboard Flask
├── notifications.py     # Notificaciones Telegram
├── strategies/
│   ├── __init__.py
│   ├── bonding.py       # Estrategia bonding
│   └── arbitrage.py     # Estrategia arbitraje
├── requirements.txt
├── .env.example
├── data/                # Base de datos SQLite (auto-creado)
└── logs/                # Logs diarios (auto-creado)
```

## Paso a LIVE

1. Configura tu wallet de Polymarket en Polygon
2. Agrega `POLYMARKET_PRIVATE_KEY` a `.env`
3. Cambia `bot.mode: LIVE` en `config.yaml` o ejecuta con `--mode live`
4. Empieza con capital bajo y parametros conservadores
5. Monitorea el dashboard y logs constantemente

## Seguridad

- **NUNCA** compartas tu `.env` o private key
- En modo DEMO no se hacen llamadas a endpoints de ejecucion
- Los stop-losses protegen contra perdidas excesivas
- El Kelly Criterion al 25% es conservador por diseno
- Limite de exposicion total del 20% del portfolio

## Logs

Los logs se guardan en `logs/bot_YYYY-MM-DD.log` con rotacion diaria. Colores en consola para facilitar lectura.
