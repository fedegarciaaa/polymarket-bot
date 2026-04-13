"""
Dashboard - Flask web interface for monitoring the bot.
Run: python dashboard.py
"""

import os
import yaml
from flask import Flask, jsonify, render_template_string
from database import Database

app = Flask(__name__)

# Load config
CONFIG_PATH = os.environ.get("BOT_CONFIG", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = CONFIG.get("database", {}).get("path", "data/bot.db")
INITIAL_CAPITAL = CONFIG["bot"]["demo_capital"]
REFRESH_INTERVAL = CONFIG.get("dashboard", {}).get("refresh_interval_seconds", 60)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="{{ refresh_interval }}">
    <title>Polymarket Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117;
            color: #e1e4e8;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #2d333b;
        }
        .header h1 { font-size: 24px; color: #58a6ff; }
        .badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
        }
        .badge-demo { background: #d29922; color: #000; }
        .badge-live { background: #238636; color: #fff; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: #161b22;
            border: 1px solid #2d333b;
            border-radius: 12px;
            padding: 20px;
        }
        .card-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
        .card-value { font-size: 28px; font-weight: 700; margin-top: 8px; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #e1e4e8; }
        .chart-container {
            background: #161b22;
            border: 1px solid #2d333b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .chart-container h2 { font-size: 16px; color: #8b949e; margin-bottom: 16px; }
        .section-title {
            font-size: 18px;
            color: #58a6ff;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2d333b;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #161b22;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 24px;
        }
        th {
            background: #1c2129;
            padding: 12px 16px;
            text-align: left;
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            padding: 10px 16px;
            border-top: 1px solid #2d333b;
            font-size: 13px;
        }
        tr:hover { background: #1c2129; }
        .status-open { color: #58a6ff; }
        .status-closed { color: #8b949e; }
        .status-simulated { color: #d29922; }
        .rules-container {
            background: #161b22;
            border: 1px solid #2d333b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .rule-item {
            padding: 10px 0;
            border-bottom: 1px solid #2d333b;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rule-item:last-child { border-bottom: none; }
        .rule-text { flex: 1; margin-right: 12px; }
        .rule-meta { font-size: 11px; color: #8b949e; white-space: nowrap; }
        .effectiveness-bar {
            display: inline-block;
            width: 60px;
            height: 6px;
            background: #2d333b;
            border-radius: 3px;
            overflow: hidden;
            margin-left: 8px;
        }
        .effectiveness-fill {
            height: 100%;
            border-radius: 3px;
            background: #3fb950;
        }
        .adjustments-table td { font-size: 12px; }
        .footer {
            text-align: center;
            padding: 20px;
            color: #484f58;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Polymarket Bot Dashboard</h1>
        <span class="badge {{ 'badge-demo' if stats.get('mode', 'DEMO') == 'DEMO' else 'badge-live' }}">
            {{ stats.get('mode', 'DEMO') }}
        </span>
    </div>

    <div class="cards">
        <div class="card">
            <div class="card-label">Capital Actual</div>
            <div class="card-value neutral">${{ "%.2f"|format(stats.current_portfolio_value) }}</div>
        </div>
        <div class="card">
            <div class="card-label">P&L Total</div>
            <div class="card-value {{ 'positive' if stats.total_pnl >= 0 else 'negative' }}">
                ${{ "%+.4f"|format(stats.total_pnl) }}
            </div>
        </div>
        <div class="card">
            <div class="card-label">ROI</div>
            <div class="card-value {{ 'positive' if stats.roi_pct >= 0 else 'negative' }}">
                {{ "%+.2f"|format(stats.roi_pct) }}%
            </div>
        </div>
        <div class="card">
            <div class="card-label">Win Rate</div>
            <div class="card-value {{ 'positive' if stats.win_rate >= 50 else 'negative' }}">
                {{ "%.1f"|format(stats.win_rate) }}%
            </div>
        </div>
        <div class="card">
            <div class="card-label">Trades Totales</div>
            <div class="card-value neutral">{{ stats.total_trades }}</div>
        </div>
        <div class="card">
            <div class="card-label">Posiciones Abiertas</div>
            <div class="card-value neutral">{{ stats.open_positions }}</div>
        </div>
        <div class="card">
            <div class="card-label">Max Drawdown</div>
            <div class="card-value negative">{{ "%.2f"|format(stats.max_drawdown) }}%</div>
        </div>
        <div class="card">
            <div class="card-label">Reglas Aprendidas</div>
            <div class="card-value neutral">{{ rules|length }}</div>
        </div>
    </div>

    <div class="chart-container">
        <h2>Evolucion del Capital</h2>
        <canvas id="capitalChart" height="80"></canvas>
    </div>

    <!-- Learned Rules Section -->
    {% if rules %}
    <h2 class="section-title">Memoria del Bot - Reglas Aprendidas</h2>
    <div class="rules-container">
        {% for rule in rules %}
        <div class="rule-item">
            <div class="rule-text">
                <span style="color: #58a6ff; font-size: 11px;">[{{ rule.category }}]</span>
                {{ rule.rule_text }}
            </div>
            <div class="rule-meta">
                Conf: {{ "%.0f"|format(rule.confidence * 100) }}%
                <span class="effectiveness-bar">
                    <span class="effectiveness-fill" style="width: {{ rule.effectiveness_pct }}%"></span>
                </span>
                {{ "%.0f"|format(rule.effectiveness_pct) }}%
                | x{{ rule.times_applied }}
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Parameter Adjustments -->
    {% if adjustments %}
    <h2 class="section-title">Ajustes de Parametros</h2>
    <table class="adjustments-table">
        <thead>
            <tr>
                <th>Fecha</th>
                <th>Parametro</th>
                <th>Anterior</th>
                <th>Nuevo</th>
                <th>Razon</th>
            </tr>
        </thead>
        <tbody>
            {% for adj in adjustments %}
            <tr>
                <td>{{ adj.timestamp[:16] }}</td>
                <td>{{ adj.parameter_name }}</td>
                <td>{{ "%.4f"|format(adj.old_value) }}</td>
                <td>{{ "%.4f"|format(adj.new_value) }}</td>
                <td>{{ adj.reason[:80] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}

    <h2 class="section-title">Ultimos Trades</h2>
    <table>
        <thead>
            <tr>
                <th>Fecha</th>
                <th>Mercado</th>
                <th>Estrategia</th>
                <th>Accion</th>
                <th>Precio</th>
                <th>Size</th>
                <th>EV</th>
                <th>P&L</th>
                <th>Estado</th>
            </tr>
        </thead>
        <tbody>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.timestamp[:16] }}</td>
                <td title="{{ trade.market_question }}">{{ trade.market_question[:40] }}{% if trade.market_question|length > 40 %}...{% endif %}</td>
                <td>{{ trade.strategy or '-' }}</td>
                <td>{{ trade.action }} {{ trade.side or '' }}</td>
                <td>{{ "%.4f"|format(trade.price_entry or 0) }}</td>
                <td>${{ "%.2f"|format(trade.size_usdc or 0) }}</td>
                <td>{{ "%.4f"|format(trade.ev_calculated or 0) }}</td>
                <td class="{{ 'positive' if (trade.profit_loss or 0) > 0 else 'negative' if (trade.profit_loss or 0) < 0 else 'neutral' }}">
                    ${{ "%+.4f"|format(trade.profit_loss or 0) }}
                </td>
                <td class="status-{{ trade.status|lower }}">{{ trade.status }}</td>
            </tr>
            {% endfor %}
            {% if not trades %}
            <tr><td colspan="9" style="text-align: center; color: #484f58;">No hay trades todavia</td></tr>
            {% endif %}
        </tbody>
    </table>

    <div class="footer">
        Polymarket Bot Dashboard | Auto-refresh: {{ refresh_interval }}s | Powered by Claude Sonnet
    </div>

    <script>
        const cyclesData = {{ cycles_json|safe }};
        const labels = cyclesData.map((c, i) => 'C' + (i + 1));
        const values = cyclesData.map(c => c.portfolio_value);

        new Chart(document.getElementById('capitalChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Portfolio Value ($)',
                    data: values,
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                    pointBackgroundColor: '#58a6ff',
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        grid: { color: '#2d333b' },
                        ticks: { color: '#8b949e', maxTicksLimit: 20 },
                    },
                    y: {
                        grid: { color: '#2d333b' },
                        ticks: { color: '#8b949e', callback: v => '$' + v.toFixed(2) },
                    }
                }
            }
        });
    </script>
</body>
</html>
"""


def get_db():
    return Database(DB_PATH)


@app.route("/")
def index():
    db = get_db()
    stats = db.get_statistics(INITIAL_CAPITAL)
    stats["mode"] = CONFIG["bot"].get("mode", "DEMO")
    trades_data = db.get_trades_paginated(page=1, per_page=20)
    cycles = db.get_cycles_history(100)
    cycles.reverse()  # oldest first for chart
    rules = db.get_learned_rules(active_only=True)
    adjustments = db.get_parameter_adjustments(10)
    db.close()

    import json
    cycles_json = json.dumps(cycles)

    return render_template_string(
        HTML_TEMPLATE,
        stats=stats,
        trades=trades_data["trades"],
        cycles_json=cycles_json,
        rules=rules,
        adjustments=adjustments,
        refresh_interval=REFRESH_INTERVAL,
    )


@app.route("/api/stats")
def api_stats():
    db = get_db()
    stats = db.get_statistics(INITIAL_CAPITAL)
    stats["mode"] = CONFIG["bot"].get("mode", "DEMO")
    performance = db.get_performance_summary()
    rules = db.get_learned_rules()
    db.close()

    stats["performance"] = performance
    stats["learned_rules_count"] = len(rules)
    return jsonify(stats)


@app.route("/api/trades")
def api_trades():
    from flask import request
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    db = get_db()
    result = db.get_trades_paginated(page, per_page)
    db.close()
    return jsonify(result)


@app.route("/api/portfolio")
def api_portfolio():
    db = get_db()
    portfolio_value = db.get_portfolio_value(INITIAL_CAPITAL)
    exposure = db.get_total_exposure()
    open_positions = db.get_open_positions()
    stats = db.get_statistics(INITIAL_CAPITAL)
    rules = db.get_learned_rules(active_only=True)
    adjustments = db.get_parameter_adjustments(5)
    db.close()

    return jsonify({
        "portfolio_value": portfolio_value,
        "available_capital": portfolio_value - exposure,
        "total_exposure": exposure,
        "open_positions": [
            {
                "id": p["id"],
                "market_question": p["market_question"],
                "side": p["side"],
                "price_entry": p["price_entry"],
                "size_usdc": p["size_usdc"],
                "strategy": p["strategy"],
            }
            for p in open_positions
        ],
        "total_pnl": stats["total_pnl"],
        "win_rate": stats["win_rate"],
        "roi_pct": stats["roi_pct"],
        "active_rules": len(rules),
        "recent_adjustments": adjustments,
    })


@app.route("/api/memory")
def api_memory():
    db = get_db()
    rules = db.get_learned_rules(active_only=False)
    adjustments = db.get_parameter_adjustments(20)
    analyses = db.get_recent_analyses(20)
    db.close()

    return jsonify({
        "rules": rules,
        "adjustments": adjustments,
        "recent_analyses": analyses,
    })


if __name__ == "__main__":
    host = CONFIG.get("dashboard", {}).get("host", "0.0.0.0")
    port = CONFIG.get("dashboard", {}).get("port", 5000)
    print(f"Dashboard running on http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
