# Crypto-Lag LIVE Smoke — Operations Runbook

Documento operativo del primer smoke test con dinero real ($100 USDC) en
Polymarket CLOB. Lectura obligatoria antes de cualquier modificación
posterior al stack LIVE.

## 1. Estado actual del despliegue

| Item | Valor |
|------|-------|
| Variante LIVE | `live_focused` |
| Modelo subyacente | maker_focused (BS-digital + AS, edge=3c, T<60m, spread≥2t) |
| Capital nominal | $100 USDC (real wallet: $113.14 al deposit) |
| Halt threshold | -$50 lifetime PnL |
| Whitelist mercados | BTCUSDT, ETHUSDT |
| Max order notional | $5 USDC |
| Max concurrent orders | 5 |
| Funder address | `0x263984…1f5D` |
| Signature type | 1 (Polymarket / Magic.link proxy) |
| Cred cache | `data/clob_api_creds.json` (chmod 600) |

Variantes paper (para comparación side-by-side):
`maker_focused`, `maker_wide_spread`, `maker_near_30m`, `maker_near_60m`,
`maker_near_2h`, `maker_selective`, `maker_aggressive`.

## 2. Arquitectura del stack LIVE

```
Binance feed ─────┐
Deribit IV ───────┤
                  ├─→ probability_model (BS-digital)
                  ├─→ MakerOrderEngine (AS reservation + extreme guard F5)
                  │       │
                  │       ▼
                  └─→ LiveExecutor.place_order(order, market.token_yes)
                          │
                          ├─→ py-clob-client.create_order + post_order
                          ├─→ Sanity guards (whitelist, size, price, halt latch)
                          └─→ Logs LIVE-PLACE / LIVE-FILL / LIVE-CLOSE

Telegram listener (daemon thread)
   /setpk, /setfunder      — overwrite .env
   /halt_live              — touch data/halt_live (instant kill)
   /unhalt_live            — remove file (latch persists in memory)
   /restart_bot            — exit(0); systemd reloads .env
   /clag_status            — quick reply with bankroll/PnL/halts
```

## 3. Logs forensic-grade (`journalctl -fu polymarket-bot`)

### `LIVE-PLACE`
```
LIVE-PLACE [live_focused] BUY BTCUSDT slug=btc-updown-5m-... \
  px=0.4800 size_shares=10.4167 size_usdc=$5.00 \
  is_taker=False ext=0xabc123... resp_keys=[orderID, status, ...]
```
Campos clave:
- `px` — precio cotizado (lo que el modelo decidió)
- `size_shares` — shares = notional / price (CLOB usa shares, no USDC)
- `ext` — orderID retornado por el CLOB (rastreable on-chain)
- `resp_keys` — campos del response del CLOB; útil para debug si la API cambia

### `LIVE-FILL`
```
LIVE-FILL [live_focused] BUY BTCUSDT slug=... \
  placed_px=0.4800 actual_px=0.4810 \
  shares=10.4167 usdc=$5.01 fee=$0.0451 \
  slippage_pct=+0.208 ext=0xabc123
```
**`slippage_pct`** es la métrica clave para calibrar el simulador post-mortem:
- En paper, `actual_px = placed_px` (sin slippage)
- En LIVE, `actual_px ≠ placed_px` por movimientos del book entre place y match
- Si slippage >0.5% sostenido → el `adverse_haircut_pct` del simulador (1.5%) es OK; si <0.5% sostenido → el simulador over-castiga

### `LIVE-CLOSE`
```
LIVE-CLOSE [live_focused] BTCUSDT cid=0x... \
  outcome=YES entry_px=0.4810 yes_final=1.00 \
  shares=10.42 cost=$5.01 payoff=$10.42 pnl=+$5.41 \
  lifetime=+$5.41 slug=...
```
Reconstrucción manual:
- `payoff = shares × yes_final` (si outcome=YES) o `shares × (1-yes_final)` (NO)
- `pnl = sign × (payoff - cost)` donde sign=+1 long, -1 short
- `lifetime` actualiza el contador interno del executor que dispara el halt

## 4. Limitaciones conocidas del primer smoke

1. **Bot solo trades el token YES de cada market.** Si quiere vender YES
   pero no tiene inventory, la API CLOB rechazará con "insufficient
   balance". Esperar muchos errores `LIVE-PLACE` rejected los primeros
   minutos hasta que acumulemos posición. No es bug.
2. **Fee real no contabilizado en lifetime PnL.** El close usa
   `cost - payoff` sin restar fees pagados en el camino. La estimación
   del halt es por tanto OPTIMISTA; el halt real puede llegar antes que
   el calculado por una cantidad de orden ~3-5% del PnL.
3. **`get_trades` poll cada cycle.** No hay websocket de fills; usamos
   REST con `TradeParams(after=last_trade_ts)`. Latencia típica de
   detección: 1-3 seconds entre fill on-chain y nuestro registro.
4. **Halt latch en memoria.** Una vez tripped, queda halted incluso si
   se borra el file. Requiere `/restart_bot` para re-armar. Esto es
   intencional (defensa contra clears accidentales).
5. **Allowances ya aprobadas (MAX) en wallet del usuario.** Verificado
   en pre-flight; si en el futuro se rotan, el flujo de re-approval
   está manejado por la UI de Polymarket, no por nuestro bot.

## 5. Comandos operativos

### Vía Telegram (sin SSH)
```
/clag_status     — bankroll + PnL + halt + open orders
/halt_live       — kill switch instantáneo
/unhalt_live     — elimina halt file (necesita /restart_bot después)
/restart_bot     — exit(0); systemd reinicia, .env recargado
/setpk <hex>     — actualizar private key (auto-borra mensaje)
/setfunder 0x... — actualizar funder address
/help            — listado completo
```

### Vía SSH (operario)
```bash
# Status
ssh polymarket "sudo systemctl status polymarket-bot.service --no-pager | head -10"

# Logs LIVE en tiempo real
ssh polymarket "sudo journalctl -fu polymarket-bot.service | grep LIVE-"

# Solo placements
ssh polymarket "sudo journalctl -u polymarket-bot.service --since '1h ago' | grep LIVE-PLACE | wc -l"

# Solo fills
ssh polymarket "sudo journalctl -u polymarket-bot.service --since '1h ago' | grep LIVE-FILL"

# Closes con PnL
ssh polymarket "sudo journalctl -u polymarket-bot.service --since '1h ago' | grep LIVE-CLOSE"

# Halt manual (file-based, equivalente a /halt_live)
ssh polymarket "sudo -u botuser touch /home/botuser/polymarket-bot/data/halt_live"

# Re-arm tras halt (rm + restart)
ssh polymarket "sudo -u botuser rm /home/botuser/polymarket-bot/data/halt_live && sudo systemctl restart polymarket-bot.service"
```

### Audit SQL (para extraer datos al final del smoke)
Script disponible en `/tmp/audit_quick.py` (ver código en repo:
`audit_quick.py`). Ejecutar con:
```bash
ssh polymarket "sudo -u botuser /home/botuser/polymarket-bot/venv/bin/python3 /tmp/audit_quick.py"
```

## 6. Qué observar en las primeras 24 horas

### Hour 0-1: validación del wiring
- ✅ Esperado: 1 mensaje Telegram con balance USDC al startup
- ✅ Esperado: primeros `LIVE-PLACE` logs (BUY YES en BTC/ETH)
- ⚠️ Esperado: errores `insufficient balance` en intentos de SELL
  iniciales (sin inventory) → no son fallos del bot
- ❌ Si: 0 LIVE-PLACE en 5 min → algo falló en wiring → revisar logs
  para excepción de `py-clob-client` y diagnose

### Hour 1-6: primer fills + slippage real
- Calcular `slippage_pct` promedio de los `LIVE-FILL`
- Comparar fill rate (`fills/placements`) con la variante paper
  `maker_focused` corriendo en paralelo
- Anotar mercados (cid) que generan más fills — útil para whitelist
  más estrecha en próximo smoke

### Hour 6-24: pattern emergence
- `lifetime` PnL trend: ¿bleed o profit?
- Confirmar que el hourly summary Telegram llega a las :07 (cron + variant_stats)
- Verificar que el halt automático no se trippeó si PnL > -50

### Si halt trippea
- Telegram avisará inmediatamente (`HALT ACTIVE` en summary)
- Bot cancela todas las órdenes resting
- Posiciones abiertas siguen on-chain hasta resolución natural
- **NO restartear sin diagnose** — los logs entre las últimas N órdenes
  y el halt son la evidencia más valiosa

## 7. Análisis post-smoke (template)

Al final del smoke (sea por halt o decisión manual):

1. **Extract LIVE-* logs**:
   ```bash
   ssh polymarket "sudo journalctl -u polymarket-bot.service --since '24h ago' | grep -E 'LIVE-(PLACE|FILL|CLOSE|HALT)'" > smoke_logs.txt
   ```

2. **Métricas clave a calcular**:
   - Total placements vs fills → fill rate real
   - Distribución `slippage_pct` (median, p90)
   - PnL/close real vs paper (¿coincide la dirección?)
   - Adverse fill rate inferido (fills donde el price se mueve contra
     nosotros en los 30s post-fill)
   - Tiempo medio entre place y first fill (para calibrar polling lag)

3. **Comparación paper-vs-LIVE en mismos cids**:
   - Match cids tradeados por ambas variantes
   - Calcular ratio de PnL paper / LIVE → lo que el simulator infla

4. **Decisiones derivables**:
   - Si slippage real > 1% → re-tunear `adverse_haircut_pct` en paper
   - Si fill rate LIVE << paper → race-lost real es mayor → tunear F2
   - Si PnL real ≈ 0 con muchos fills → modelo no tiene edge real
     en mid-band → invalidate the strategy as-is
   - Si PnL real positivo modesto → confirma edge real → escalar a $500

## 8. Próximas modificaciones esperadas (backlog post-smoke)

Documentadas para mantener contexto entre iteraciones:

- **F8** (planificado): modelar polling staleness en `paper_executor`.
  Cuando el book leído tiene >1.5s de antigüedad, aplicar slippage
  proporcional a la volatilidad observada en ese intervalo.
- **F9** (condicional): si fill rate LIVE > 50%, considerar añadir
  shorting via NO token (segundo book). Necesita refactor de
  `order_engine` para tracking de inventario por token, no por cid.
- **F10** (condicional): retry-with-backoff en `place_order` cuando
  el CLOB devuelva 429 (rate limit). Hoy: una orden rejected es
  perdida sin retry.

## 9. Archivos críticos del stack LIVE

| Archivo | Rol |
|---------|-----|
| `strategies/crypto_lag/live_executor.py` | LiveExecutor: place/cancel/poll/book/resolve + halt |
| `strategies/crypto_lag/cycle.py` | Mode-aware notification + hourly summary trigger |
| `strategies/telegram_commands.py` | Operator commands + .env writer |
| `crypto_lag_runner.py` | Per-variant mode → LiveExecutor vs PaperExecutor |
| `notifications.py:notify_crypto_lag_hourly_summary` | Hourly Telegram digest |
| `config.yaml > crypto_lag.variants.live_focused` | Activación + safety params |
| `data/clob_api_creds.json` | API L2 cache (chmod 600) |
| `data/halt_live` | Kill-switch file (touch para parar) |
| `data/.tg_cmd_offset` | Telegram poll offset persistence |
| `/home/botuser/polymarket-bot/.env` | PRIVATE_KEY + FUNDER_ADDRESS + TG creds |

## 10. Contactos / referencias externas

- Polymarket CLOB docs: https://docs.polymarket.com/
- py-clob-client repo: https://github.com/Polymarket/py-clob-client
- Polymarket account / deposit: https://polymarket.com/wallet
- Polygonscan (verificar tx on-chain): https://polygonscan.com/address/0x263984…1f5D

---

**Last updated**: 2026-05-10 — primer arranque LIVE.
**Próxima revisión**: cuando concluya el smoke (halt o manual stop).
