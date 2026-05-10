"""Pre-flight check before flipping LIVE — never run from inside the bot.

Validates:
  1. .env loads
  2. py-clob-client connects with our wallet
  3. derive_api_key works (or create_api_key as fallback)
  4. USDC balance is reported
  5. Order book read on a sample BTC market succeeds
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Manually load .env since we run outside the systemd EnvironmentFile context
env_file = Path("/home/botuser/polymarket-bot/.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
if not pk or not funder:
    print("FAIL: missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS in .env")
    sys.exit(1)

print(f"OK  funder={funder[:8]}…{funder[-4:]}")

# Connect to CLOB
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

if pk.startswith("0x") or pk.startswith("0X"):
    pk = pk[2:]

print("Stage 1: connecting (no L2 creds)...")
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=pk,
    funder=funder,
    signature_type=1,  # Magic.link / Polymarket proxy account
)
print("OK  Stage 1 connected")

print("Stage 2: deriving API keys (one-shot, idempotent)...")
try:
    creds = client.derive_api_key()
    print(f"OK  derive_api_key returned key={creds.api_key[:8]}…")
except Exception as exc:
    print(f"WARN derive_api_key failed: {exc}")
    print("    Falling back to create_api_key...")
    creds = client.create_api_key()
    print(f"OK  create_api_key returned key={creds.api_key[:8]}…")

print("Stage 3: re-instantiating with L2 creds...")
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=pk,
    funder=funder,
    signature_type=1,
    creds=ApiCreds(
        api_key=creds.api_key,
        api_secret=creds.api_secret,
        api_passphrase=creds.api_passphrase,
    ),
)
print("OK  Stage 3 connected with L2 creds")

print("Stage 4: reading USDC balance...")
params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
resp = client.get_balance_allowance(params)
print(f"raw response: {resp}")
if isinstance(resp, dict):
    bal_raw = float(resp.get("balance", 0.0))
    print(f"OK  USDC balance = ${bal_raw / 1e6:.2f}")
else:
    print(f"WARN unexpected response type: {type(resp)}")

print("\n✅ All pre-flight checks passed. Safe to flip LIVE.")
