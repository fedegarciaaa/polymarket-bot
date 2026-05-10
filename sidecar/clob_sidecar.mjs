/**
 * Polymarket CLOB sidecar — exposes the @polymarket/clob-client-v2 SDK
 * over a local HTTP API so the Python bot can use POLY_1271 (sig_type=3)
 * orders without re-implementing ERC-7739 signing in Python.
 *
 * Listens on 127.0.0.1:8765 only (localhost). No external exposure.
 *
 * Endpoints:
 *   GET  /healthz                    — liveness probe
 *   GET  /balance                    — USDC balance on the deposit wallet
 *   GET  /book?token_id=...          — top-of-book {best_bid, bid_size, best_ask, ask_size}
 *   GET  /trades?after=<ts>          — trades since unix timestamp seconds
 *   POST /place_order                — {token_id, price, size_usdc, side, is_taker} → {order_id, raw}
 *   POST /cancel                     — {order_id} → {ok}
 *   POST /update_allowance           — refresh COLLATERAL allowance (call once at start)
 *
 * The sidecar derives or creates L2 API keys on startup and caches them
 * to data/clob_api_creds_v2.json (chmod 600). Fully separate cache from
 * the v1 creds so we don't mix.
 */
import express from "express";
import fs from "fs";
import path from "path";
import {
  ClobClient,
  SignatureTypeV2,
  Side,
  OrderType,
  AssetType,
} from "@polymarket/clob-client-v2";
import { createWalletClient, http } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { polygon } from "viem/chains";

// ─── env loading (manual; bot manages .env) ────────────────────────
const ENV_PATH = process.env.SIDECAR_ENV_PATH || "/home/botuser/polymarket-bot/.env";
if (fs.existsSync(ENV_PATH)) {
  for (const line of fs.readFileSync(ENV_PATH, "utf8").split("\n")) {
    if (!line.includes("=")) continue;
    const [k, ...v] = line.split("=");
    if (!process.env[k.trim()]) process.env[k.trim()] = v.join("=").trim();
  }
}

const PRIVATE_KEY = (process.env.POLYMARKET_PRIVATE_KEY || "").replace(/^0x/i, "");
const FUNDER = process.env.POLYMARKET_FUNDER_ADDRESS;
const RPC_URL = process.env.POLYGON_RPC_URL || "https://polygon.gateway.tenderly.co";
const HOST = process.env.SIDECAR_HOST || "127.0.0.1";
const PORT = Number(process.env.SIDECAR_PORT || 8765);
const CREDS_PATH = process.env.SIDECAR_CREDS_PATH
  || "/home/botuser/polymarket-bot/data/clob_api_creds_v2.json";

if (!PRIVATE_KEY || !FUNDER) {
  console.error("FATAL: POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS missing");
  process.exit(1);
}

const account = privateKeyToAccount("0x" + PRIVATE_KEY);
console.log(`[init] EOA: ${account.address}`);
console.log(`[init] FUNDER (deposit wallet): ${FUNDER}`);

const walletClient = createWalletClient({
  account,
  chain: polygon,
  transport: http(RPC_URL),
});

// ─── L2 auth setup ─────────────────────────────────────────────────
async function loadOrCreateCreds() {
  if (fs.existsSync(CREDS_PATH)) {
    try {
      const cached = JSON.parse(fs.readFileSync(CREDS_PATH, "utf8"));
      if (cached.api_key && cached.api_secret && cached.api_passphrase) {
        console.log(`[init] loaded cached L2 creds (key=${cached.api_key.slice(0, 8)}…)`);
        return cached;
      }
    } catch (e) {
      console.warn(`[init] cached creds unreadable: ${e.message}`);
    }
  }
  // No cache — derive (or create) via the L1 client
  const l1 = new ClobClient({
    host: "https://clob.polymarket.com",
    chain: 137,
    signer: walletClient,
    signatureType: SignatureTypeV2.POLY_1271,
    funderAddress: FUNDER,
  });
  let creds;
  try {
    console.log("[init] deriving API key (L1 sig)…");
    creds = await l1.deriveApiKey();
  } catch (e) {
    console.warn(`[init] derive failed: ${e.message}; trying create…`);
    creds = await l1.createApiKey();
  }
  const out = {
    api_key: creds.key,
    api_secret: creds.secret,
    api_passphrase: creds.passphrase,
  };
  fs.mkdirSync(path.dirname(CREDS_PATH), { recursive: true });
  fs.writeFileSync(CREDS_PATH, JSON.stringify(out, null, 2));
  fs.chmodSync(CREDS_PATH, 0o600);
  console.log(`[init] wrote new L2 creds → ${CREDS_PATH} (chmod 600)`);
  return out;
}

let clob; // populated in init()

async function init() {
  const creds = await loadOrCreateCreds();
  clob = new ClobClient({
    host: "https://clob.polymarket.com",
    chain: 137,
    signer: walletClient,
    signatureType: SignatureTypeV2.POLY_1271,
    funderAddress: FUNDER,
    creds: {
      key: creds.api_key,
      secret: creds.api_secret,
      passphrase: creds.api_passphrase,
    },
  });
  // Sync allowance — required for POLY_1271 to register CLOB buying power
  try {
    await clob.updateBalanceAllowance({ asset_type: AssetType.COLLATERAL });
    console.log("[init] balance/allowance synced");
  } catch (e) {
    console.warn(`[init] update_allowance failed (non-fatal): ${e.message}`);
  }
}

// ─── HTTP server ───────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: "256kb" }));

app.get("/healthz", (_req, res) => {
  res.json({ ok: true, eoa: account.address, funder: FUNDER });
});

app.get("/balance", async (_req, res) => {
  try {
    const r = await clob.getBalanceAllowance({ asset_type: AssetType.COLLATERAL });
    const balanceRaw = Number(r.balance ?? r.collateral?.balance ?? 0);
    res.json({ ok: true, usdc: balanceRaw / 1e6, raw: r });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.get("/book", async (req, res) => {
  const token_id = String(req.query.token_id || "");
  if (!token_id) return res.status(400).json({ ok: false, error: "token_id required" });
  try {
    const book = await clob.getOrderBook(token_id);
    const bids = book?.bids || [];
    const asks = book?.asks || [];
    const topBid = bids.length ? bids.reduce((m, x) => +x.price > +m.price ? x : m, bids[0]) : null;
    const topAsk = asks.length ? asks.reduce((m, x) => +x.price < +m.price ? x : m, asks[0]) : null;
    res.json({
      ok: true,
      best_bid: topBid ? Number(topBid.price) : 0,
      bid_size: topBid ? Number(topBid.size) : 0,
      best_ask: topAsk ? Number(topAsk.price) : 0,
      ask_size: topAsk ? Number(topAsk.size) : 0,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.get("/trades", async (req, res) => {
  const after = req.query.after ? Number(req.query.after) : undefined;
  try {
    const trades = await clob.getTrades(after ? { after } : undefined);
    res.json({ ok: true, trades: Array.isArray(trades) ? trades : [] });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.post("/place_order", async (req, res) => {
  const { token_id, price, size_usdc, side, is_taker } = req.body || {};
  if (!token_id || price == null || size_usdc == null || !side) {
    return res.status(400).json({
      ok: false, error: "token_id, price, size_usdc, side required"
    });
  }
  try {
    const sizeShares = Math.max(1, Math.round((Number(size_usdc) / Number(price)) * 1e4) / 1e4);
    const sideEnum = String(side).toUpperCase() === "BUY" ? Side.BUY : Side.SELL;
    const orderType = is_taker ? OrderType.FAK : OrderType.GTC;
    // The SDK reads tickSize and negRisk from the market metadata; we let
    // it auto-detect by passing options=undefined.
    const negRisk = await clob.getNegRisk(token_id);
    const tickSize = await clob.getTickSize(token_id);
    const resp = await clob.createAndPostOrder(
      { tokenID: token_id, price: Number(price), size: sizeShares, side: sideEnum },
      { tickSize, negRisk },
      orderType,
    );
    res.json({ ok: true, order_id: resp?.orderID || resp?.orderHash, raw: resp });
  } catch (e) {
    console.error(`[place_order] ${e.message}`);
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.post("/cancel", async (req, res) => {
  const { order_id } = req.body || {};
  if (!order_id) return res.status(400).json({ ok: false, error: "order_id required" });
  try {
    const resp = await clob.cancel({ orderID: order_id });
    res.json({ ok: true, raw: resp });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

app.post("/update_allowance", async (_req, res) => {
  try {
    const r = await clob.updateBalanceAllowance({ asset_type: AssetType.COLLATERAL });
    res.json({ ok: true, raw: r });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

// ─── boot ──────────────────────────────────────────────────────────
init()
  .then(() => {
    app.listen(PORT, HOST, () => {
      console.log(`[ready] sidecar listening at http://${HOST}:${PORT}`);
    });
  })
  .catch((e) => {
    console.error(`[fatal] init failed: ${e.message}`);
    process.exit(1);
  });
