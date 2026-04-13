"""
Polymarket API client - Gamma API for market data, CLOB API for trading.
"""

import json
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("polymarket_bot.api")


class PolymarketAPI:
    def __init__(self, config: dict):
        self.gamma_base = config["apis"]["gamma_base"]
        self.clob_base = config["apis"]["clob_base"]
        self.delay = config["apis"].get("cycle_delay_seconds", 0.5)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketBot/1.0",
        })
        logger.info("PolymarketAPI initialized")

    def _request_with_retry(self, method: str, url: str, params: dict = None,
                            json_data: dict = None, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay)
                if method == "GET":
                    resp = self.session.get(url, params=params, timeout=30)
                elif method == "POST":
                    resp = self.session.post(url, json=json_data, timeout=30)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error {resp.status_code} on {url}: {e}")
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    logger.warning(f"Server error. Waiting 30s before retry {attempt + 1}/{max_retries}")
                    time.sleep(30)
                else:
                    return None
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on {url}: {e}")
                logger.warning(f"Waiting 30s before retry {attempt + 1}/{max_retries}")
                time.sleep(30)
            except requests.exceptions.Timeout:
                logger.error(f"Timeout on {url}")
                logger.warning(f"Waiting 10s before retry {attempt + 1}/{max_retries}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected error on {url}: {e}")
                return None

        logger.error(f"All {max_retries} retries exhausted for {url}")
        return None

    def scan_markets(self, min_volume: float = 10000, min_liquidity: float = 5000) -> list[dict]:
        url = f"{self.gamma_base}/markets"
        params = {
            "closed": "false",
            "limit": 100,
            "order": "volume24hr",
            "ascending": "false",
        }

        data = self._request_with_retry("GET", url, params=params)
        if not data:
            logger.error("Failed to fetch markets from Gamma API")
            return []

        markets = []
        for market in data:
            try:
                volume_24h = float(market.get("volume24hr", 0) or 0)
                liquidity = float(market.get("liquidityClob", 0) or 0)

                if volume_24h < min_volume or liquidity < min_liquidity:
                    continue

                outcome_prices_raw = market.get("outcomePrices", "[]")
                if isinstance(outcome_prices_raw, str):
                    outcome_prices = json.loads(outcome_prices_raw)
                else:
                    outcome_prices = outcome_prices_raw

                prices = [float(p) for p in outcome_prices]
                if len(prices) < 2:
                    continue

                parsed = {
                    "id": market.get("id", ""),
                    "condition_id": market.get("conditionId", ""),
                    "question": market.get("question", ""),
                    "description": market.get("description", ""),
                    "end_date": market.get("endDate", ""),
                    "price_yes": prices[0],
                    "price_no": prices[1],
                    "volume_24h": volume_24h,
                    "liquidity": liquidity,
                    "category": market.get("category", ""),
                    "slug": market.get("slug", ""),
                    "clob_token_ids": market.get("clobTokenIds", ""),
                    "active": market.get("active", True),
                }
                markets.append(parsed)

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing market {market.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Scanned {len(data)} markets, {len(markets)} passed filters (vol>={min_volume}, liq>={min_liquidity})")
        return markets

    def get_market_prices(self, market_id: str, token_id: str = None) -> Optional[dict]:
        if token_id:
            url = f"{self.clob_base}/book"
            params = {"token_id": token_id}
        else:
            url = f"{self.gamma_base}/markets/{market_id}"
            params = None

        data = self._request_with_retry("GET", url, params=params)
        if not data:
            return None

        try:
            if token_id:
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                best_bid = float(bids[0]["price"]) if bids else 0.0
                best_ask = float(asks[0]["price"]) if asks else 1.0
                return {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price": (best_bid + best_ask) / 2,
                    "spread": best_ask - best_bid,
                }
            else:
                outcome_prices_raw = data.get("outcomePrices", "[]")
                if isinstance(outcome_prices_raw, str):
                    prices = [float(p) for p in json.loads(outcome_prices_raw)]
                else:
                    prices = [float(p) for p in outcome_prices_raw]

                return {
                    "price_yes": prices[0] if len(prices) > 0 else 0.5,
                    "price_no": prices[1] if len(prices) > 1 else 0.5,
                }

        except Exception as e:
            logger.error(f"Error parsing prices for market {market_id}: {e}")
            return None

    def place_order_demo(self, market_id: str, side: str, size_usdc: float,
                         price: float, market_question: str = "") -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        shares = size_usdc / price if price > 0 else 0

        order = {
            "order_id": f"DEMO-{timestamp}",
            "market_id": market_id,
            "market_question": market_question,
            "side": side,
            "size_usdc": size_usdc,
            "price": price,
            "shares": round(shares, 6),
            "status": "FILLED",
            "mode": "DEMO",
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"[DEMO] Order placed: {side} ${size_usdc:.2f} @ {price:.4f} "
            f"({shares:.2f} shares) on {market_question[:50]}"
        )
        return order

    def place_order_live(self, market_id: str, side: str, size_usdc: float,
                         price: float, private_key: str, order_type: str = "GTC",
                         market_question: str = "") -> Optional[dict]:
        url = f"{self.clob_base}/order"

        # Parse clob token IDs to get the correct token for the side
        payload = {
            "market": market_id,
            "side": side,
            "size": str(size_usdc),
            "price": str(price),
            "type": order_type,
        }

        headers = {
            "Authorization": f"Bearer {private_key}",
            "Content-Type": "application/json",
        }

        try:
            time.sleep(self.delay)
            resp = self.session.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            logger.info(
                f"[LIVE] Order placed: {side} ${size_usdc:.2f} @ {price:.4f} "
                f"type={order_type} on {market_question[:50]}"
            )
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"[LIVE] Order failed HTTP {resp.status_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"[LIVE] Order failed: {e}")
            return None

    def get_market_detail(self, market_id: str) -> Optional[dict]:
        url = f"{self.gamma_base}/markets/{market_id}"
        data = self._request_with_retry("GET", url)
        if not data:
            return None

        try:
            outcome_prices_raw = data.get("outcomePrices", "[]")
            if isinstance(outcome_prices_raw, str):
                prices = [float(p) for p in json.loads(outcome_prices_raw)]
            else:
                prices = [float(p) for p in outcome_prices_raw]

            return {
                "id": data.get("id", ""),
                "question": data.get("question", ""),
                "price_yes": prices[0] if prices else 0.5,
                "price_no": prices[1] if len(prices) > 1 else 0.5,
                "volume_24h": float(data.get("volume24hr", 0) or 0),
                "liquidity": float(data.get("liquidityClob", 0) or 0),
                "end_date": data.get("endDate", ""),
                "active": data.get("active", True),
            }
        except Exception as e:
            logger.error(f"Error parsing market detail {market_id}: {e}")
            return None
