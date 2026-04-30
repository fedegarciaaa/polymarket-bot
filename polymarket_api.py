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

    def scan_markets(self, min_volume: float = 10000, min_liquidity: float = 5000,
                     max_results: int = 500) -> list[dict]:
        """
        Fetch up to max_results active markets via pagination.
        Default max_results=500 to reach low-volume weather markets
        that don't appear in the top 100 by volume.
        """
        url = f"{self.gamma_base}/markets"
        all_raw = []
        page_size = 100

        for offset in range(0, max_results, page_size):
            params = {
                "closed": "false",
                "limit": page_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            data = self._request_with_retry("GET", url, params=params)
            if not data:
                break
            all_raw.extend(data)
            if len(data) < page_size:
                break  # last page

        if not all_raw:
            logger.error("Failed to fetch markets from Gamma API")
            return []

        # Deduplicate by market ID (API pagination occasionally repeats markets)
        seen_ids: set = set()
        deduped_raw = []
        for m in all_raw:
            mid = m.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                deduped_raw.append(m)
        all_raw = deduped_raw

        markets = []
        for market in all_raw:
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
                # Skip markets whose end date is already in the past
                end_date_str = parsed.get("end_date", "")
                if end_date_str:
                    try:
                        end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        if end_dt <= datetime.now(timezone.utc):
                            logger.debug(
                                f"Skipping expired market {parsed['id']}: "
                                f"endDate={end_date_str} | {parsed['question'][:50]}"
                            )
                            continue
                    except (ValueError, TypeError):
                        pass  # unparseable date — let strategies handle it

                markets.append(parsed)

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing market {market.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Scanned {len(all_raw)} markets, {len(markets)} passed filters (vol>={min_volume}, liq>={min_liquidity})")
        return markets

    # Substrings/regex fragments that identify a weather market. Applied to
    # question + category. Matches scan_weather_markets().
    WEATHER_KEYWORDS = (
        "temperature", "rainfall", "rain", "snowfall", "snow",
        "precipitation", "hurricane", "tornado", "blizzard", "wind",
        "forecast", "degrees", "°c", "°f", "celsius", "fahrenheit",
    )
    # Exclude sports teams and other false positives. Only unambiguously-sports
    # terms — avoid weather-event words ("storm", "thunder", "heat", etc.) even
    # though they are sometimes team names, because they also appear in real
    # weather questions ("heat wave", "thunderstorm", "storm expected").
    WEATHER_EXCLUDE = (
        " nhl ", " nba ", " nfl ", " mlb ", "stanley cup", "super bowl",
        "world series", "nba finals", "nhl finals", "championship",
        "playoff", "season mvp", "head coach", "stanley",
        "carolina hurricanes", "miami heat", "okc thunder",
        "oklahoma city thunder", "tampa bay lightning", "seattle storm",
        "phoenix suns", "golden state warriors", "washington wizards",
        "portland blazers", "houston rockets",
    )

    def scan_weather_markets(
        self, min_volume: float = 200, min_liquidity: float = 100, max_results: int = 1000
    ) -> list[dict]:
        """Return only markets whose question/category indicates a weather market.
        Uses low volume/liquidity thresholds at the API layer; the strategy
        applies its own stricter filters after confidence scoring."""
        all_markets = self.scan_markets(
            min_volume=min_volume, min_liquidity=min_liquidity, max_results=max_results
        )
        out: list[dict] = []
        excluded = 0
        for m in all_markets:
            haystack = f"{m.get('question','')} {m.get('category','')} {m.get('slug','')}".lower()
            if not any(k in haystack for k in self.WEATHER_KEYWORDS):
                continue
            if any(bad in haystack for bad in self.WEATHER_EXCLUDE):
                excluded += 1
                continue
            out.append(m)
        logger.info(
            f"scan_weather_markets: {len(out)}/{len(all_markets)} matched "
            f"(excluded {excluded} sports/non-weather)"
        )
        return out

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

    # ------------------------------------------------------------------
    # Orderbook helper (used by reeval_engine for mark-to-market and SELL)
    # ------------------------------------------------------------------
    def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Return raw best bid/ask for a given CLOB token_id."""
        if not token_id:
            return None
        url = f"{self.clob_base}/book"
        data = self._request_with_retry("GET", url, params={"token_id": token_id})
        if not data:
            return None
        try:
            bids = data.get("bids", []) or []
            asks = data.get("asks", []) or []
            best_bid = float(bids[0]["price"]) if bids else None
            best_ask = float(asks[0]["price"]) if asks else None
            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": (best_bid + best_ask) / 2 if (best_bid and best_ask) else None,
                "spread": (best_ask - best_bid) if (best_bid and best_ask) else None,
                "bid_size": float(bids[0].get("size", 0)) if bids else 0.0,
                "ask_size": float(asks[0].get("size", 0)) if asks else 0.0,
            }
        except Exception as e:
            logger.error(f"Error parsing orderbook for token {token_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Order cancellation (LIVE) / simulated (DEMO)
    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str, private_key: Optional[str] = None) -> dict:
        """
        Cancel a pending CLOB order by order_id.

        DEMO (no private_key) → simulated success.
        LIVE → DELETE /order with Bearer auth.
        """
        if not private_key:
            logger.info(f"[DEMO] cancel_order simulated for {order_id}")
            return {"success": True, "order_id": order_id, "mode": "DEMO"}

        url = f"{self.clob_base}/order"
        headers = {
            "Authorization": f"Bearer {private_key}",
            "Content-Type": "application/json",
        }
        try:
            time.sleep(self.delay)
            resp = self.session.delete(url, json={"orderID": order_id}, headers=headers, timeout=30)
            resp.raise_for_status()
            return {"success": True, "order_id": order_id, "mode": "LIVE", "response": resp.json()}
        except Exception as e:
            logger.error(f"[LIVE] cancel_order {order_id} failed: {e}")
            return {"success": False, "order_id": order_id, "mode": "LIVE", "error": str(e)}

    # ------------------------------------------------------------------
    # Close position by selling shares (LIVE) / simulated (DEMO)
    # ------------------------------------------------------------------
    def sell_position(
        self,
        token_id: str,
        shares: float,
        min_price: float = 0.0,
        slippage_pct: float = 0.005,
        private_key: Optional[str] = None,
        market_id: str = "",
        market_question: str = "",
    ) -> dict:
        """
        Sell an open position by placing a SELL order at best bid (FOK).

        DEMO: simulates the fill using the current best bid minus `slippage_pct`.
        LIVE: places a SELL FOK order via CLOB /order.

        Returns {success, shares_sold, fill_price, gross_usdc, fees_usdc, error?}.
        """
        book = self.get_orderbook(token_id)
        if not book or book.get("best_bid") is None:
            return {"success": False, "error": "no_orderbook", "shares_sold": 0.0, "fill_price": 0.0}

        bid = float(book["best_bid"])
        if bid < min_price:
            return {
                "success": False,
                "error": f"bid_below_min ({bid:.4f}<{min_price:.4f})",
                "shares_sold": 0.0,
                "fill_price": bid,
            }

        fill_price = round(bid * (1.0 - slippage_pct), 6)
        gross = round(fill_price * shares, 6)
        fees = 0.0

        if not private_key:
            logger.info(
                f"[DEMO] sell_position {shares:.2f} shares @ {fill_price:.4f} "
                f"(bid={bid:.4f}, slip={slippage_pct*100:.2f}%) → ${gross:.2f}"
            )
            return {
                "success": True,
                "mode": "DEMO",
                "shares_sold": shares,
                "fill_price": fill_price,
                "gross_usdc": gross,
                "fees_usdc": fees,
                "best_bid": bid,
            }

        url = f"{self.clob_base}/order"
        payload = {
            "market": market_id,
            "side": "SELL",
            "size": str(shares),
            "price": str(fill_price),
            "type": "FOK",
            "tokenID": token_id,
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
                f"[LIVE] sell_position {shares:.2f} @ {fill_price:.4f} on {market_question[:50]}"
            )
            return {
                "success": True,
                "mode": "LIVE",
                "shares_sold": shares,
                "fill_price": fill_price,
                "gross_usdc": gross,
                "fees_usdc": fees,
                "response": result,
            }
        except Exception as e:
            logger.error(f"[LIVE] sell_position failed: {e}")
            return {
                "success": False,
                "mode": "LIVE",
                "shares_sold": 0.0,
                "fill_price": fill_price,
                "gross_usdc": 0.0,
                "fees_usdc": fees,
                "error": str(e),
            }

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
