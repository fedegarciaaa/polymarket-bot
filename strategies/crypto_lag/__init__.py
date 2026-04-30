"""Crypto-Lag MAKER bot — Polymarket BTC/ETH/SOL 15-min markets.

Strategy: post maker orders against Polymarket's CLOB using a Black-Scholes
digital probability derived from real-time Binance prices. Edge comes from a
better probability model than the late-lagging Polymarket mid, not from speed.
"""
