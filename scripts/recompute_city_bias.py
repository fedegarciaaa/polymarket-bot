"""Recompute the city-bias table from closed-trade history.

Reads `trades` joined with `market_resolutions` to compute per-(city, metric, month)
average error (ensemble_mean - actual_value). Cells with ≥ MIN_SAMPLES_FOR_OVERWRITE
samples replace the hand-tuned seeds.

Usage:
    python scripts/recompute_city_bias.py
    python scripts/recompute_city_bias.py --dry-run

Output: data/city_bias.json (overwritten on disk; the running bot picks it up at
next get_default_table() call after reload_default_table()).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this file directly (python scripts/recompute_city_bias.py).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import Database
from strategies.city_bias import recompute_from_history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print but don't write to disk.")
    parser.add_argument("--db", default="data/bot.db", help="SQLite path")
    args = parser.parse_args()

    db = Database(db_path=args.db)
    entries = recompute_from_history(db, write=not args.dry_run)

    print(json.dumps(entries, indent=2))
    print(f"\n{len(entries)} entries {'computed' if args.dry_run else 'written to data/city_bias.json'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
