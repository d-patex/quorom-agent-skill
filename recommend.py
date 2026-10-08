"""Offline recommendation demo; no model, network, or API key required."""

import argparse
import json

from tools import recommend_activities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--start-time", required=True, help="24-hour HH:MM")
    parser.add_argument("--max-cost", type=float, help="Total group cost limit")
    parser.add_argument("--category")
    args = parser.parse_args()
    result = recommend_activities(args.day, args.start_time, args.max_cost, args.category)
    print("Offline tool demonstration (not a live LLM conversation)")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
