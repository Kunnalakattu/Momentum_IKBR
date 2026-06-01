"""
Daily runner for the momentum strategy — paper & live trading via IBKR.

─────────────────────────────────────────────────────────────────────
QUICK START (paper trading)
─────────────────────────────────────────────────────────────────────
1. Open TWS, log into your PAPER account.
2. Enable API:  Edit → Global Configuration → API → Settings
                Socket port 7497, allow localhost.
3. Install deps:  pip install ib_insync
4. Run modes:

   # Just see today's signal — no broker needed
   python run_live.py --signal-only

   # Preview orders (connects to IBKR but doesn't trade)
   python run_live.py --dry-run

   # Actually place orders on paper account
   python run_live.py --execute

─────────────────────────────────────────────────────────────────────
WHEN TO RUN
─────────────────────────────────────────────────────────────────────
Run once per month (every ~21 trading days) — the strategy rebalances
monthly. A good time is 09:31 ET (just after market open) so market
orders fill quickly.

You can also run --signal-only any day to see the current standings.
─────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.live_signal import get_live_signal, print_signal
from src.broker_ibkr import IBKRBroker

# ── IBKR connection settings ──────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497   # TWS paper: 7497 | IB Gateway paper: 4002
CLIENT_ID = 1      # must be unique if multiple scripts connect


def run(dry_run: bool = True, signal_only: bool = False, port: int = IBKR_PORT) -> None:

    # ── 1. Compute today's signal ────────────────────────────────
    print("Downloading data and computing signal...")
    signal_df, as_of = get_live_signal()
    print_signal(signal_df, as_of)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    signal_df.to_csv(os.path.join(out_dir, "live_signal.csv"), index=False)
    print(f"\nSignal saved → output/live_signal.csv")

    if signal_only:
        return

    # ── 2. Connect to IBKR ───────────────────────────────────────
    broker = IBKRBroker(host=IBKR_HOST, port=port, client_id=CLIENT_ID)
    broker.connect()

    try:
        account_value = broker.get_account_value()
        print(f"\nAccount net liquidation : ${account_value:>12,.2f}")

        current_positions = broker.get_positions()
        if current_positions:
            print("\nCurrent positions:")
            for ticker, shares in current_positions.items():
                print(f"  {ticker:<8} {shares:>8.0f} shares")
        else:
            print("Current positions: none (fresh account)")

        # ── 3. Build target weights ──────────────────────────────
        holdings = signal_df[signal_df["hold"]]["ticker"].tolist()
        weight = 1.0 / len(holdings) if holdings else 0.0
        target_weights = {t: weight for t in holdings}

        # ── 4. Fetch live prices ─────────────────────────────────
        all_tickers = list(set(holdings) | set(current_positions.keys()))
        print(f"\nFetching live prices for {len(all_tickers)} tickers...")
        prices = broker.get_current_prices(all_tickers)

        missing = [t for t in all_tickers if t not in prices]
        if missing:
            print(f"  WARNING: no price data for: {missing}")

        # ── 5. Compute orders ────────────────────────────────────
        orders = broker.compute_orders(
            target_weights=target_weights,
            account_value=account_value,
            current_positions=current_positions,
            current_prices=prices,
        )

        # ── 6. Place / preview orders ────────────────────────────
        broker.place_market_orders(orders, dry_run=dry_run)

        # ── 7. Show updated portfolio ────────────────────────────
        if not dry_run:
            print("\nUpdated portfolio:")
            print(broker.portfolio_summary().to_string(index=False))

    finally:
        broker.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Momentum strategy live/paper trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--signal-only",
        action="store_true",
        help="Print today's signal without connecting to IBKR.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Connect to IBKR, compute orders, but do NOT execute. (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually place market orders on your IBKR account.",
    )

    parser.add_argument("--port", type=int, default=IBKR_PORT, help="IBKR API port.")
    args = parser.parse_args()

    if args.signal_only:
        run(signal_only=True, port=args.port)
    elif args.execute:
        print("⚠  LIVE ORDER MODE — orders will be placed on your IBKR account.")
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return
        run(dry_run=False, port=args.port)
    else:
        run(dry_run=True, port=args.port)


if __name__ == "__main__":
    main()
