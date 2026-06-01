"""
IBKR paper / live trading integration via ib_insync.

Prerequisites
-------------
1. Install:  pip install ib_insync
2. Open TWS or IB Gateway and log into your PAPER account.
3. Enable API:  TWS → Edit → Global Configuration → API → Settings
                ✓ Enable ActiveX and Socket Clients
                Socket port: 7497  (paper TWS)  or  4002  (paper IB Gateway)
                ✓ Allow connections from localhost only

Connection ports (paper trading)
---------------------------------
    TWS paper:         7497   ← default here
    IB Gateway paper:  4002
"""

import pandas as pd
from ib_insync import IB, Stock, MarketOrder


class IBKRBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        print(f"Connected to IBKR  {self.host}:{self.port}  (clientId={self.client_id})")

    def disconnect(self) -> None:
        self.ib.disconnect()
        print("Disconnected from IBKR.")

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_value(self) -> float:
        """Net liquidation value in USD."""
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        raise RuntimeError("Could not read NetLiquidation from IBKR.")

    def get_positions(self) -> dict[str, float]:
        """Returns {ticker: shares} for all open positions."""
        return {pos.contract.symbol: pos.position for pos in self.ib.positions()}

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch last traded price for each ticker via snapshot."""
        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        self.ib.qualifyContracts(*contracts)
        snapshots = self.ib.reqTickers(*contracts)
        prices = {}
        for snap in snapshots:
            sym = snap.contract.symbol
            # prefer last; fall back to close if market is closed
            price = snap.last if (snap.last and snap.last > 0) else snap.close
            if price and price > 0:
                prices[sym] = float(price)
        return prices

    # ------------------------------------------------------------------
    # Order sizing
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        target_weights: dict[str, float],
        account_value: float,
        current_positions: dict[str, float],
        current_prices: dict[str, float],
        min_trade_pct: float = 0.005,
    ) -> list[dict]:
        """
        Diff current portfolio vs target weights and return a list of orders.

        Args:
            target_weights  : {ticker: weight}  weights must sum to ≤ 1.
            account_value   : net liquidation value in USD.
            current_positions: {ticker: shares} from IBKR.
            current_prices  : {ticker: price}   latest prices.
            min_trade_pct   : skip rebalance if weight drift < this threshold.

        Returns list of dicts: {ticker, action, shares, target_weight,
                                current_weight, est_value}
        """
        current_weights: dict[str, float] = {}
        for ticker, shares in current_positions.items():
            price = current_prices.get(ticker, 0.0)
            current_weights[ticker] = (shares * price) / account_value

        orders = []
        for ticker in set(target_weights) | set(current_weights):
            target_w = target_weights.get(ticker, 0.0)
            current_w = current_weights.get(ticker, 0.0)
            delta_w = target_w - current_w

            if abs(delta_w) < min_trade_pct:
                continue

            price = current_prices.get(ticker)
            if not price:
                print(f"  WARNING: no price for {ticker}, skipping.")
                continue

            delta_dollars = delta_w * account_value
            shares = int(delta_dollars / price)   # whole shares only
            if shares == 0:
                continue

            orders.append({
                "ticker": ticker,
                "action": "BUY" if shares > 0 else "SELL",
                "shares": abs(shares),
                "target_weight": target_w,
                "current_weight": current_w,
                "est_value": abs(delta_dollars),
            })

        # Sells first so buying power is freed before buys settle
        orders.sort(key=lambda o: (0 if o["action"] == "SELL" else 1))
        return orders

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def place_market_orders(self, orders: list[dict], dry_run: bool = True) -> list:
        """
        Place market orders.

        Args:
            orders  : from compute_orders()
            dry_run : if True, print the order book but don't execute.
                      Default is True — you must explicitly pass False to trade.

        Returns list of ib_insync Trade objects (empty on dry run).
        """
        if not orders:
            print("No orders to place — portfolio is already at target.")
            return []

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Order book:")
        header = f"  {'Ticker':<8} {'Action':<6} {'Shares':>7} {'Target%':>9} {'Current%':>10} {'Est. $':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for o in orders:
            print(
                f"  {o['ticker']:<8} {o['action']:<6} {o['shares']:>7} "
                f"{o['target_weight']:>8.1%} {o['current_weight']:>9.1%} "
                f"${o['est_value']:>10,.0f}"
            )

        if dry_run:
            print("\n[DRY RUN] Set dry_run=False (or use --execute flag) to place orders.")
            return []

        trades = []
        for o in orders:
            contract = Stock(o["ticker"], "SMART", "USD")
            self.ib.qualifyContracts(contract)
            order = MarketOrder(o["action"], o["shares"])
            trade = self.ib.placeOrder(contract, order)
            trades.append(trade)
            print(f"  Placed {o['action']} {o['shares']} {o['ticker']}")

        self.ib.sleep(2)  # let orders register before disconnecting
        return trades

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> pd.DataFrame:
        """Return a DataFrame of current positions with market value."""
        rows = []
        for pos in self.ib.positions():
            rows.append({
                "ticker": pos.contract.symbol,
                "shares": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "shares", "avg_cost", "market_value"])
