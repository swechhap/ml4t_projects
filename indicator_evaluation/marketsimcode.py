import pandas as pd
import numpy as np

from util import get_data


def author():
    return "spandey343"


def _validate_trades(df_trades):
    if isinstance(df_trades, pd.Series):
        if not df_trades.name:
            raise ValueError("A trade Series must have a ticker-symbol name.")
        df_trades = df_trades.to_frame()

    if not isinstance(df_trades, pd.DataFrame) or df_trades.empty:
        raise ValueError("df_trades must be a non-empty pandas DataFrame.")

    if not isinstance(df_trades.index, pd.DatetimeIndex):
        raise TypeError("df_trades must use a DatetimeIndex.")

    if df_trades.index.has_duplicates:
        raise ValueError("df_trades may contain only one row per date.")

    if len(df_trades.columns) == 0:
        raise ValueError("At least one ticker column is required.")

    if "CASH" in df_trades.columns:
        raise ValueError("Do not include a CASH column in df_trades.")

    trades = df_trades.sort_index().copy().astype(float)

    if not np.isfinite(trades.to_numpy()).all():
        raise ValueError("Trades must contain only finite numeric values.")

    return trades


def compute_portvals(df_trades, start_val=100000,
                     commission=0.0, impact=0.0):
    """Compute daily portfolio values for signed share trades."""
    trades_input = _validate_trades(df_trades)
    symbols = list(trades_input.columns)

    prices = get_data(symbols, trades_input.index)[symbols]

    # Do not silently substitute future prices for missing prices.
    if prices.isna().any().any():
        raise ValueError("Missing prices found for one or more trade dates.")

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Prices must contain only finite values.")

    if not np.isfinite([start_val, commission, impact]).all():
        raise ValueError("Portfolio value and costs must be finite.")

    if start_val < 0 or commission < 0 or impact < 0:
        raise ValueError("Start value and costs cannot be negative.")

    dollar_trades = trades_input * prices

    trade_costs = (
        commission * trades_input.ne(0.0).sum(axis=1)
        + impact * dollar_trades.abs().sum(axis=1)
    )

    trades = trades_input.copy()
    trades["CASH"] = -dollar_trades.sum(axis=1) - trade_costs
    trades.iloc[0, trades.columns.get_loc("CASH")] += start_val

    holdings = trades.cumsum()

    values = prices.copy()
    values["CASH"] = 1.0

    return (holdings * values).sum(axis=1).rename("Portfolio Value")
