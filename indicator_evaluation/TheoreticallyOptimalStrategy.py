import datetime as dt

import numpy as np
import pandas as pd

from util import get_data


def author():
    return "spandey343"


def study_group():
    return "spandey343"


def testPolicy(
    symbol="JPM",
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    sv=100000,
):

    del sv
    prices = get_data([symbol], pd.date_range(sd, ed))[symbol].dropna()
    if prices.empty:
        raise ValueError("No price data was found for the requested dates.")


    next_day_change = prices.shift(-1) - prices
    target = pd.Series(
        np.select(
            [next_day_change.gt(0), next_day_change.lt(0)], [1000, -1000], default=0
        ),
        index=prices.index,
        dtype=int,
    )
    if len(target) > 1:
        target.iloc[-1] = target.iloc[-2]

    trades = target.diff()
    trades.iloc[0] = target.iloc[0]
    return trades.astype(int).to_frame(name=symbol)
