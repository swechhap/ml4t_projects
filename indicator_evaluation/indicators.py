import datetime as dt
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util import get_data


def author():
    return "spandey343"

def _series(prices):
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    if isinstance(prices, pd.DataFrame) and prices.shape[1] == 1:
        return prices.iloc[:, 0].astype(float)
    raise ValueError("Pass a Series or a one-column DataFrame of prices.")


def bollinger_percent_b(prices, window=20, num_std=2.0):
    price = _series(prices)
    middle = price.rolling(window=window, min_periods=window).mean()
    deviation = price.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    return ((price - lower) / (upper - lower)).rename("Bollinger %B")


def relative_strength_index(prices, window=14):
    """Return the 0--100 RSI using simple rolling average gains and losses."""
    price = _series(prices)
    delta = price.diff()
    avg_gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    return rsi.rename("RSI")


def momentum(prices, window=20):
    """Return N-day rate of change, price[t] / price[t-N] - 1."""
    price = _series(prices)
    return (price / price.shift(window) - 1.0).rename("Momentum")


def macd_histogram(prices, fast=12, slow=26, signal=9):
    """Return MACD line minus its signal line as one one-dimensional vector."""
    price = _series(prices)
    fast_ema = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()
    return (macd_line - signal_line).rename("MACD histogram")


def commodity_channel_index(high, low, close, window=20):
    """Return the Commodity Channel Index (CCI) from high, low, and close."""
    high, low, close = _series(high), _series(low), _series(close)
    typical_price = (high + low + close) / 3.0
    sma = typical_price.rolling(window=window, min_periods=window).mean()
    mean_deviation = typical_price.rolling(window=window, min_periods=window).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    return ((typical_price - sma) / (0.015 * mean_deviation)).rename("CCI")


def _indicator_data(symbol, sd, ed, warmup_days=90):
    """Fetch adjusted prices plus high/low, including enough prehistory."""
    dates = pd.date_range(sd - dt.timedelta(days=warmup_days), ed)
    close = get_data([symbol], dates)[symbol]
    high = get_data([symbol], dates, colname="High")[symbol]
    low = get_data([symbol], dates, colname="Low")[symbol]
    return pd.concat({"close": close, "high": high, "low": low}, axis=1).dropna()


def _plot(price, indicator, filename, title, ylabel, levels=(), helpers=None):
    """Save a two-panel price/indicator chart and always close its figure."""
    fig, (price_ax, indicator_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    normalized = price / price.iloc[0]
    price_ax.plot(normalized.index, normalized, color="black", label="Normalized price")
    for label, line in (helpers or {}).items():
        price_ax.plot(line.index, line / price.iloc[0], label=label)
    price_ax.set_title(title)
    price_ax.set_ylabel("Normalized price")
    price_ax.legend(loc="best")
    price_ax.grid(alpha=0.25)

    indicator_ax.plot(indicator.index, indicator, color="tab:blue", label=indicator.name)
    for level, label in levels:
        indicator_ax.axhline(level, linestyle="--", linewidth=1, label=label)
    indicator_ax.set_xlabel("Date")
    indicator_ax.set_ylabel(ylabel)
    indicator_ax.legend(loc="best")
    indicator_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_indicators(
    symbol="JPM",
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    output_dir="images",
):
    """Generate exactly five indicator charts and return their output paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = _indicator_data(symbol, sd, ed)
    close, high, low = data["close"], data["high"], data["low"]
    visible = data.index >= pd.Timestamp(sd)
    price = close.loc[visible]

    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std()
    bb = bollinger_percent_b(close).loc[visible]
    rsi = relative_strength_index(close).loc[visible]
    mom = momentum(close).loc[visible]
    macd = macd_histogram(close).loc[visible]
    cci = commodity_channel_index(high, low, close).loc[visible]

    paths = []
    filename = output_path / "indicator_bollinger_percent_b.png"
    _plot(
        price,
        bb,
        filename,
        f"{symbol}: Bollinger %B (20-day, 2 standard deviations)",
        "%B",
        [(0, "Long threshold"), (1, "Short threshold")],
        {"20-day SMA": middle.loc[visible], "Upper band": (middle + 2 * deviation).loc[visible], "Lower band": (middle - 2 * deviation).loc[visible]},
    )
    paths.append(filename)

    filename = output_path / "indicator_rsi.png"
    _plot(price, rsi, filename, f"{symbol}: RSI (14-day)", "RSI", [(30, "Long threshold"), (70, "Short threshold")])
    paths.append(filename)

    filename = output_path / "indicator_momentum.png"
    _plot(price, mom, filename, f"{symbol}: Momentum (20-day)", "Rate of change", [(0, "Neutral")])
    paths.append(filename)

    filename = output_path / "indicator_macd_histogram.png"
    _plot(price, macd, filename, f"{symbol}: MACD histogram (12, 26, 9)", "MACD histogram", [(0, "Neutral")])
    paths.append(filename)

    filename = output_path / "indicator_cci.png"
    _plot(price, cci, filename, f"{symbol}: CCI (20-day)", "CCI", [(-100, "Long threshold"), (100, "Short threshold")])
    paths.append(filename)
    return paths
