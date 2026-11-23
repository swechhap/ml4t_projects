"""Single entry point for the ML4T Indicator Evaluation deliverables."""

import datetime as dt
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import TheoreticallyOptimalStrategy as tos
from indicators import plot_indicators
from marketsimcode import compute_portvals


def author():
    return "spandey343"


def _daily_stats(portvals):
    """Return the three report statistics using pandas' sample deviation."""
    daily_returns = portvals.pct_change().iloc[1:]
    return {
        "Cumulative Return": portvals.iloc[-1] / portvals.iloc[0] - 1.0,
        "Standard Deviation of Daily Returns": daily_returns.std(),
        "Mean of Daily Returns": daily_returns.mean(),
    }


def _benchmark_trades(index, symbol):
    """Buy 1,000 shares on the first day and then hold them."""
    benchmark = pd.DataFrame(0, index=index, columns=[symbol], dtype=int)
    benchmark.iloc[0, 0] = 1000
    return benchmark


def _write_results(results, filename):
    """Write the required six-decimal comparison table."""
    table = pd.DataFrame(results).T[
        [
            "Cumulative Return",
            "Standard Deviation of Daily Returns",
            "Mean of Daily Returns",
        ]
    ]
    filename.write_text(table.to_string(float_format=lambda value: f"{value:.6f}") + "\n")


def _plot_tos(benchmark_values, tos_values, filename):
    """Save the normalized purple-benchmark/red-TOS chart."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        benchmark_values.index,
        benchmark_values / benchmark_values.iloc[0],
        color="purple",
        label="Benchmark",
    )
    ax.plot(
        tos_values.index,
        tos_values / tos_values.iloc[0],
        color="red",
        label="Theoretically Optimal Strategy",
    )
    ax.set_title("JPM: Benchmark vs. Theoretically Optimal Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    """Run all requested analysis, plots, and text output in one command."""
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    start_value = 100000
    here = Path(__file__).resolve().parent
    images = here / "images"
    images.mkdir(exist_ok=True)

    tos_trades = tos.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_value)
    benchmark_trades = _benchmark_trades(tos_trades.index, symbol)
    benchmark_values = compute_portvals(benchmark_trades, start_val=start_value)
    tos_values = compute_portvals(tos_trades, start_val=start_value)


    _plot_tos(benchmark_values, tos_values, images / "tos_vs_benchmark.png")
    results = {
        "Benchmark": _daily_stats(benchmark_values),
        "Theoretically Optimal Strategy": _daily_stats(tos_values),
    }
    _write_results(results, here / "p6_results.txt")
    indicator_paths = plot_indicators(symbol=symbol, sd=sd, ed=ed, output_dir=images)

    print((here / "p6_results.txt").read_text(), end="")
    print("Saved TOS chart and {} indicator charts in {}".format(len(indicator_paths), images))
    return results


if __name__ == "__main__":
    main()
