# Indicator Evaluation deliverable

This folder is a clean implementation of the requirements from the supplied
strategy guide. It deliberately does **not** modify the earlier course files
in `ML4T_2026Fall`.

## Run it against the downloaded ML4T data

From this folder, run:

```bash
MARKET_DATA_DIR=/Users/swechhap/Downloads/ML4T_2026Fall/data \
PYTHONPATH=/Users/swechhap/Downloads/ML4T_2026Fall:. \
python testproject.py
```

The course starter layout normally places `util.py` and `data/` one directory
above the project. In that layout, copy these four Python files into the
assigned project directory and use the course command:

```bash
PYTHONPATH=../:. python testproject.py
```

## Deliverables produced

- `p6_results.txt`: Benchmark/TOS statistics to six decimal places.
- `images/tos_vs_benchmark.png`: normalized purple benchmark and red TOS.
- Five images named `indicator_*.png`: %B, RSI, Momentum, MACD histogram,
  and CCI.
- `report.md`: concise report content to move into the instructor-provided
  report format once that template is available.

The five choices satisfy the guide's constraints: all five are from the
approved list, only Bollinger Bands and RSI are from the restricted
{SMA, BB, RSI} set, and Momentum is used without ROC.
