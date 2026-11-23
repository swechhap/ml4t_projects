# Indicator Evaluation report draft

**Symbol and period:** JPM, January 1, 2008 through December 31, 2009. Prices
are adjusted closing prices except CCI, which uses unadjusted daily high, low,
and close. Each indicator is computed with approximately 90 calendar days of
prehistory so its lookback window is available at the beginning of the plotted
period.

## Technical indicators

### 1. Bollinger %B (20 days, 2 standard deviations)

Let `SMA20` be the 20-day mean of price and `sigma20` the 20-day sample standard
deviation. The upper and lower bands are `SMA20 + 2*sigma20` and
`SMA20 - 2*sigma20`. The indicator is `%B = (price - lower) / (upper - lower)`.
I would treat `%B < 0` as a long candidate and `%B > 1` as a short candidate;
otherwise the signal is cash. The intuition is mean reversion: unusually far
moves from a recent trading range can be temporary. RSI below 30 or above 70
can confirm the corresponding long or short signal.

Insert `images/indicator_bollinger_percent_b.png` here.

### 2. Relative Strength Index (RSI, 14 days)

For daily price changes, compute the separate 14-day simple average of positive
changes `avg_gain` and of absolute negative changes `avg_loss`. Then
`RS = avg_gain / avg_loss` and `RSI = 100 - 100/(1 + RS)`. I would treat RSI
below 30 as long, above 70 as short, and the middle range as cash. The indicator
captures overbought and oversold behavior: a one-sided recent run may exhaust
buyers or sellers. Bollinger %B is its confirmation indicator.

Insert `images/indicator_rsi.png` here.

### 3. Momentum (20 days)

The indicator is `momentum[t] = price[t] / price[t-20] - 1`. A positive value
means price has risen over the lookback; a negative value means it has fallen.
On its own I would use a positive value for long and a negative value for short,
with cash when it is near zero. Momentum represents trend persistence: investors
can underreact to information, allowing a move to continue. A same-sign MACD
histogram is a useful trend confirmation.

Insert `images/indicator_momentum.png` here.

### 4. MACD histogram (12, 26, 9)

Compute exponential moving averages `EMA12` and `EMA26`. The MACD line is
`EMA12 - EMA26`; the 9-day exponential average of that line is the signal line.
This report uses the single-vector histogram `MACD - signal`. I would treat a
positive histogram as long, negative as short, and exactly zero as cash. The
histogram turns the relative strength of fast and slow trends into a directional
signal. Twenty-day momentum confirms its direction.

Insert `images/indicator_macd_histogram.png` here.

### 5. Commodity Channel Index (CCI, 20 days)

For each day, typical price is `(high + low + close) / 3`. Let `TP_SMA20` be its
20-day average and let `mean_deviation20` be the average absolute distance of
each 20-day typical price from the window mean. Then
`CCI = (typical_price - TP_SMA20) / (0.015 * mean_deviation20)`. I would treat
CCI below -100 as long, above 100 as short, and the middle range as cash. CCI
measures unusually large moves relative to a recent price range, so it is a
mean-reversion complement to the two trend indicators. %B or RSI can confirm
an extreme CCI reading.

Insert `images/indicator_cci.png` here.

## Theoretically Optimal Strategy

The TOS has perfect knowledge of the next trading day's price. If tomorrow's
price is higher, it targets +1,000 JPM shares; if lower, it targets -1,000
shares; and if unchanged, it targets zero shares. The daily trade is the change
in target holdings, so a direct short-to-long flip is a +2,000-share trade.
The final trading day retains its previous target because its next-day price is
unknown. The strategy assumes no commission or market impact and never holds
more than 1,000 shares long or short. It is therefore an unattainable upper
bound, not a usable strategy.

The benchmark buys 1,000 JPM shares on the first trading day and never trades
again. Both portfolios start at $100,000 and are normalized to 1.0 for the
comparison chart. Insert `images/tos_vs_benchmark.png` below.

```text
                                Cumulative Return  Standard Deviation of Daily Returns  Mean of Daily Returns
Benchmark                                0.012300                             0.017004               0.000168
Theoretically Optimal Strategy           5.786100                             0.004548               0.003817
```
