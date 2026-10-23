""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 		  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 		  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 		  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 		  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  		  		  		    	 		 		   		 		  
or edited.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 		  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 		  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
Student Name: Swechha Pandey  	   		 		  		  		  		    	 		 		   		 		  
GT User ID: spandey343 	  	   		 		  		  		  		    	 		 		   		 		  
GT ID: 904262734 		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 		  		  		  		    	 		 		   		 		  
import os  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 		  		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
def author():
    return "spandey343"

def study_group():
    return "spandey343"

def compute_stats(series):
    daily_rets = (series / series.shift(1) - 1).iloc[1:]
    cum_ret = series.iloc[-1] / series.iloc[0] - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()          # sample std (ddof=1)
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def compute_portvals(  		  	   		 		  		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		 		  		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		 		  		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		 		  		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		 		  		  		  		    	 		 		   		 		  
):  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		 		  		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 		  		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 		  		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 		  		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 		  		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 		  		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 		  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 		  		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 		  		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 		  		  		  		    	 		 		   		 		  
    """
    # 1. Read orders and sort by date
    orders = pd.read_csv(
        orders_file,
        index_col="Date",
        parse_dates=True,
        na_values=["nan"]
    )
    orders = orders.sort_index()

    start_date = orders.index.min()
    end_date = orders.index.max()
    symbols = list(orders["Symbol"].unique())
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices.drop(
        columns=["SPY"],
        errors="ignore"
    ) if "SPY" not in symbols else prices
    prices = prices.ffill().bfill()
    prices["CASH"] = 1.0
    trades = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=prices.columns
    )

    for date, row in orders.iterrows():
        if date not in prices.index:
            later_days = prices.index[prices.index > date]
            if len(later_days) == 0:
                continue
            date = later_days[0]

        sym = row["Symbol"]
        order = row["Order"]
        shares = row["Shares"]

        price = prices.loc[date, sym]

        if order == "BUY":
            trades.loc[date, sym] += shares
            trades.loc[date, "CASH"] -= price * shares

        elif order == "SELL":
            trades.loc[date, sym] -= shares
            trades.loc[date, "CASH"] += price * shares

        trades.loc[date, "CASH"] -= commission
        trades.loc[date, "CASH"] -= impact * price * shares

    holdings = trades.copy()
    holdings.iloc[0, holdings.columns.get_loc("CASH")] += start_val
    holdings = holdings.cumsum()
    values = prices * holdings
    portvals = values.sum(axis=1).to_frame(name="Portfolio Value")
    return portvals
  		  	   		 		  		  		  		    	 		 		   		 		  
def test_code():  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		 		  		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		 		  		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    of = "./orders/orders2.csv"  		  	   		 		  		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    # Process orders  		  	   		 		  		  		  		    	 		 		   		 		  

    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        print("warning, code did not return a DataFrame")

    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    spy = get_data(["SPY"], pd.date_range(start_date, end_date))["SPY"]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_stats(spy)
  		  	   		 		  		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print()  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print()  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print()  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print()  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 		  		  		  		    	 		 		   		 		  
    print()  		  	   		 		  		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals.iloc[-1]}")
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 		  		  		  		    	 		 		   		 		  
    test_code()  		  	   		 		  		  		  		    	 		 		   		 		  
