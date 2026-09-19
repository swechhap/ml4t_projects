""""""  		  	   		 		  		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
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
  		  	   		 		  		  		  		    	 		 		   		 		  
Student Name: Swechha Pandey (replace with your name)  		  	   		 		  		  		  		    	 		 		   		 		  
GT User ID: spandey343 (replace with your User ID)  		  	   		 		  		  		  		    	 		 		   		 		  
GT ID: 904262734 (replace with your GT ID) 		  	   		 		  		  		  		    	 		 		   		 		  
"""  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 		  		  		  		    	 		 		   		 		  
from scipy.optimize import minimize
import matplotlib.pyplot as plt  		  	   		 		  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 		  		  		  		    	 		 		   		 		  
from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "spandey343"  # replace tb34 with your Georgia Tech username.


def study_group():
    return "spandey343"

# This is the function that will be tested by the autograder  		  	   		 		  		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		 		  		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		 		  		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		 		  		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		 		  		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 		  		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 		  		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 		  		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 		  		  		  		    	 		 		   		 		  
    statistics.  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 		  		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 		  		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 		  		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 		  		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 		  		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 		  		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 		  		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 		  		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 		  		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 		  		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 		  		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 		  		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		 		  		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		 		  		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  

    allocs = np.ones(len(syms))/len(syms)
    normed = prices/prices.iloc[0]
    bounds = [(0.0, 1.0)] * len(syms)

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
    )

    result = minimize(
        negative_sharpe,
        allocs,
        args=(normed,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    allocs = result.x


    # Get daily portfolio value  		  	   		 		  		  		  		    	 		 		   		 		  
    port_val = (normed * allocs).sum(axis=1)


    daily_returns = port_val.pct_change().dropna()

    cr = port_val.iloc[-1] / port_val.iloc[0] - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(252)*adr/sddr

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 		  		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here
        normed_SPY = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat(  		  	   		 		  		  		  		    	 		 		   		 		  
            [port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        ax = df_temp.plot(
            title="Daily Portfolio Value and SPY",
            fontsize=12,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")

        plt.savefig("Figure1.png")
        plt.close()

    return allocs, cr, adr, sddr, sr  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
def test_code():  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 		  		  		  		    	 		 		   		 		  
    """  		  	   		 		  		  		  		    	 		 		   		 		  
  		  	   		 		  		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

  		  	   		 		  		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		 		  		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 		  		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )



def negative_sharpe(allocs,normed):
    port_val = (normed * allocs).sum(axis=1)
    daily_returns = port_val.pct_change().dropna()
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(252) * adr / sddr

    return -sr
  		  	   		 		  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 		  		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		 		  		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		 		  		  		  		    	 		 		   		 		  
    test_code()  		  	   		 		  		  		  		    	 		 		   		 		  
