import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import risk_models
from pypfopt import plotting
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from pypfopt import objective_functions
import pandas as pd
import numpy as np

tickers = ["btc-usd", "eth-usd", "ltc-usd"]

ohlc = yf.download(tickers, period="max")
print(ohlc)
prices = ohlc["Adj Close"].dropna(how="all")
print(prices)
prices.tail()
#prices[prices.index >= "2008-01-01"].plot(figsize=(15, 10));

#sample_cov = risk_models.sample_cov(prices, frequency=365)
sample_wolf = risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf()
print(sample_wolf)
#plotting.plot_covariance(sample_cov, plot_correlation=True);
plotting.plot_covariance(sample_wolf, plot_correlation=True);

mu = expected_returns.capm_return(prices)
plt.figure()
mu.plot.barh(figsize=(10, 6));

# You don't have to provide expected returns in this case
ef = EfficientFrontier(mu, sample_wolf)
#ef.add_objective(objective_functions.L2_reg, gamma=1)
#ef.min_volatility()
#ef.efficient_return(target_return=0.8)

'''
btc_index = ef.tickers.index("btc-usd")
ef.add_constraint(lambda w: w[btc_index] == 0.30)
eth_index = ef.tickers.index("eth-usd")
ef.add_constraint(lambda w: w[eth_index] == 0.30)
'''
raw_weights = ef.max_sharpe()
#ef.efficient_risk(0.25)
cleaned_weights = ef.clean_weights()


plt.figure()
pd.Series(cleaned_weights).plot.barh();
ef.portfolio_performance(verbose=True);
print(cleaned_weights)
plt.show()