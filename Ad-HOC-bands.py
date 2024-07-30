# region imports
from AlgorithmImports import *
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import statsmodels.tsa.api as smt
from sklearn.linear_model import LinearRegression
# endregion

class WellDressedYellowGreenGoat(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        self.AddEquity('SPY', Resolution.Daily)
        self.pairs = []
        self.lookback = 30
        self.threshold = 2
        self.tradedPairs = set()
        self.cointegration_threshold = 0.05  # p-value threshold for cointegration
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY", 10), self.Rebalance)

    def CoarseSelectionFunction(self, coarse):
        # Sort by dollar volume and take the top 50
        sorted_by_dollar_volume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        top_stocks = [x.Symbol for x in sorted_by_dollar_volume[:50]]
        return top_stocks

    def OnSecuritiesChanged(self, changes):
        # Reset the pairs for the new universe
        for security in changes.AddedSecurities:
            security.SetLeverage(20)
        self.high_coint_pairs, self.low_coint_pairs = self.FindCointegratedPairs(changes.AddedSecurities)
        for security in changes.RemovedSecurities:
            if security.Symbol in [pair[0] for pair in self.high_coint_pairs] + [pair[1] for pair in self.high_coint_pairs]:
                self.Liquidate(security.Symbol)
        

    def FindCointegratedPairs(self, securities):
        symbols = [s.Symbol for s in securities]
        prices = self.History(symbols, self.lookback, Resolution.Daily)
        # Filter out the symbols that don't have the 'close' column
        # Check if the 'close' column exists in the prices DataFrame
        if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
            prices = prices['close'].unstack(level=0).dropna(axis=1)
        elif isinstance(prices, pd.Series):
            # If prices is a Series, check if the 'close' column exists in the Series
            if 'close' in prices.index:
                prices = prices['close'].unstack(level=0).dropna(axis=1)
            else:
                # Filter out the symbols that don't have the 'close' column
                valid_symbols = [sym for sym in symbols if 'close' in prices[sym].index]
                prices = self.History(valid_symbols, self.lookback, Resolution.Daily)['close'].unstack(level=0).dropna(axis=1)
        n = prices.shape[1]
        high_coint_pairs=[]
        low_coint_pairs=[]
        for i in range(n):
            for j in range(i+1, n):
                stock1 = prices.iloc[:, i]
                stock2 = prices.iloc[:, j]
                score, pvalue, _ = coint(stock1, stock2)
                if pvalue < self.cointegration_threshold:
                    high_coint_pairs.append(((prices.columns[i], prices.columns[j]), pvalue))
                elif pvalue > self.cointegration_threshold:
                    low_coint_pairs.append(((prices.columns[i], prices.columns[j]), pvalue))

        # Sort pairs by p-value and select the top 10
        high_coint_pairs  = sorted(high_coint_pairs, key=lambda x: x[1])[:15]
        low_coint_pairs = sorted(low_coint_pairs, key=lambda x: x[1])[:15]
        return [pair[0] for pair in high_coint_pairs], [pair[0] for pair in low_coint_pairs]
    
        
    def Rebalance(self):
        if self.Time.date() == datetime(2024, 1, 1).date():
            self.LiquidateIfUnrealizedGains()
        # Check for unrealized profit across the portfolio
        totalUnrealizedProfit = self.Portfolio.TotalUnrealizedProfit
        portfolioValue = self.Portfolio.TotalPortfolioValue
        profitPercentage = (totalUnrealizedProfit / portfolioValue) * 100

        if profitPercentage >= 25 :
            self.Liquidate()
            self.log(f"Liquidated all positions due to 25% or greater unrealized profit at {self.Time}")
            return

        # VAR Model
        high_var_model = self.BuildVARModel(self.high_coint_pairs)
        low_var_model = self.BuildVARModel(self.low_coint_pairs)
        if high_var_model is not None:
            # Identify Dynamics of the Co-integration Factor
            high_coint_factor_volatility = self.EstimateCointegrationFactorVolatility(self.high_coint_pairs, high_var_model)
            self.CheckAutoCorrelation(high_var_model.endog)

            # Trading Strategy using Ad-hoc Bands
            self.TradingStrategyWithAdHocBands(self.high_coint_pairs,high_var_model, high_var_model.endog, -1, 1)
        '''if low_var_model is not None:
            # Identify Dynamics of the Co-integration Factor
            low_coint_factor_volatility = self.EstimateCointegrationFactorVolatility(self.low_coint_pairs, low_var_model)
            self.CheckAutoCorrelation(low_var_model.endog)

            # Trading Strategy using Ad-hoc Bands
            self.TradingStrategyWithAdHocBands(self.low_coint_pairs,low_var_model, low_var_model.endog, -1, 1)'''

    def BuildVARModel(self, pairs):
        # Retrieve the mid-prices for the selected pairs
        prices = self.History([pair[0] for pair in pairs] + [pair[1] for pair in pairs], self.lookback, Resolution.Daily)
        self.Log(f"Prices data structure: {type(prices)}")
        self.Log(f"Prices columns: {prices.columns}")
        
        # Filter and unstack prices for valid pairs
        if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
            prices = prices['close'].unstack(level=0).dropna()
        else:
            self.Log("'close' column not found in prices DataFrame.")
            return None
        
        # Fit the VAR model with lag 1
        model = smt.VAR(prices)
        results = model.fit(1)
        return results
        

    def EstimateCointegrationFactorVolatility(self, pairs, var_model):
        # Retrieve the mid-prices for the selected pairs
        prices = self.History([pair[0] for pair in pairs] + [pair[1] for pair in pairs], self.lookback, Resolution.Daily).close.unstack(level=0).dropna()
        
        # Compute the co-integration factor
        coint_factor = np.dot(prices, var_model.coefs[0])
        #coint_factor = var_model.coefs[0][0] * prices[pairs[0][0]] + var_model.coefs[0][1] * prices[pairs[0][1]]
        
        # Estimate the integrated volatility using the two-scale estimator
        integrated_volatility = []
        for i in range(len(coint_factor)):
            if i == 0:
                integrated_volatility.append(0)
            else:
                X = pd.DataFrame({'time': range(i+1)})
                y = coint_factor[:i+1]
                model = LinearRegression()
                model.fit(X, y)
                integrated_volatility.append(model.coef_[0][0] ** 2)
        
        # Compute the averaged instantaneous volatility
        averaged_volatility = np.sqrt(np.mean(integrated_volatility) / self.Time.resolution.total_seconds())
        
        return averaged_volatility

    def CheckAutoCorrelation(self, coint_factor):
        # Flatten the coint_factor array if it's 2-dimensional
        if coint_factor.ndim == 2:
            coint_factor = coint_factor.flatten()
        acf, confint, qstat, pvals = smt.acf(coint_factor, fft=False, alpha=0.05, qstat=True)
        if pvals[-1] < 0.05:
            self.Log("Co-integration factor has strong auto-correlation.")
        else:
            self.Log("Co-integration factor does not have strong auto-correlation.")

    def TradingStrategyWithAdHocBands(self,high_pairs,var_model, coint_factor, lower_band, upper_band):
        # Retrieve the mid-prices for the selected pairs
        prices = self.History([pair[0] for pair in high_pairs] + [pair[1] for pair in high_pairs], self.lookback, Resolution.Daily).close.unstack(level=0).dropna()
        
        # Compute the co-integration factor
        coint_factor = np.dot(prices, var_model.coefs[0])
        
        # Check if the co-integration factor crosses the lower or upper band
        if (coint_factor[-1] < lower_band).any():
            # Buy the co-integration factor
            for symbol1, symbol2 in high_pairs:
                if not self.Portfolio[symbol1].Invested and  not self.Portfolio[symbol2].Invested:
                    self.SetHoldings(symbol1, -0.075 / len(high_pairs))
                    self.SetHoldings(symbol2, 0.05 * var_model.coefs[:, 1].sum() / len(high_pairs))
                    self.tradedPairs.add((min(symbol1, symbol2), max(symbol1, symbol2)))
        elif (coint_factor[-1] > upper_band).any():
            # Sell the co-integration factor
            for symbol1, symbol2 in high_pairs:
                if not self.Portfolio[symbol1].Invested and  not self.Portfolio[symbol2].Invested:
                    self.SetHoldings(symbol1, 0.075 / len(high_pairs))
                    self.SetHoldings(symbol2, -0.05 * var_model.coefs[:, 1].sum() / len(high_pairs))
                    self.tradedPairs.add((min(symbol1, symbol2), max(symbol1, symbol2)))
        else:
            # Liquidate the positions
            for symbol1, symbol2 in high_pairs:
                self.Liquidate(symbol1)
                self.Liquidate(symbol2)


    def LiquidateIfUnrealizedGains(self):
        """Liquidates all positions to realize profits on the last day if there are any unrealized gains."""
        totalUnrealizedProfit = self.Portfolio.TotalUnrealizedProfit
        if totalUnrealizedProfit > 0:
            self.Liquidate()
            self.Log(f"Liquidated all positions due to unrealized gains on {self.Time} of {totalUnrealizedProfit}")           
    
    def OnEndOfAlgorithm(self):
        self.Log(f"Unique pairs traded during the backtest: {len(self.tradedPairs)}")
        for pair in self.tradedPairs:
            self.Log(f"Traded pair: {pair}")