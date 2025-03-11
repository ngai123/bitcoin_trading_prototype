import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bitcoin_scalping_day_trader')

class BitcoinScalpingDayTrader:
    """
    Advanced Bitcoin Trading Engine combining scalping and day trading strategies
    """
    
    def __init__(self, api_key=None, api_secret=None, test_mode=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.historical_data = None
        self.wallet = {"BTC": 0.0, "USD": 10000.0}  # Default testing wallet
        self.trade_history = []
        self.scalping_trades = []
        self.day_trades = []
        
        logger.info(f"Trading Engine initialized in {'test' if test_mode else 'live'} mode")
    
    def fetch_market_data(self, symbol='BTC/USD', timeframe='1m', limit=1000):
        """
        Fetch historical market data from API
        For scalping, we use 1-minute timeframe
        """
        try:
            # In a real implementation, this would connect to an exchange API
            # For demonstration, we'll generate realistic sample data
            
            # Create date range
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=limit)
            date_range = pd.date_range(start=start_time, end=end_time, periods=limit)
            
            # Generate realistic bitcoin price data
            base_price = 50000
            np.random.seed(42)  # For reproducibility
            price_changes = np.random.normal(0, 1, limit).cumsum() * 50
            closes = base_price + price_changes
            
            # Generate OHLCV data
            volatility = np.abs(price_changes) * 0.1
            opens = closes - np.random.normal(0, 1, limit) * volatility
            highs = np.maximum(opens, closes) + np.random.exponential(1, limit) * volatility
            lows = np.minimum(opens, closes) - np.random.exponential(1, limit) * volatility
            volumes = np.random.normal(100, 30, limit) * (1 + np.abs(price_changes) / 500)
            
            # Create DataFrame
            data = pd.DataFrame({
                'timestamp': date_range,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # Set timestamp as index
            data.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(data)} candles of market data for {symbol}")
            self.historical_data = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
    
    def add_indicators(self, data=None):
        """
        Calculate and add technical indicators to the data
        """
        if data is None:
            if self.historical_data is None:
                logger.error("No historical data available")
                return None
            data = self.historical_data.copy()
        
        # --- Scalping Indicators ---
        
        # Bollinger Bands (key for scalping)
        data['BB_middle'] = data['close'].rolling(window=20).mean()
        data['BB_std'] = data['close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + 2 * data['BB_std']
        data['BB_lower'] = data['BB_middle'] - 2 * data['BB_std']
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # RSI (good for scalping to identify overbought/oversold conditions)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator (useful for scalping)
        high_14 = data['high'].rolling(window=14).max()
        low_14 = data['low'].rolling(window=14).min()
        data['%K'] = 100 * ((data['close'] - low_14) / (high_14 - low_14))
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        # --- Day Trading Indicators ---
        
        # Moving Averages
        data['SMA7'] = data['close'].rolling(window=7).mean()
        data['SMA25'] = data['close'].rolling(window=25).mean()
        data['SMA99'] = data['close'].rolling(window=99).mean()
        
        # Exponential Moving Averages
        data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        # ATR for volatility measurement
        data['TR'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                np.abs(data['high'] - data['close'].shift()),
                np.abs(data['low'] - data['close'].shift())
            )
        )
        data['ATR'] = data['TR'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        data['OBV'] = np.where(
            data['close'] > data['close'].shift(),
            data['volume'],
            np.where(
                data['close'] < data['close'].shift(),
                -data['volume'],
                0
            )
        ).cumsum()
        
        logger.info("Technical indicators added to market data")
        return data
    
    def generate_signals(self, data=None):
        """
        Generate combined scalping and day trading signals
        """
        if data is None:
            if self.historical_data is None:
                logger.error("No historical data with indicators available")
                return None
            data = self.historical_data.copy()
        
        # Make sure indicators are added
        if 'RSI' not in data.columns:
            data = self.add_indicators(data)
        
        # Initialize signal columns
        data['scalping_signal'] = 0  # 0: no signal, 1: buy, -1: sell
        data['day_trading_signal'] = 0  # 0: no signal, 1: buy, -1: sell
        data['combined_signal'] = 0  # Final trading signal
        
        # --- Scalping Signals ---
        
        # 1. Bollinger Band signals for scalping
        data['signal_bb_scalp'] = 0
        # Buy when price touches lower band and RSI is below 30 (oversold)
        data.loc[(data['close'] <= data['BB_lower']) & (data['RSI'] < 30), 'signal_bb_scalp'] = 1
        # Sell when price touches upper band and RSI is above 70 (overbought)
        data.loc[(data['close'] >= data['BB_upper']) & (data['RSI'] > 70), 'signal_bb_scalp'] = -1
        
        # 2. Stochastic crossover for scalping
        data['signal_stoch_scalp'] = 0
        # Buy when %K crosses above %D while both are below 20
        data.loc[(data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & 
                 (data['%K'] < 20) & (data['%D'] < 20), 'signal_stoch_scalp'] = 1
        # Sell when %K crosses below %D while both are above 80
        data.loc[(data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & 
                 (data['%K'] > 80) & (data['%D'] > 80), 'signal_stoch_scalp'] = -1
        
        # --- Day Trading Signals ---
        
        # 1. MACD crossover for day trading
        data['signal_macd'] = 0
        data.loc[data['MACD'] > data['MACD_signal'], 'signal_macd'] = 1
        data.loc[data['MACD'] < data['MACD_signal'], 'signal_macd'] = -1
        
        # 2. Moving Average crossings for day trading
        data['signal_ma'] = 0
        data.loc[(data['SMA7'] > data['SMA25']) & (data['SMA7'].shift(1) <= data['SMA25'].shift(1)), 'signal_ma'] = 1
        data.loc[(data['SMA7'] < data['SMA25']) & (data['SMA7'].shift(1) >= data['SMA25'].shift(1)), 'signal_ma'] = -1
        
        # 3. RSI signals for day trading
        data['signal_rsi_day'] = 0
        data.loc[data['RSI'] < 30, 'signal_rsi_day'] = 1
        data.loc[data['RSI'] > 70, 'signal_rsi_day'] = -1
        
        # --- Combine Signals ---
        
        # Scalping signal - weighted average of scalping indicators
        data['scalping_signal'] = (0.6 * data['signal_bb_scalp'] + 
                                   0.4 * data['signal_stoch_scalp'])
        
        # Day trading signal - weighted average of day trading indicators
        data['day_trading_signal'] = (0.4 * data['signal_macd'] + 
                                      0.4 * data['signal_ma'] + 
                                      0.2 * data['signal_rsi_day'])
        
        # Apply thresholds for clear signals
        data.loc[data['scalping_signal'] > 0.3, 'scalping_signal'] = 1
        data.loc[data['scalping_signal'] < -0.3, 'scalping_signal'] = -1
        data.loc[(data['scalping_signal'] >= -0.3) & (data['scalping_signal'] <= 0.3), 'scalping_signal'] = 0
        
        data.loc[data['day_trading_signal'] > 0.3, 'day_trading_signal'] = 1
        data.loc[data['day_trading_signal'] < -0.3, 'day_trading_signal'] = -1
        data.loc[(data['day_trading_signal'] >= -0.3) & (data['day_trading_signal'] <= 0.3), 'day_trading_signal'] = 0
        
        # Final combined signal (weighted towards day trading for trend direction)
        # For the combined strategy: we want day trading to confirm overall trend
        # and scalping to provide precise entry/exit
        data['combined_signal'] = 0
        
        # Buy when day trading says buy or neutral, and scalping says buy
        data.loc[(data['day_trading_signal'] >= 0) & (data['scalping_signal'] == 1), 'combined_signal'] = 1
        
        # Sell when day trading says sell or neutral, and scalping says sell
        data.loc[(data['day_trading_signal'] <= 0) & (data['scalping_signal'] == -1), 'combined_signal'] = -1
                
        logger.info("Trading signals generated")
        return data
    
    def execute_trade(self, trade_type, amount, trade_style='combined'):
        """
        Execute a trade with a specified style (scalping or day trading)
        
        Parameters:
        -----------
        trade_type : str
            'buy' or 'sell'
        amount : float
            Amount to buy/sell in BTC
        trade_style : str
            'scalping', 'day_trading', or 'combined'
        """
        if self.test_mode:
            current_price = self.historical_data.iloc[-1]['close'] if self.historical_data is not None else 50000
            
            # Simulate trade execution
            if trade_type == 'buy':
                cost = amount * current_price
                if cost > self.wallet['USD']:
                    logger.warning(f"Insufficient USD balance for buy order: {cost} needed, {self.wallet['USD']} available")
                    return {
                        'success': False,
                        'message': 'Insufficient funds',
                        'time': datetime.now()
                    }
                
                # Execute buy
                self.wallet['USD'] -= cost
                self.wallet['BTC'] += amount
                
                trade_info = {
                    'success': True,
                    'time': datetime.now(),
                    'type': 'buy',
                    'style': trade_style,
                    'amount': amount,
                    'price': current_price,
                    'cost': cost,
                    'wallet': self.wallet.copy()
                }
                
            elif trade_type == 'sell':
                if amount > self.wallet['BTC']:
                    logger.warning(f"Insufficient BTC balance for sell order: {amount} needed, {self.wallet['BTC']} available")
                    return {
                        'success': False,
                        'message': 'Insufficient BTC',
                        'time': datetime.now()
                    }
                
                # Execute sell
                revenue = amount * current_price
                self.wallet['BTC'] -= amount
                self.wallet['USD'] += revenue
                
                trade_info = {
                    'success': True,
                    'time': datetime.now(),
                    'type': 'sell',
                    'style': trade_style,
                    'amount': amount,
                    'price': current_price,
                    'revenue': revenue,
                    'wallet': self.wallet.copy()
                }
            
            else:
                logger.error(f"Invalid trade type: {trade_type}")
                return {
                    'success': False,
                    'message': f'Invalid trade type: {trade_type}',
                    'time': datetime.now()
                }
            
            # Record trade by style
            if trade_style == 'scalping':
                self.scalping_trades.append(trade_info)
            elif trade_style == 'day_trading':
                self.day_trades.append(trade_info)
            
            # Record in overall history
            self.trade_history.append(trade_info)
            logger.info(f"{trade_style.capitalize()} {trade_type} order executed for {amount} BTC at ${current_price}")
            return trade_info
            
        else:
            # In a real implementation, this would connect to an exchange API
            logger.warning("Live trading not implemented")
            return {
                'success': False,
                'message': 'Live trading not implemented',
                'time': datetime.now()
            }
    
    def run_trading_bot(self, interval_seconds=60, max_iterations=None):
        """
        Run the trading bot combining scalping and day trading strategies
        """
        iteration = 0
        logger.info(f"Starting trading bot with {interval_seconds}s interval")
        
        try:
            while True:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                
                logger.info(f"Trading bot iteration {iteration}")
                
                # Update market data
                self.fetch_market_data()
                
                # Add indicators and generate signals
                data = self.add_indicators()
                data = self.generate_signals(data)
                
                # Get latest signals
                latest_data = data.iloc[-1]
                scalping_signal = latest_data['scalping_signal']
                day_trading_signal = latest_data['day_trading_signal']
                combined_signal = latest_data['combined_signal']
                
                # Get current balance
                balance = {
                    'USD': self.wallet['USD'],
                    'BTC': self.wallet['BTC'],
                    'current_price': latest_data['close']
                }
                
                # Execute trades based on combined strategy
                if combined_signal == 1:  # Buy signal
                    if balance['USD'] > 0:
                        # For scalping, use smaller position size (2% of USD balance)
                        amount_usd = balance['USD'] * 0.02
                        amount_btc = amount_usd / balance['current_price']
                        
                        # Execute buy
                        self.execute_trade('buy', amount_btc, 'combined')
                
                elif combined_signal == -1:  # Sell signal
                    if balance['BTC'] > 0:
                        # For scalping, sell a smaller portion (5% of BTC holdings)
                        amount_btc = balance['BTC'] * 0.05
                        
                        # Execute sell
                        self.execute_trade('sell', amount_btc, 'combined')
                
                # Wait for next iteration
                iteration += 1
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Waiting {interval_seconds}s for next iteration")
                    time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading bot: {str(e)}")
        
        logger.info("Trading bot stopped")
    
    def backtest(self, initial_balance=10000.0):
        """
        Backtest the combined scalping and day trading strategy
        """
        if self.historical_data is None:
            logger.error("No historical data available for backtesting")
            return None
        
        # Generate signals if not already done
        if 'combined_signal' not in self.historical_data.columns:
            self.historical_data = self.generate_signals()
        
        # Initialize backtest variables
        data = self.historical_data.copy()
        balance = initial_balance
        btc_holdings = 0.0
        trades = []
        
        # Set scalping and day trading parameters
        scalping_position_size = 0.02  # 2% of balance for scalping trades
        day_trading_position_size = 0.1  # 10% of balance for day trading
        
        # Iterate through the data
        for i in range(1, len(data)):
            combined_signal = data.iloc[i]['combined_signal']
            scalping_signal = data.iloc[i]['scalping_signal']
            day_trading_signal = data.iloc[i]['day_trading_signal']
            
            current_price = data.iloc[i]['close']
            current_time = data.index[i]
            
            # Execute combined strategy
            if combined_signal == 1 and balance > 0:  # Buy signal
                # Use scalping position size for precision entries
                amount_to_spend = balance * scalping_position_size
                btc_to_buy = amount_to_spend / current_price
                
                # Execute buy
                if btc_to_buy > 0:
                    balance -= amount_to_spend
                    btc_holdings += btc_to_buy
                    
                    trades.append({
                        'time': current_time,
                        'type': 'buy',
                        'strategy': 'combined',
                        'price': current_price,
                        'amount_btc': btc_to_buy,
                        'amount_usd': amount_to_spend,
                        'balance_btc': btc_holdings,
                        'balance_usd': balance
                    })
            
            elif combined_signal == -1 and btc_holdings > 0:  # Sell signal
                # Use scalping position size for precision exits
                btc_to_sell = btc_holdings * scalping_position_size
                amount_to_receive = btc_to_sell * current_price
                
                # Execute sell
                if btc_to_sell > 0:
                    balance += amount_to_receive
                    btc_holdings -= btc_to_sell
                    
                    trades.append({
                        'time': current_time,
                        'type': 'sell',
                        'strategy': 'combined',
                        'price': current_price,
                        'amount_btc': btc_to_sell,
                        'amount_usd': amount_to_receive,
                        'balance_btc': btc_holdings,
                        'balance_usd': balance
                    })
        
        # Calculate final portfolio value
        final_btc_value = btc_holdings * data.iloc[-1]['close']
        final_portfolio_value = balance + final_btc_value
        
        # Calculate performance metrics
        total_return = (final_portfolio_value - initial_balance) / initial_balance * 100
        buy_and_hold_return = (data.iloc[-1]['close'] - data.iloc[0]['close']) / data.iloc[0]['close'] * 100
        
        # Calculate Sharpe ratio
        portfolio_values = []
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            if i == 0:
                portfolio_values.append(initial_balance)
            else:
                portfolio_value = trades[-1]['balance_usd'] + trades[-1]['balance_btc'] * current_price if trades else initial_balance
                portfolio_values.append(portfolio_value)
        
        portfolio_values = np.array(portfolio_values)
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
        
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'final_btc_holdings': btc_holdings,
            'final_btc_value': final_btc_value,
            'final_portfolio_value': final_portfolio_value,
            'total_return_pct': total_return,
            'buy_and_hold_return_pct': buy_and_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'trades': trades
        }
        
        logger.info(f"Backtest completed: {len(trades)} trades, {total_return:.2f}% return")
        return results
    
    def visualize_strategy(self):
        """
        Visualize the trading strategy with indicators and signals
        """
        if self.historical_data is None:
            logger.error("No historical data available for visualization")
            return
        
        # Generate signals if not already done
        if 'combined_signal' not in self.historical_data.columns:
            self.historical_data = self.generate_signals()
        
        data = self.historical_data.copy()
        
        # Create figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Plot 1: Price with Bollinger Bands and Buy/Sell signals
        axs[0].plot(data.index, data['close'], label='Close Price', color='blue')
        axs[0].plot(data.index, data['BB_upper'], label='Upper BB', color='red', alpha=0.5)
        axs[0].plot(data.index, data['BB_lower'], label='Lower BB', color='green', alpha=0.5)
        axs[0].plot(data.index, data['BB_middle'], label='Middle BB', color='orange', alpha=0.5)
        axs[0].plot(data.index, data['SMA7'], label='SMA7', color='purple', alpha=0.5)
        axs[0].plot(data.index, data['SMA25'], label='SMA25', color='brown', alpha=0.5)
        
        # Plot buy signals
        buy_signals = data[data['combined_signal'] == 1]
        axs[0].scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', 
                      label='Buy Signal', s=100)
        
        # Plot sell signals
        sell_signals = data[data['combined_signal'] == -1]
        axs[0].scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', 
                      label='Sell Signal', s=100)
        
        axs[0].set_title('Bitcoin Price with Bollinger Bands and Trading Signals')
        axs[0].set_ylabel('Price ($)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: RSI
        axs[1].plot(data.index, data['RSI'], label='RSI', color='purple')
        axs[1].axhline(y=70, color='r', linestyle='-', alpha=0.3)
        axs[1].axhline(y=30, color='g', linestyle='-', alpha=0.3)
        axs[1].set_title('Relative Strength Index (RSI)')
        axs[1].set_ylabel('RSI')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot 3: Stochastic Oscillator
        axs[2].plot(data.index, data['%K'], label='%K', color='blue')
        axs[2].plot(data.index, data['%D'], label='%D', color='red')
        axs[2].axhline(y=80, color='r', linestyle='-', alpha=0.3)
        axs[2].axhline(y=20, color='g', linestyle='-', alpha=0.3)
        axs[2].set_title('Stochastic Oscillator')
        axs[2].set_ylabel('Value')
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot 4: MACD
        axs[3].plot(data.index, data['MACD'], label='MACD', color='blue')
        axs[3].plot(data.index, data['MACD_signal'], label='Signal Line', color='red')
        axs[3].bar(data.index, data['MACD_histogram'], label='Histogram', color='green', alpha=0.5)
        axs[3].set_title('MACD')
        axs[3].set_xlabel('Date')
        axs[3].set_ylabel('Value')
        axs[3].legend()
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize trading engine
    trader = BitcoinScalpingDayTrader(test_mode=True)
    
    # Fetch historical data
    trader.fetch_market_data()
    
    # Add indicators and generate signals
    data = trader.add_indicators()
    data = trader.generate_signals()
    
    # Visualize the strategy
    trader.visualize_strategy()
    
    # Run backtest
    results = trader.backtest()
    
    print(f"\nBacktest Results:")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Buy and Hold Return: {results['buy_and_hold_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Run the trading bot for a few iterations
    # trader.run_trading_bot(interval_seconds=5, max_iterations=10)
