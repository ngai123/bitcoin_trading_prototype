import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Import the trading engine
from trading_engine import BitcoinScalpingDayTrader

app = Flask(__name__)

# Global trader instance
trader = BitcoinScalpingDayTrader(test_mode=True)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/fetch-data', methods=['POST'])
def fetch_market_data():
    """Fetch and return market data"""
    try:
        # Fetch market data
        data = trader.fetch_market_data()
        
        # Convert data to JSON-serializable format
        data_json = data.reset_index().to_dict(orient='records')
        
        return jsonify({
            'success': True, 
            'data': data_json,
            'columns': list(data.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-indicators', methods=['POST'])
def add_indicators():
    """Add technical indicators to market data"""
    try:
        # Add indicators
        data = trader.add_indicators()
        
        # Convert data to JSON-serializable format
        data_json = data.reset_index().to_dict(orient='records')
        
        return jsonify({
            'success': True, 
            'data': data_json,
            'columns': list(data.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate-signals', methods=['POST'])
def generate_signals():
    """Generate trading signals"""
    try:
        # Generate signals
        data = trader.generate_signals()
        
        # Convert data to JSON-serializable format
        data_json = data.reset_index().to_dict(orient='records')
        
        return jsonify({
            'success': True, 
            'data': data_json,
            'columns': list(data.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtesting and return results"""
    try:
        # Get initial balance from request or use default
        initial_balance = float(request.form.get('initial_balance', 10000.0))
        
        # Run backtest
        results = trader.backtest(initial_balance)
        
        return jsonify({
            'success': True, 
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/visualize-strategy', methods=['POST'])
def visualize_strategy():
    """Generate and return strategy visualization"""
    try:
        # Clear any existing plots
        plt.close('all')
        
        # Create a new figure
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), 
                                sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Get historical data with indicators and signals
        data = trader.historical_data
        
        if data is None:
            return jsonify({'success': False, 'error': 'No historical data available'})
        
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
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the plot as base64 string
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        
        return jsonify({
            'success': True, 
            'plot': plot_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)