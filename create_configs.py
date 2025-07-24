import configparser
import os

def create_main_bot_config():
    """Generates the main bot_config.ini file."""
    config = configparser.ConfigParser()
    
    config['BOT_MODE'] = {
        'TRADING_MODE': 'SCALPING' # Default mode: SCALPING, INTRADAY, or SWING
    }

    config['TRADING_CONFIG'] = {
        'SYMBOL': 'XAUUSDm',
        'PRIMARY_TIMEFRAME': 'TIMEFRAME_M15',
        'TIMEFRAMES': 'TIMEFRAME_H4,TIMEFRAME_H1,TIMEFRAME_M30',
        'BARS_TO_FETCH_FOR_FEATURES': '500',
    }
    
    config['RISK_CONFIG'] = {
        # This is a base setting that is used across all modes
        'TRADE_SIZE_PERCENT': '0.1' 
    }
    
    config['STRATEGY_CONFIG'] = {
        'MIN_CONFIDENCE': '0.55',
        'AGGREGATE_BUY_THRESHOLD_H4': '0.50',
        'AGGREGATE_BUY_THRESHOLD_H1': '0.60',
        'AGGREGATE_BUY_THRESHOLD_M30': '0.60',
        'AGGREGATE_SELL_THRESHOLD_H4': '0.50',
        'AGGREGATE_SELL_THRESHOLD_H1': '0.65',
        'AGGREGATE_SELL_THRESHOLD_M30': '0.60',
        'ENABLE_NEWS_SENTIMENT': 'False',
        'NEWS_SENTIMENT_BUY_BOOST_THRESHOLD': '0.5',
        'NEWS_SENTIMENT_SELL_BOOST_THRESHOLD': '-0.5'
    }
    
    config['BOT_SETTINGS'] = {
        'POLL_INTERVAL_SECONDS': '10',
        'SENTIMENT_UPDATE_INTERVAL_MINUTES': '30',
        'LOG_LEVEL': 'INFO'
    }

    with open('bot_config.ini', 'w') as configfile:
        config.write(configfile)
    print("Generated bot_config.ini")

def create_trading_modes_config():
    """Generates the trading_modes.ini file with profiles for different styles."""
    config = configparser.ConfigParser()

    config['SCALPING'] = {
        'max_open_trades': '3',
        'risk_reward': '1.5',
        'use_atr_sltp': 'True',
        'atr_timeframe': 'TIMEFRAME_M5',
        'atr_multiplier': '1.0',
        'enable_trailing_stop': 'True',
        'trailing_stop_atr_multiplier': '0.8'
    }

    config['INTRADAY'] = {
        'max_open_trades': '2',
        'risk_reward': '2.0',
        'use_atr_sltp': 'True',
        'atr_timeframe': 'TIMEFRAME_M15',
        'atr_multiplier': '2.0',
        'enable_trailing_stop': 'True',
        'trailing_stop_atr_multiplier': '1.5'
    }

    config['SWING'] = {
        'max_open_trades': '1',
        'risk_reward': '3.0',
        'use_atr_sltp': 'True',
        'atr_timeframe': 'TIMEFRAME_H1',
        'atr_multiplier': '3.0',
        'enable_trailing_stop': 'False',
        'trailing_stop_atr_multiplier': '2.0'
    }

    with open('trading_modes.ini', 'w') as configfile:
        config.write(configfile)
    print("Generated trading_modes.ini")


def create_dashboard_config():
    """Generates the dashboard_config.ini file with default settings."""
    config = configparser.ConfigParser()
    
    config['DASHBOARD'] = {
        'AUTO_START_BOT': 'False',
        'REFRESH_RATE_MS': '1500'
    }
    
    config['MT5'] = {
        'LOGIN': '269166497',
        'PASSWORD': 'Prasath@009',
        'SERVER': 'Exness-MT5Trial17'
    }
    
    config['TELEGRAM'] = {
        'BOT_TOKEN': '7602450918:AAGiooZynvXQGaGFzvgana7KjVncPvJY3ps',
        'CHAT_ID': '5361041668',
        'TELEGRAM_MESSAGE_LEVEL': 'INFO' 
    }

    with open('dashboard_config.ini', 'w') as configfile:
        config.write(configfile)
    print("Generated dashboard_config.ini")

if __name__ == "__main__":
    create_main_bot_config()
    create_trading_modes_config()
    create_dashboard_config()
    print("\nConfiguration files generated.")
    print("IMPORTANT: Please update 'dashboard_config.ini' with your actual MT5 and Telegram credentials.")
