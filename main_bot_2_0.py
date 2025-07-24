import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import joblib
import xgboost as xgb
import pytz
import warnings
import queue
import threading
import os
import logging
import configparser

# --- Get the absolute path to the directory containing this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Import Modular Components ---
from mt5_utils_2_0 import (
    initialize_mt5, shutdown_mt5, get_latest_bars,
    get_current_open_positions, get_account_info,
    calculate_trade_volume, send_trade_request,
    close_position, modify_position, mt5_initialized
)
from telegram_handler_2_0 import TelegramHandler
from news_sentiment_analyzer_2_0 import NewsSentimentAnalyzer
from strategy_manager_2_0 import StrategyManager
from timeframe_predictor_2_0 import TimeframePredictor

# --- Global Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State Management ---
_bot_logic_thread = None
_data_queue = None
_stop_event = None
_main_bot_instance = None

# --- Configuration Loading ---
def load_config():
    """Loads all necessary configuration files using absolute paths."""
    bot_config = configparser.ConfigParser()
    bot_config_path = os.path.join(SCRIPT_DIR, 'bot_config.ini')
    if not os.path.exists(bot_config_path):
        raise FileNotFoundError("bot_config.ini not found. Please run create_configs_2.0.py.")
    bot_config.read(bot_config_path)

    dashboard_config = configparser.ConfigParser()
    dashboard_config_path = os.path.join(SCRIPT_DIR, 'dashboard_config.ini')
    if not os.path.exists(dashboard_config_path):
        raise FileNotFoundError("dashboard_config.ini not found. Please run create_configs_2.0.py.")
    dashboard_config.read(dashboard_config_path)

    modes_config = configparser.ConfigParser()
    modes_config_path = os.path.join(SCRIPT_DIR, 'trading_modes.ini')
    if not os.path.exists(modes_config_path):
        raise FileNotFoundError("trading_modes.ini not found. Please run create_configs_2.0.py.")
    modes_config.read(modes_config_path)
    
    return bot_config, dashboard_config, modes_config

# --- Main Bot Class ---
class MainBot:
    def __init__(self, config: configparser.ConfigParser, dashboard_config: configparser.ConfigParser, modes_config: configparser.ConfigParser, data_queue: queue.Queue, stop_event: threading.Event):
        global _main_bot_instance
        _main_bot_instance = self

        self.config = config
        self.dashboard_config = dashboard_config
        self.modes_config = modes_config
        self.data_queue = data_queue
        self.stop_event = stop_event

        self._load_bot_configuration()
        self._initialize_services()
        self._initialize_predictors()
        self._initialize_strategy_manager()

        logger.info("MainBot successfully initialized.")

    def _load_bot_configuration(self):
        """Loads core settings and the selected trading mode's parameters."""
        self.trading_mode = self.config.get('BOT_MODE', 'TRADING_MODE', fallback='SCALPING').upper()
        if not self.modes_config.has_section(self.trading_mode):
            raise ValueError(f"Trading mode '{self.trading_mode}' not found in trading_modes.ini")
        
        logger.info(f"--- Loading settings for TRADING MODE: {self.trading_mode} ---")
        mode_settings = self.modes_config[self.trading_mode]

        # Trading Config
        self.symbol = self.config.get('TRADING_CONFIG', 'SYMBOL')
        self.primary_timeframe_str = self.config.get('TRADING_CONFIG', 'PRIMARY_TIMEFRAME')
        self.timeframe_strings = [tf.strip() for tf in self.config.get('TRADING_CONFIG', 'TIMEFRAMES').split(',')]
        
        # Risk Config
        self.trade_size_percent = self.config.getfloat('RISK_CONFIG', 'TRADE_SIZE_PERCENT')
        self.max_open_trades = mode_settings.getint('max_open_trades', 1)
        self.risk_reward = mode_settings.getfloat('risk_reward', 1.5)
        self.use_atr_sltp = mode_settings.getboolean('use_atr_sltp', True)
        self.atr_timeframe = mode_settings.get('atr_timeframe', 'TIMEFRAME_M15')
        self.atr_multiplier = mode_settings.getfloat('atr_multiplier', 1.5)
        self.enable_trailing_stop = mode_settings.getboolean('enable_trailing_stop', True)
        self.trailing_stop_atr_multiplier = mode_settings.getfloat('trailing_stop_atr_multiplier', 1.0)
        
        self.all_timeframes = list(set(self.timeframe_strings + [self.primary_timeframe_str, self.atr_timeframe]))
        logger.info(f"Bot will process the following timeframes: {self.all_timeframes}")
        
        self.bars_to_fetch = self.config.getint('TRADING_CONFIG', 'BARS_TO_FETCH_FOR_FEATURES')

        # Strategy Config
        self.strategy_config = {}
        strategy_options = self.config.options('STRATEGY_CONFIG')
        for option in strategy_options:
            if option.lower() == 'enable_news_sentiment':
                self.strategy_config[option] = self.config.getboolean('STRATEGY_CONFIG', option)
            else:
                self.strategy_config[option] = self.config.getfloat('STRATEGY_CONFIG', option)
        
        # --- NEW: Load Telegram Report settings ---
        self.enable_periodic_pnl_report = self.config.getboolean('TELEGRAM_REPORTS', 'enable_periodic_pnl_report', fallback=True)
        self.pnl_report_interval_minutes = self.config.getint('TELEGRAM_REPORTS', 'pnl_report_interval_minutes', fallback=60)
        
        # Bot Settings
        self.poll_interval = self.config.getint('BOT_SETTINGS', 'POLL_INTERVAL_SECONDS')
        log_level_str = self.config.get('BOT_SETTINGS', 'LOG_LEVEL', fallback='INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)
        logger.info(f"Bot configuration loaded. Max open trades: {self.max_open_trades}, Trailing Stop: {self.enable_trailing_stop}")

    def _initialize_services(self):
        """Initializes MT5 connection, Telegram handler, and P&L tracking."""
        self.mt5_login = self.dashboard_config.getint('MT5', 'LOGIN')
        self.mt5_password = self.dashboard_config.get('MT5', 'PASSWORD')
        self.mt5_server = self.dashboard_config.get('MT5', 'SERVER')

        telegram_token = self.dashboard_config.get('TELEGRAM', 'BOT_TOKEN')
        telegram_chat_id = self.dashboard_config.get('TELEGRAM', 'CHAT_ID')
        telegram_level = self.dashboard_config.get('TELEGRAM', 'TELEGRAM_MESSAGE_LEVEL')

        self.telegram_handler = TelegramHandler(telegram_token, telegram_chat_id, telegram_level)
        self.send_telegram_message(f"Bot starting in {self.trading_mode} mode...", "INFO")

        # --- NEW: Initialize P&L tracking variables ---
        self.last_pnl_report_time = datetime.now(pytz.utc)
        self.periodic_realized_pnl = 0.0

    def _initialize_predictors(self):
        """Loads the ML models and scalers for each timeframe."""
        self.timeframe_predictors = {}
        logger.info("Loading ML models and scalers for all timeframes...")

        for tf_str in self.all_timeframes:
            mt5_tf = getattr(mt5, tf_str, None)
            if mt5_tf is None:
                logger.error(f"Invalid timeframe '{tf_str}' in config. Skipping.")
                continue

            model_path = os.path.join(SCRIPT_DIR, f"{self.symbol}_{tf_str}_xgboost_model.json")
            scaler_path = os.path.join(SCRIPT_DIR, f"{self.symbol}_{tf_str}_scaler.pkl")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.error(f"Model or scaler not found for {tf_str}. Searched for '{model_path}' and '{scaler_path}'. Please train first.")
                continue

            try:
                scaler = joblib.load(scaler_path)
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                
                predictor = TimeframePredictor(self.symbol, mt5_tf, tf_str, scaler, model, self.bars_to_fetch)
                predictor.fetch_bars = get_latest_bars
                self.timeframe_predictors[mt5_tf] = predictor
                logger.info(f"Successfully loaded model and scaler for {tf_str}.")

            except Exception as e:
                logger.error(f"Failed to load predictor for {tf_str}: {e}", exc_info=True)
        
        if not self.timeframe_predictors:
            logger.critical("No models were loaded. The bot cannot make predictions. Halting.")
            self.stop_event.set()

    def _initialize_strategy_manager(self):
        """Initializes the strategy and news sentiment components."""
        self.news_analyzer = NewsSentimentAnalyzer(self.config['BOT_SETTINGS'])
        self.strategy_manager = StrategyManager(self.strategy_config, self.news_analyzer)
        logger.info("Strategy Manager initialized.")
        
    def send_telegram_message(self, message: str, level: str = 'INFO'):
        """Convenience method to send a message via Telegram."""
        if self.telegram_handler:
            self.telegram_handler.send_message(message, level)

    def run_bot_loop(self):
        """The main operational loop of the bot."""
        logger.info("Bot logic loop started. Waiting for signals...")
        while not self.stop_event.is_set():
            try:
                if not mt5_initialized():
                    logger.warning("MT5 connection lost. Attempting to reconnect...")
                    if not initialize_mt5(self.mt5_login, self.mt5_password, self.mt5_server):
                        logger.error("MT5 reconnection failed. Will retry in next cycle.")
                        time.sleep(self.poll_interval)
                        continue
                    else:
                        logger.info("MT5 reconnected successfully.")
                self.run_trading_cycle()
            except Exception as e:
                logger.error(f"An unhandled error occurred in the main bot loop: {e}", exc_info=True)
                self.send_telegram_message(f"CRITICAL BOT ERROR: {e}. Check logs.", "CRITICAL")
            time.sleep(self.poll_interval)
        logger.info("Bot logic loop has been stopped.")
        shutdown_mt5()

    def run_trading_cycle(self):
        """Executes a single cycle of trading logic."""
        logger.info(f"--- New Trading Cycle [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")
        
        # --- NEW: Handle periodic P&L reporting ---
        self._handle_periodic_pnl_report()

        current_predictions = {}
        for mt5_tf, predictor in self.timeframe_predictors.items():
            try:
                prediction_result = predictor.predict()
                current_predictions[predictor.tf_string] = prediction_result
                self.update_dashboard('predictions', prediction_result)
            except Exception as e:
                logger.error(f"Error getting prediction for {predictor.tf_string}: {e}", exc_info=True)
        
        if not current_predictions:
            logger.warning("No predictions were generated in this cycle. Skipping.")
            return

        trade_signal, signal_message = self.strategy_manager.determine_trade_signal(current_predictions)
        logger.info(f"Strategy Signal: {trade_signal} | Reason: {signal_message}")
        self.update_dashboard('signal', {'signal': trade_signal, 'message': signal_message})

        positions_df = get_current_open_positions(self.symbol)
        
        if self.enable_trailing_stop and not positions_df.empty:
            self._manage_trailing_stops(positions_df, current_predictions)

        last_tick = mt5.symbol_info_tick(self.symbol)
        if not last_tick:
            logger.error(f"Could not get current price for {self.symbol}. Skipping trade execution.")
            return
        current_price = last_tick.ask if trade_signal == "BUY" else last_tick.bid

        atr_prediction_data = current_predictions.get(self.atr_timeframe)
        self.execute_trade_logic(trade_signal, current_price, positions_df, atr_prediction_data)

    def _manage_trailing_stops(self, open_positions, current_predictions):
        """
        Checks and updates the stop loss for open positions if the trailing stop condition is met.
        """
        logger.debug("Managing trailing stops for open positions...")
        atr_data = current_predictions.get(self.atr_timeframe)
        if not (atr_data and atr_data.get('status') == 'OK'):
            logger.warning(f"Cannot manage trailing stops: ATR data from {self.atr_timeframe} is not available.")
            return

        atr_value = atr_data['features'].get('atr')
        if not atr_value:
            logger.warning("Cannot manage trailing stops: ATR value is missing.")
            return

        trailing_distance = atr_value * self.trailing_stop_atr_multiplier

        for _, position in open_positions.iterrows():
            current_sl = position['sl']
            current_tp = position['tp']
            ticket = position['ticket']
            
            last_tick = mt5.symbol_info_tick(self.symbol)
            if not last_tick:
                continue

            if position['type'] == mt5.ORDER_TYPE_BUY:
                new_sl = last_tick.bid - trailing_distance
                if new_sl > current_sl:
                    logger.info(f"Trailing SL for BUY ticket #{ticket}. Old SL: {current_sl:.4f}, New SL: {new_sl:.4f}")
                    self.send_telegram_message(f"🔒 Trailing SL for BUY #{ticket} to {new_sl:.4f}", "INFO")
                    modify_position(ticket, self.symbol, sl=new_sl, tp=current_tp)

            elif position['type'] == mt5.ORDER_TYPE_SELL:
                new_sl = last_tick.ask + trailing_distance
                if new_sl < current_sl or current_sl == 0.0:
                    logger.info(f"Trailing SL for SELL ticket #{ticket}. Old SL: {current_sl:.4f}, New SL: {new_sl:.4f}")
                    self.send_telegram_message(f"🔒 Trailing SL for SELL #{ticket} to {new_sl:.4f}", "INFO")
                    modify_position(ticket, self.symbol, sl=new_sl, tp=current_tp)

    # --- NEW: P&L Reporting Functions ---
    def _handle_periodic_pnl_report(self):
        """Checks if it's time to send a periodic P&L report and sends it."""
        if not self.enable_periodic_pnl_report:
            return
        
        time_since_last_report = datetime.now(pytz.utc) - self.last_pnl_report_time
        if time_since_last_report >= timedelta(minutes=self.pnl_report_interval_minutes):
            logger.info("Sending periodic P&L report to Telegram...")
            
            # Get unrealized P&L from open positions
            open_positions = get_current_open_positions(self.symbol)
            unrealized_pnl = open_positions['profit'].sum() if not open_positions.empty else 0.0
            
            total_pnl = self.periodic_realized_pnl + unrealized_pnl
            
            # Format the message
            report_msg = (
                f"📊 **P&L Report ({self.trading_mode})**\n"
                f"------------------------------------\n"
                f"**Realized P&L (since last report):** ${self.periodic_realized_pnl:,.2f}\n"
                f"**Unrealized P&L (open trades):** ${unrealized_pnl:,.2f}\n"
                f"**Total P&L for Period:** ${total_pnl:,.2f}"
            )
            
            self.send_telegram_message(report_msg, "INFO")
            
            # Reset for the next period
            self.last_pnl_report_time = datetime.now(pytz.utc)
            self.periodic_realized_pnl = 0.0

    def _report_closed_trade(self, position):
        """Sends a Telegram message for a single closed trade and updates periodic P&L."""
        pnl = position['profit']
        trade_type = "BUY" if position['type'] == mt5.ORDER_TYPE_BUY else "SELL"
        
        # Add to the periodic total
        self.periodic_realized_pnl += pnl
        
        # Format the message
        if pnl >= 0:
            msg = f"✅ **Trade Closed (Profit)**\n`{trade_type} {position['volume']} {self.symbol}`\n**P&L:** `${pnl:,.2f}`"
        else:
            msg = f"❌ **Trade Closed (Loss)**\n`{trade_type} {position['volume']} {self.symbol}`\n**P&L:** `${pnl:,.2f}`"
            
        self.send_telegram_message(msg, "WARNING") # Use WARNING level to ensure it gets sent


    def execute_trade_logic(self, trade_signal, current_price, open_positions, atr_data):
        """Handles the logic for entering, exiting, or holding positions."""
        if not open_positions.empty:
            for _, position in open_positions.iterrows():
                # --- MODIFIED: Report P&L on close ---
                if position['type'] == mt5.ORDER_TYPE_BUY and trade_signal == "SELL":
                    logger.info(f"Closing BUY position #{position['ticket']} due to SELL signal.")
                    self._report_closed_trade(position) # Report before closing
                    close_position(position['ticket'], position['volume'])
                
                elif position['type'] == mt5.ORDER_TYPE_SELL and trade_signal == "BUY":
                    logger.info(f"Closing SELL position #{position['ticket']} due to BUY signal.")
                    self._report_closed_trade(position) # Report before closing
                    close_position(position['ticket'], position['volume'])
        
        if len(open_positions) < self.max_open_trades and trade_signal != "NONE":
            logger.info(f"Signal '{trade_signal}' received. Current trades: {len(open_positions)}/{self.max_open_trades}. Checking to open new trade.")
            self.open_new_trade(trade_signal, current_price, atr_data)
        elif trade_signal != "NONE":
            logger.info(f"Signal '{trade_signal}' received, but max trades ({self.max_open_trades}) already open. Holding position.")


    def open_new_trade(self, trade_signal, price, atr_data):
        """Opens a new trade based on the signal, using ATR for SL/TP if configured."""
        logger.info(f"Attempting to open new {trade_signal} trade for {self.symbol} at ~{price}")
        
        account_info = get_account_info()
        if not account_info:
            logger.error("Could not get account info to calculate trade volume.")
            return

        volume = calculate_trade_volume(account_info['balance'], self.trade_size_percent, price, self.symbol)
        if volume <= 0:
            logger.warning(f"Calculated trade volume is {volume}. Cannot open trade.")
            return

        sl = 0.0
        tp = 0.0

        if self.use_atr_sltp and atr_data and atr_data.get('status') == 'OK':
            atr_value = atr_data['features'].get('atr')
            if atr_value:
                logger.info(f"Using ATR ({self.atr_timeframe}) for SL/TP calculation. ATR Value: {atr_value:.5f}")
                sl_distance = atr_value * self.atr_multiplier
                tp_distance = sl_distance * self.risk_reward
                if trade_signal == "BUY":
                    sl = price - sl_distance
                    tp = price + tp_distance
                else: # SELL
                    sl = price + sl_distance
                    tp = price - tp_distance
            else:
                logger.warning("ATR value not found in prediction data. Using fallback.")
                self.use_atr_sltp = False 
        
        if not self.use_atr_sltp or sl == 0.0:
            logger.warning("Using fixed factor for SL/TP calculation as a fallback.")
            if trade_signal == "BUY":
                sl = price * (1 - 0.007) 
                tp = price * (1 + 0.007 * self.risk_reward)
            else: # SELL
                sl = price * (1 + 0.007)
                tp = price * (1 - 0.007 * self.risk_reward)

        trade_type = mt5.ORDER_TYPE_BUY if trade_signal == "BUY" else mt5.ORDER_TYPE_SELL
        result = send_trade_request(self.symbol, trade_type, volume, price, sl, tp)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            msg = f"✅ **New Trade Opened**\n`{trade_signal} {volume} {self.symbol} @ {result.price:.4f}`\n**SL:** `{sl:.4f}`\n**TP:** `{tp:.4f}`"
            logger.info(msg)
            self.send_telegram_message(msg, "INFO")
        else:
            comment = result.comment if result else "No result object"
            msg = f"❌ Failed to open {trade_signal} trade. Reason: {comment}"
            logger.error(msg)
            self.send_telegram_message(msg, "ERROR")

    def update_dashboard(self, update_type: str, data: dict):
        """Puts data into the queue for the Streamlit dashboard to display."""
        try:
            if not self.data_queue.full():
                self.data_queue.put({'type': update_type, 'data': data}, block=False)
        except queue.Full:
            logger.warning("Dashboard data queue is full. Skipping update.")
        except Exception as e:
            logger.error(f"Error updating dashboard queue: {e}")

# --- Thread Management Functions (called by dashboard) ---
def start_bot_threads(data_queue, stop_event, mt5_login, mt5_password, mt5_server, telegram_token, telegram_chat_id, telegram_level):
    """Initializes and starts the main bot logic in a separate thread."""
    global _bot_logic_thread, _main_bot_instance
    
    if _bot_logic_thread and _bot_logic_thread.is_alive():
        logger.warning("Bot thread is already running.")
        return data_queue, stop_event, _bot_logic_thread

    logger.info("Attempting to start bot threads...")
    
    try:
        if not initialize_mt5(mt5_login, mt5_password, mt5_server):
            raise ConnectionError("Failed to connect to MetaTrader 5.")

        bot_config, dashboard_config, modes_config = load_config()
        _main_bot_instance = MainBot(bot_config, dashboard_config, modes_config, data_queue, stop_event)
        
        _bot_logic_thread = threading.Thread(target=_main_bot_instance.run_bot_loop, daemon=True)
        _bot_logic_thread.start()
        
        logger.info("Bot logic thread started successfully.")
        return data_queue, stop_event, _bot_logic_thread

    except Exception as e:
        logger.critical(f"Failed to start bot threads: {e}", exc_info=True)
        if stop_event:
            stop_event.set()
        shutdown_mt5()
        return None, None, None

def stop_bot_threads(stop_event, bot_thread):
    """Signals the bot thread to stop and waits for it to terminate."""
    logger.info("Attempting to stop bot threads...")
    if stop_event:
        stop_event.set()
    
    if bot_thread and bot_thread.is_alive():
        logger.info("Waiting for bot logic thread to terminate...")
        bot_thread.join(timeout=15)
        if bot_thread.is_alive():
            logger.warning("Bot logic thread did not terminate gracefully.")
        else:
            logger.info("Bot logic thread stopped.")
    
    shutdown_mt5()
    logger.info("All bot services shut down.")

if __name__ == '__main__':
    logger.info("Running bot in standalone mode (without dashboard).")
    try:
        _, dashboard_config, _ = load_config()
        login = dashboard_config.getint('MT5', 'LOGIN')
        password = dashboard_config.get('MT5', 'PASSWORD')
        server = dashboard_config.get('MT5', 'SERVER')
        token = dashboard_config.get('TELEGRAM', 'BOT_TOKEN')
        chat_id = dashboard_config.get('TELEGRAM', 'CHAT_ID')
        level = dashboard_config.get('TELEGRAM', 'TELEGRAM_MESSAGE_LEVEL')

        q = queue.Queue()
        event = threading.Event()
        data_q, stop_e, bot_t = start_bot_threads(q, event, login, password, server, token, chat_id, level)

        if bot_t:
            logger.info("Bot is running. Press Ctrl+C to stop.")
            while bot_t.is_alive():
                time.sleep(1)
        else:
            logger.critical("Bot failed to start.")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping bot...")
    except FileNotFoundError as e:
        logger.critical(f"Configuration file error: {e}")
    except Exception as e:
        logger.critical(f"An error occurred during standalone startup: {e}", exc_info=True)
    finally:
        if 'stop_e' in locals() and 'bot_t' in locals():
            stop_bot_threads(stop_e, bot_t)
        logger.info("Bot shutdown complete.")
