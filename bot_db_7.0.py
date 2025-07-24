import streamlit as st
import threading
import queue
import time
import configparser
import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import logging

# --- Get the absolute path to the directory containing this script ---
# This makes file loading more robust.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Import Modular Components ---
try:
    from mt5_utils_2_0 import (
        initialize_mt5, get_current_open_positions,
        get_account_info, shutdown_mt5, mt5_initialized
    )
    from main_bot_2_0 import start_bot_threads, stop_bot_threads
except ImportError as e:
    st.error(f"Fatal Error: Could not import a required bot module (e.g., 'main_bot_2_0.py').")
    st.error(f"Please ensure all bot files have the '_2_0.py' suffix and are in the same directory.")
    st.exception(e)
    st.stop()

# --- Configuration Loading ---
def load_all_configs():
    """Loads all necessary configuration files using absolute paths."""
    configs = {}
    for filename in ['dashboard_config.ini', 'bot_config.ini', 'trading_modes.ini']:
        config_path = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(config_path):
            st.error(f"'{filename}' not found in the script's directory. Please run create_configs_2.0.py first.")
            st.stop()
        config = configparser.ConfigParser()
        config.read(config_path)
        configs[filename] = config
    return configs

configs = load_all_configs()
dashboard_config = configs['dashboard_config.ini']
bot_config = configs['bot_config.ini']
modes_config = configs['trading_modes.ini']

# --- Logging Setup ---
class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        self.log_queue.put(self.format(record))

log_queue = queue.Queue()
log_level_str = bot_config.get('BOT_SETTINGS', 'LOG_LEVEL', fallback='INFO').upper()
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level_str, logging.INFO))

if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
    root_logger.addHandler(StreamlitLogHandler(log_queue))

dashboard_logger = logging.getLogger(__name__)

# --- Streamlit Page and Session State Setup ---
st.set_page_config(layout="wide", page_title="AI Trading Bot Dashboard v7.0")
st.title("🤖 AI Trading Bot Dashboard v7.0")

def init_session_state():
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False
    if 'bot_thread' not in st.session_state:
        st.session_state.bot_thread = None
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if 'data_queue' not in st.session_state:
        st.session_state.data_queue = queue.Queue()
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'latest_status' not in st.session_state:
        st.session_state.latest_status = "Bot is idle."
    if 'latest_predictions' not in st.session_state:
        st.session_state.latest_predictions = {}
    if 'latest_signal' not in st.session_state:
        st.session_state.latest_signal = {"signal": "NONE", "message": "No signal yet."}
    
    try:
        st.session_state.mt5_login = dashboard_config.getint('MT5', 'LOGIN')
        st.session_state.mt5_password = dashboard_config.get('MT5', 'PASSWORD')
        st.session_state.mt5_server = dashboard_config.get('MT5', 'SERVER')
        st.session_state.telegram_bot_token = dashboard_config.get('TELEGRAM', 'BOT_TOKEN')
        st.session_state.telegram_chat_id = dashboard_config.get('TELEGRAM', 'CHAT_ID')
        st.session_state.telegram_message_level = dashboard_config.get('TELEGRAM', 'TELEGRAM_MESSAGE_LEVEL')
        
        st.session_state.current_trading_mode = bot_config.get('BOT_MODE', 'TRADING_MODE', fallback='SCALPING')
        st.session_state.primary_timeframe = bot_config.get('TRADING_CONFIG', 'PRIMARY_TIMEFRAME', fallback='TIMEFRAME_M15')

        global AUTO_START_BOT, REFRESH_RATE_MS
        AUTO_START_BOT = dashboard_config.getboolean('DASHBOARD', 'AUTO_START_BOT', fallback=False)
        REFRESH_RATE_MS = dashboard_config.getint('DASHBOARD', 'REFRESH_RATE_MS', fallback=1500)
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        st.error(f"Configuration file is missing a required section or option: {e}")
        st.stop()

init_session_state()

# --- Bot Control Functions ---
def start_bot():
    if st.session_state.bot_running:
        dashboard_logger.warning("Start command ignored: Bot is already running.")
        return

    dashboard_logger.info("Attempting to start bot threads...")
    st.session_state.stop_event.clear()

    returned_data_queue, returned_stop_event, returned_bot_thread = \
        start_bot_threads(
            st.session_state.data_queue, st.session_state.stop_event,
            st.session_state.mt5_login, st.session_state.mt5_password, st.session_state.mt5_server,
            st.session_state.telegram_bot_token, st.session_state.telegram_chat_id, st.session_state.telegram_message_level
        )
    
    if returned_bot_thread:
        st.session_state.data_queue, st.session_state.stop_event, st.session_state.bot_thread = \
            returned_data_queue, returned_stop_event, returned_bot_thread
        st.session_state.bot_running = True
        st.session_state.latest_status = "Bot started and running..."
        st.success("Bot started successfully!")
    else:
        st.session_state.bot_running = False
        st.session_state.latest_status = "Bot failed to start. Check logs for details."
        st.error("Bot failed to start. Please check the logs.")

def stop_bot():
    if not st.session_state.bot_running:
        dashboard_logger.warning("Stop command ignored: Bot is not running.")
        return

    dashboard_logger.info("Attempting to stop bot threads...")
    stop_bot_threads(st.session_state.stop_event, st.session_state.bot_thread)
    st.session_state.bot_running = False
    st.session_state.latest_status = "Bot stopped."
    st.info("Bot stopped.")

def apply_and_restart():
    """Updates the config file with selected settings and restarts the bot."""
    new_mode = st.session_state.selected_mode
    new_tf = st.session_state.selected_tf
    dashboard_logger.info(f"Applying new settings - Mode: {new_mode}, Primary TF: {new_tf}")
    
    # Update the bot_config.ini file
    bot_config.set('BOT_MODE', 'TRADING_MODE', new_mode)
    bot_config.set('TRADING_CONFIG', 'PRIMARY_TIMEFRAME', new_tf)
    with open(os.path.join(SCRIPT_DIR, 'bot_config.ini'), 'w') as configfile:
        bot_config.write(configfile)
    
    st.session_state.current_trading_mode = new_mode
    st.session_state.primary_timeframe = new_tf
    st.success(f"Settings updated. Restarting bot...")
    
    if st.session_state.bot_running:
        stop_bot()
        time.sleep(2) 
    start_bot()
    st.rerun()

# --- UI Layout ---
with st.sidebar:
    st.header("⚙️ Bot Controls")
    
    # Trading Mode Selector
    trading_modes = modes_config.sections()
    st.selectbox(
        "Select Trading Mode",
        options=trading_modes,
        index=trading_modes.index(st.session_state.current_trading_mode) if st.session_state.current_trading_mode in trading_modes else 0,
        key='selected_mode'
    )
    
    # Primary Timeframe Selector
    available_tfs = [tf.strip() for tf in bot_config.get('TRADING_CONFIG', 'TIMEFRAMES').split(',')]
    # Ensure the current primary timeframe is in the list, even if it's not in the default list
    if st.session_state.primary_timeframe not in available_tfs:
        available_tfs.append(st.session_state.primary_timeframe)
        
    st.selectbox(
        "Select Primary Timeframe",
        options=available_tfs,
        index=available_tfs.index(st.session_state.primary_timeframe) if st.session_state.primary_timeframe in available_tfs else 0,
        key='selected_tf'
    )

    st.button("Apply & Restart Bot", on_click=apply_and_restart, use_container_width=True, type="primary")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    col1.button("▶️ Start Bot", on_click=start_bot, use_container_width=True, disabled=st.session_state.bot_running)
    col2.button("⏹️ Stop Bot", on_click=stop_bot, use_container_width=True, disabled=not st.session_state.bot_running)
    
    st.divider()
    st.header("🔑 MT5 Credentials")
    st.number_input("Login", value=st.session_state.mt5_login, key="mt5_login", disabled=st.session_state.bot_running)
    st.text_input("Password", value=st.session_state.mt5_password, type="password", key="mt5_password", disabled=st.session_state.bot_running)
    st.text_input("Server", value=st.session_state.mt5_server, key="mt5_server", disabled=st.session_state.bot_running)

# Main content area placeholders
status_placeholder = st.empty()
account_info_placeholder = st.empty()
open_positions_placeholder = st.empty()
predictions_placeholder = st.empty()
signal_placeholder = st.empty()
log_display_area = st.empty()

# --- Dashboard Update Function ---
def update_dashboard_metrics():
    """Pulls data from queues and updates the UI elements."""
    while not log_queue.empty():
        st.session_state.log_messages.append(log_queue.get_nowait())
    
    while not st.session_state.data_queue.empty():
        try:
            update = st.session_state.data_queue.get_nowait()
            if update['type'] == 'status':
                st.session_state.latest_status = update['data']
            elif update['type'] == 'predictions':
                tf_string = update['data'].get('tf_string', 'UNKNOWN')
                st.session_state.latest_predictions[tf_string] = update['data']
            elif update['type'] == 'signal':
                st.session_state.latest_signal = update['data']
        except queue.Empty:
            break
        except Exception as e:
            dashboard_logger.error(f"Error processing data queue: {e}")

    # Update UI elements
    status_placeholder.info(f"**Status:** {st.session_state.latest_status} | **Mode:** {st.session_state.current_trading_mode} ({st.session_state.primary_timeframe}) | **MT5:** {'✅ Connected' if mt5_initialized() else '❌ Disconnected'}")
    
    acc_info = get_account_info()
    if acc_info:
        delta_color = "normal" if acc_info.get('profit', 0) >= 0 else "inverse"
        account_info_placeholder.metric(
            label="Account Balance | Equity | Profit",
            value=f"${acc_info.get('balance', 0):,.2f} | ${acc_info.get('equity', 0):,.2f}",
            delta=f"${acc_info.get('profit', 0):,.2f}", delta_color=delta_color
        )
    
    with predictions_placeholder.container():
        st.subheader("Latest Predictions")
        if st.session_state.latest_predictions:
            pred_list = [v for k, v in sorted(st.session_state.latest_predictions.items())]
            pred_df = pd.DataFrame(pred_list)
            if not pred_df.empty:
                formatters = {'up_proba': '{:.2%}', 'down_proba': '{:.2%}'}
                for col, f in formatters.items():
                    if col in pred_df.columns: pred_df[col] = pred_df[col].map(f.format)
                if 'last_bar_time' in pred_df.columns:
                    pred_df['last_bar_time'] = pd.to_datetime(pred_df['last_bar_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("Waiting for first predictions...")

    with signal_placeholder.container():
        st.subheader("Current Trade Signal")
        signal = st.session_state.latest_signal.get('signal', 'NONE')
        message = st.session_state.latest_signal.get('message', 'No signal yet.')
        if signal == "BUY": st.success(f"**BUY:** {message}")
        elif signal == "SELL": st.error(f"**SELL:** {message}")
        else: st.info(f"**HOLD/NO_TRADE:** {message}")
            
    with open_positions_placeholder.container():
        st.subheader("Open Positions")
        positions_df = get_current_open_positions()
        if not positions_df.empty: st.dataframe(positions_df, use_container_width=True)
        else: st.info("No open positions.")
    
    with log_display_area.container():
        st.text_area("Live Logs", value="\n".join(st.session_state.log_messages[-100:]), height=300, key="log_area")

# --- Main Application Loop ---
update_dashboard_metrics()

if AUTO_START_BOT and not st.session_state.bot_running:
    dashboard_logger.info("Auto-starting bot as per configuration...")
    start_bot()
    time.sleep(1)
    st.rerun()

if st.session_state.bot_running:
    time.sleep(REFRESH_RATE_MS / 1000.0)
    st.rerun()
