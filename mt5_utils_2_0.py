import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# Global variable to track MT5 initialization status
_mt5_initialized_status = False

def mt5_initialized() -> bool:
    """Returns the current MT5 initialization status."""
    return _mt5_initialized_status

def initialize_mt5(login: int, password: str, server: str, retries=3, delay=5) -> bool:
    """
    Initializes the MetaTrader 5 connection with retry logic.
    Returns True if successful, False otherwise.
    """
    global _mt5_initialized_status
    if _mt5_initialized_status:
        logger.debug("MT5 connection is already active.")
        return True

    for attempt in range(retries):
        logger.info(f"Attempting to initialize MT5... (Attempt {attempt + 1}/{retries})")
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize() failed. Error: {mt5.last_error()}")
                time.sleep(delay)
                continue

            if not mt5.login(login=login, password=password, server=server):
                logger.error(f"MT5 login() failed. Account: {login}, Server: {server}. Error: {mt5.last_error()}")
                mt5.shutdown()
                time.sleep(delay)
                continue
            
            logger.info(f"MT5 successfully initialized and logged in to account {login}.")
            _mt5_initialized_status = True
            return True

        except Exception as e:
            logger.error(f"An unexpected error occurred during MT5 initialization: {e}", exc_info=True)
            time.sleep(delay)

    logger.critical("Failed to initialize MT5 after multiple retries.")
    _mt5_initialized_status = False
    return False

def shutdown_mt5():
    """Shuts down the MetaTrader 5 connection if it's active."""
    global _mt5_initialized_status
    if _mt5_initialized_status:
        logger.info("Shutting down MT5 connection.")
        mt5.shutdown()
        _mt5_initialized_status = False

def get_account_info() -> dict:
    """Retrieves and returns the current MT5 account information as a dictionary."""
    if not _mt5_initialized_status:
        logger.warning("MT5 not initialized. Cannot retrieve account info.")
        return {}
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to get account info. Error: {mt5.last_error()}")
        return {}
    
    return account_info._asdict()

def get_current_open_positions(symbol: str = None) -> pd.DataFrame:
    """
    Retrieves currently open positions, optionally filtered by symbol.
    Returns a pandas DataFrame of positions. Returns an empty DataFrame on failure.
    """
    if not _mt5_initialized_status:
        logger.warning("MT5 not initialized. Cannot retrieve open positions.")
        return pd.DataFrame()
    
    try:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get open positions. Error: {mt5.last_error()}")
            return pd.DataFrame()
        
        if len(positions) == 0:
            return pd.DataFrame()

        return pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
    except Exception as e:
        logger.error(f"An error occurred while fetching positions: {e}", exc_info=True)
        return pd.DataFrame()


def get_latest_bars(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    """
    Retrieves the latest 'count' bars for a given symbol and timeframe.
    Returns a pandas DataFrame, or an empty one on failure.
    """
    if not _mt5_initialized_status:
        logger.warning("MT5 not initialized. Cannot get latest bars.")
        return pd.DataFrame()
    
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol} {timeframe}. Error: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"An error occurred while fetching bars for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()


def calculate_trade_volume(account_balance: float, risk_percent: float, price: float, symbol: str) -> float:
    """
    Calculates the trade volume based on risk percentage and symbol specification.
    """
    if not _mt5_initialized_status:
        logger.warning("MT5 not initialized. Cannot calculate volume.")
        return 0.0

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}. Error: {mt5.last_error()}")
        return 0.0

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step
    contract_size = symbol_info.trade_contract_size
    
    risk_amount = account_balance * risk_percent
    
    volume = (account_balance * risk_percent) / (contract_size * price)
    
    if lot_step > 0:
        volume = np.floor(volume / lot_step) * lot_step
    
    volume = max(min_lot, min(volume, max_lot))
    
    logger.info(f"Calculated trade volume: {volume:.2f} lots (Risk Amount: ${risk_amount:.2f})")
    return round(volume, 2)


def send_trade_request(symbol: str, trade_type: int, volume: float, price: float, sl: float, tp: float, deviation: int = 20) -> mt5.TradeRequest:
    """Sends a trade request to MT5 and returns the result."""
    if not _mt5_initialized_status:
        logger.error("MT5 not initialized. Cannot send trade request.")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": trade_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": deviation,
        "magic": 202401,
        "comment": "Python Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(f"Sending trade request: {request}")
    result = mt5.order_send(request)

    if result is None:
        logger.error(f"order_send failed, error code: {mt5.last_error()}")
        return None
        
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order send failed. Retcode: {result.retcode} - {result.comment}")

    return result

def close_position(position_ticket: int, volume: float, deviation: int = 20):
    """Closes an open position by its ticket."""
    if not _mt5_initialized_status:
        logger.error("MT5 not initialized. Cannot close position.")
        return None

    positions = mt5.positions_get(ticket=position_ticket)
    if not positions:
        logger.error(f"Cannot close position. Ticket #{position_ticket} not found.")
        return None
    
    position = positions[0]
    symbol = position.symbol
    
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price_info = mt5.symbol_info_tick(symbol)
    if price_info is None:
        logger.error(f"Could not get price info for {symbol} to close position.")
        return None
    
    close_price = price_info.bid if close_type == mt5.ORDER_TYPE_SELL else price_info.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position_ticket,
        "symbol": symbol,
        "volume": float(volume),
        "type": close_type,
        "price": close_price,
        "deviation": deviation,
        "magic": 202402,
        "comment": "Python Bot Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    logger.info(f"Sending close request for ticket #{position_ticket}: {request}")
    result = mt5.order_send(request)
    
    if result is None:
        logger.error(f"close_position failed, error code: {mt5.last_error()}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Position #{position_ticket} successfully closed.")
    else:
        logger.error(f"Failed to close position #{position_ticket}. Retcode: {result.retcode} - {result.comment}")
        
    return result

def modify_position(position_ticket: int, symbol: str, sl: float, tp: float, deviation: int = 20) -> mt5.TradeRequest:
    """
    Modifies the SL/TP of an open position.
    """
    if not _mt5_initialized_status:
        logger.error("MT5 not initialized. Cannot modify position.")
        return None

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_ticket,
        "symbol": symbol,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": deviation,
    }

    result = mt5.order_send(request)
    if result is None:
        logger.error(f"Modify position failed: {mt5.last_error()}")
        return None
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Modify position failed. Retcode: {result.retcode}, Comment: {result.comment}")
        return None
    else:
        logger.info(f"Position #{position_ticket} successfully modified. Result: {result}")
        return result
