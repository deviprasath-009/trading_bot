import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class TimeframePredictor:
    """
    Manages data fetching, feature calculation, scaling, and prediction
    for a single MetaTrader5 timeframe.
    """
    def __init__(self, symbol: str, mt5_timeframe: int, tf_string: str,
                 scaler, model, bars_to_fetch: int):
        self.symbol = symbol
        self.mt5_timeframe = mt5_timeframe
        self.tf_string = tf_string
        self.scaler = scaler
        self.model = model
        self.bars_to_fetch = bars_to_fetch
        self.fetch_bars = self._default_fetch_bars
        
        self.expected_feature_columns = [
            "open", "high", "low", "close", "tick_volume", "spread", "realvolume",
            "hour", "dayofweek", "sma_10", "ema_20", "sma_50", "ema_100", "rsi",
            "macd", "macd_h", "macd_s", "bbl", "bbm", "bbu", "bbb", "bbp", "atr",
            "stoch_k", "stoch_d", "ichimoku_ts", "ichimoku_ks",
            "close_lag1", "close_lag2", "volume_lag1", "volume_lag2"
        ]
        logger.info(f"TimeframePredictor initialized for {self.tf_string}")

    def _default_fetch_bars(self, symbol: str, timeframe: int, num_bars: int) -> pd.DataFrame:
        """A placeholder fetch function. This should be replaced by a live data fetcher."""
        logger.warning("Using default (placeholder) data fetcher. This should be overridden.")
        return pd.DataFrame()

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a comprehensive set of technical analysis features.
        """
        if df.empty:
            logger.warning(f"Input DataFrame is empty for {self.tf_string}. Skipping feature calculation.")
            return df

        df.columns = df.columns.str.lower()
        
        if 'real_volume' in df.columns:
            df.rename(columns={'real_volume': 'realvolume'}, inplace=True)
        
        if 'realvolume' not in df.columns and 'tick_volume' in df.columns:
            df['realvolume'] = df['tick_volume']

        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['close_lag1'] = df['close'].shift(1)
        df['close_lag2'] = df['close'].shift(2)
        df['volume_lag1'] = df['tick_volume'].shift(1)
        df['volume_lag2'] = df['tick_volume'].shift(2)

        try:
            df.ta.sma(length=10, append=True, col_names=('sma_10',))
            df.ta.ema(length=20, append=True, col_names=('ema_20',))
            df.ta.sma(length=50, append=True, col_names=('sma_50',))
            df.ta.ema(length=100, append=True, col_names=('ema_100',))
            df.ta.rsi(length=14, append=True, col_names=('rsi',))
            df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('macd', 'macd_h', 'macd_s'))
            df.ta.bbands(length=20, std=2.0, append=True, col_names=('bbl', 'bbm', 'bbu', 'bbb', 'bbp'))
            df.ta.atr(length=14, append=True, col_names=('atr',))
            df.ta.stoch(k=14, d=3, smooth_k=3, append=True, col_names=('stoch_k', 'stoch_d'))
            
            ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'])
            if ichimoku_df is not None:
                df['ichimoku_ts'] = ichimoku_df.get('ITS_9')
                df['ichimoku_ks'] = ichimoku_df.get('IKS_26')
        except Exception as e:
            logger.error(f"Error calculating TA features for {self.tf_string}: {e}", exc_info=True)
            for col in self.expected_feature_columns:
                if col not in df.columns:
                    df[col] = np.nan
        return df

    def _get_latest_feature_row(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the final feature row for prediction.
        """
        if df_features.empty:
            return pd.DataFrame()

        df_aligned = df_features.reindex(columns=self.expected_feature_columns, fill_value=np.nan)
        latest_row = df_aligned.tail(1)
        
        if latest_row.isnull().values.any():
            nan_cols = latest_row.columns[latest_row.isna().any()].tolist()
            logger.warning(f"NaN values found in latest feature row for {self.tf_string} in columns: {nan_cols}. This can happen with insufficient historical data.")
            return pd.DataFrame()
            
        return latest_row

    def predict(self) -> dict:
        """
        The main prediction pipeline for this timeframe.
        """
        error_response = {
            "direction": "ERROR", "up_proba": 0.0, "down_proba": 0.0,
            "last_bar_time": None, "status": "Prediction failed", "features": None
        }
        
        try:
            df = self.fetch_bars(self.symbol, self.mt5_timeframe, self.bars_to_fetch)
            if df.empty or len(df) < 50:
                error_response['status'] = f"Insufficient data fetched for {self.tf_string} ({len(df)} bars)."
                logger.error(error_response['status'])
                return error_response
            
            last_bar_time = df.index[-1]
            error_response['last_bar_time'] = last_bar_time

            df_features = self._calculate_features(df.copy())
            if df_features.empty:
                error_response['status'] = "Feature calculation resulted in an empty DataFrame."
                return error_response

            latest_features = self._get_latest_feature_row(df_features)
            if latest_features.empty:
                error_response['status'] = "Latest feature row is invalid or contains NaNs."
                return error_response

            scaled_features = self.scaler.transform(latest_features)
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            up_proba = float(probabilities[1])
            down_proba = float(probabilities[0])
            prediction_direction = "UP" if up_proba > down_proba else "DOWN"

            logger.info(f"Prediction for {self.tf_string}: {prediction_direction} (Up: {up_proba:.2%}, Down: {down_proba:.2%})")
            
            return {
                "direction": prediction_direction,
                "up_proba": up_proba,
                "down_proba": down_proba,
                "last_bar_time": last_bar_time,
                "status": "OK",
                "features": latest_features.to_dict('records')[0]
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred in predict() for {self.tf_string}: {e}", exc_info=True)
            error_response['status'] = f"Unexpected error: {e}"
            return error_response
