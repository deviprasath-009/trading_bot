import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Aggregates predictions from multiple timeframes and incorporates news sentiment
    to determine a final, consolidated trade signal.
    """
    def __init__(self, config: dict, news_analyzer):
        """
        Initializes the StrategyManager with configuration parameters.
        """
        self.news_analyzer = news_analyzer
        self.config = config
        
        # --- FIX: Directly use the boolean value from the config dictionary ---
        # The config dictionary passed from main_bot already contains the correct data type.
        self.enable_news_sentiment = self.config.get('enable_news_sentiment', False)
        
        # Convert other config values to the correct types
        self.min_confidence = float(self.config.get('min_confidence', 0.55))
        
        self.buy_thresholds = {
            'TIMEFRAME_H4': float(self.config.get('aggregate_buy_threshold_h4', 0.7)),
            'TIMEFRAME_H1': float(self.config.get('aggregate_buy_threshold_h1', 0.7)),
            'TIMEFRAME_M30': float(self.config.get('aggregate_buy_threshold_m30', 0.6)),
            'TIMEFRAME_M15': float(self.config.get('aggregate_buy_threshold_m15', 0.3)),
            'TIMEFRAME_M5': float(self.config.get('aggregate_buy_threshold_m5', 0.3)),
        }
        self.sell_thresholds = {
            'TIMEFRAME_H4': float(self.config.get('aggregate_sell_threshold_h4', 0.7)),
            'TIMEFRAME_H1': float(self.config.get('aggregate_sell_threshold_h1', 0.7)),
            'TIMEFRAME_M30': float(self.config.get('aggregate_sell_threshold_m30', 0.6)),
            'TIMEFRAME_M15': float(self.config.get('aggregate_sell_threshold_m15', 0.3)),
            'TIMEFRAME_M5': float(self.config.get('aggregate_sell_threshold_m5', 0.3)),
        }
        
        logger.info(f"StrategyManager initialized. News Sentiment Enabled: {self.enable_news_sentiment}")

    def determine_trade_signal(self, all_predictions: dict) -> tuple[str, str]:
        """
        Analyzes predictions from all timeframes to generate a single trade signal.
        """
        buy_score = 0
        sell_score = 0
        
        valid_tf_predictions = 0
        for tf_str, pred in all_predictions.items():
            if pred.get("status") != "OK":
                logger.warning(f"Skipping prediction for {tf_str} due to status: {pred.get('status')}")
                continue
            
            valid_tf_predictions += 1
            direction = pred.get("direction")
            up_proba = pred.get("up_proba", 0.0)
            down_proba = pred.get("down_proba", 0.0)

            if direction == "UP" and up_proba >= self.buy_thresholds.get(tf_str, self.min_confidence):
                buy_score += 1
            
            elif direction == "DOWN" and down_proba >= self.sell_thresholds.get(tf_str, self.min_confidence):
                sell_score += 1
        
        if valid_tf_predictions == 0:
            return "NONE", "No valid predictions available from any timeframe."

        sentiment_message = "News sentiment disabled."
        if self.enable_news_sentiment:
            sentiment_score = self.news_analyzer.get_sentiment(datetime.utcnow())
            sentiment_message = f"News sentiment: {sentiment_score:.2f}."
            if sentiment_score > float(self.config.get('news_sentiment_buy_boost_threshold', 0.5)):
                buy_score += 1
            elif sentiment_score < float(self.config.get('news_sentiment_sell_boost_threshold', -0.5)):
                sell_score += 1

        total_factors = valid_tf_predictions + (1 if self.enable_news_sentiment else 0)
        required_score = total_factors / 2.0
        
        if buy_score > required_score and buy_score > sell_score:
            message = f"BUY signal confirmed. Score: {buy_score}/{total_factors}. {sentiment_message}"
            return "BUY", message
        
        elif sell_score > required_score and sell_score > buy_score:
            message = f"SELL signal confirmed. Score: {sell_score}/{total_factors}. {sentiment_message}"
            return "SELL", message
            
        else:
            message = f"No consensus. Scores (Buy/Sell): {buy_score}/{sell_score} out of {total_factors} factors. {sentiment_message}"
            return "NONE", message

