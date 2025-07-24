import random
import logging
from datetime import datetime, timedelta
# import requests # Not needed for backtesting simulation
# import json # Not needed for backtesting simulation

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """
    A class to simulate news sentiment analysis for backtesting purposes.
    It does not make real API calls but generates a random sentiment score
    that updates periodically, mimicking real-world variability.
    """
    def __init__(self, config: dict):
        """
        Initializes the NewsSentimentAnalyzer for backtesting.
        Args:
            config (dict): Configuration dictionary for sentiment analysis.
                           Expected keys:
                           - 'SENTIMENT_UPDATE_INTERVAL_MINUTES': How often to update the simulated sentiment.
                           - 'SYMBOL': The trading symbol (e.g., XAUUSD).
        """
        self.config = config
        self.symbol = self.config.get('SYMBOL', 'XAUUSD')
        self.sentiment_update_interval_minutes = int(self.config.get('SENTIMENT_UPDATE_INTERVAL_MINUTES', 30))
        
        self.last_sentiment_update_time = None
        self.current_simulated_sentiment = 0.0 # Stores the simulated sentiment

        logger.info(f"NewsSentimentAnalyzer initialized for backtesting. Simulating sentiment updates every {self.sentiment_update_interval_minutes} minutes.")

    def _simulate_sentiment_generation(self) -> float:
        """
        Generates a simulated sentiment score.
        This is a simple random walk with occasional spikes.
        """
        # Simulate a sentiment that slowly drifts and occasionally has "spikes"
        drift = random.uniform(-0.05, 0.05) # Small random drift
        spike_chance = random.random()

        if spike_chance < 0.15: # 15% chance of a significant spike
            spike = random.uniform(-0.8, 0.8)
            new_sentiment = self.current_simulated_sentiment * 0.5 + spike * 0.5 # Blend with existing for smoother spikes
        else:
            new_sentiment = self.current_simulated_sentiment + drift

        # Keep sentiment within -1.0 to 1.0 range
        new_sentiment = max(-1.0, min(1.0, new_sentiment))
        
        self.current_simulated_sentiment = new_sentiment
        logger.debug(f"Simulated news sentiment generated: {self.current_simulated_sentiment:.2f}")
        return self.current_simulated_sentiment

    def get_sentiment(self, current_utc_time: datetime) -> float:
        """
        Returns the current news sentiment score.
        Updates the simulated sentiment periodically based on `sentiment_update_interval_minutes`.
        """
        if self.last_sentiment_update_time is None or \
           (current_utc_time - self.last_sentiment_update_time) >= timedelta(minutes=self.sentiment_update_interval_minutes):
            self.current_simulated_sentiment = self._simulate_sentiment_generation()
            self.last_sentiment_update_time = current_utc_time
            logger.info(f"Updated simulated news sentiment to: {self.current_simulated_sentiment:.2f} at {current_utc_time}")
        
        return self.current_simulated_sentiment

