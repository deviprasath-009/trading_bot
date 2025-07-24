import logging
import requests
import html
import time

logger = logging.getLogger(__name__)

class TelegramHandler:
    """
    A robust, simplified class to send messages to a Telegram chat.
    This version uses direct HTTP requests to avoid asyncio complexities.
    """
    def __init__(self, bot_token: str, chat_id: str, message_level: str = "INFO"):
        """
        Initializes the Telegram handler.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        if not self.bot_token or not self.chat_id or "YOUR_TELEGRAM" in self.bot_token:
            self.is_configured = False
            logger.warning("Telegram bot_token or chat_id is not configured. Notifications will be disabled.")
        else:
            self.is_configured = True
            logger.info("TelegramHandler initialized and configured.")

        self.message_level = getattr(logging, message_level.upper(), logging.INFO)

    def send_message(self, message: str, level: str = 'INFO'):
        """
        Sends a message to the configured Telegram chat ID.
        """
        if not self.is_configured:
            return

        level_int = getattr(logging, level.upper(), logging.INFO)
        if level_int < self.message_level:
            return

        if level.upper() == 'WARNING':
            message = f"⚠️ WARNING: {message}"
        elif level.upper() in ['ERROR', 'CRITICAL']:
            message = f"❌ ERROR: {message}"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": html.escape(message),
            "parse_mode": "HTML"
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                logger.debug(f"Successfully sent Telegram message: {message}")
                return
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send Telegram message (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(2)
        
        logger.error("Gave up sending Telegram message after multiple retries.")
