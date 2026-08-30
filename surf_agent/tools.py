import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def fetch_stormglass_data(lat: float, lng: float, api_key: str) -> dict:
    """
    Fetches marine forecast data from the Stormglass.io weather/point endpoint.
    
    Parameters:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        api_key (str): Stormglass API key.
        
    Returns:
        dict: The raw JSON response from the Stormglass API.
    """
    if not api_key or not str(api_key).strip():
        raise ValueError("Stormglass API key is missing. Please set STORMGLASS_KEY in your environment.")
        
    url = "https://api.stormglass.io/v2/weather/point"
    params = {
        "lat": lat,
        "lng": lng,
        "params": "waveHeight,windSpeed,windDirection,wavePeriod"
    }
    headers = {
        "Authorization": str(api_key).strip()
    }
    
    logger.info(f"Fetching Stormglass data for lat: {lat}, lng: {lng}")
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code != 200:
        logger.error(f"Stormglass API request failed with status code {response.status_code}: {response.text}")
        response.raise_for_status()
        
    return response.json()

def send_telegram_message(token: str, chat_id: str, message: str) -> dict:
    """
    Sends a message via the Telegram Bot API to a specified chat/channel.
    
    Parameters:
        token (str): Telegram Bot token.
        chat_id (str): Telegram chat or channel ID.
        message (str): The text message to send.
        
    Returns:
        dict: The raw JSON response from the Telegram Bot API.
    """
    token_str = str(token).strip() if token else ""
    chat_id_str = str(chat_id).strip() if chat_id else ""
    
    if not token_str:
        raise ValueError("Telegram Bot Token is missing. Please set TG_TOKEN in your environment.")
    if not chat_id_str:
        raise ValueError("Telegram Chat ID is missing. Please set TG_CHAT_ID in your environment.")
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    logger.info(f"Sending Telegram message to chat: {chat_id}")
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        logger.error(f"Telegram Bot API request failed with status code {response.status_code}: {response.text}")
        response.raise_for_status()
        
    return response.json()
