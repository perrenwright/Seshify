import os
import logging
from typing import TypedDict, Any, Optional
from pydantic import BaseModel, Field
from tools import fetch_stormglass_data, send_telegram_message

logger = logging.getLogger(__name__)

# State Definition
class AgentState(TypedDict):
    forecast_data: Optional[dict]
    decision: Optional[str]
    reasoning: Optional[str]

# Pydantic schema for LangChain structured output
class SurfDecision(BaseModel):
    decision: str = Field(description="Decision: either 'GO' or 'STAY' based on the surf conditions.")
    reasoning: str = Field(description="A friendly, user-facing summary of the conditions and the reasoning behind the decision.")

def extract_metric(hour_data: dict, key: str) -> float:
    """
    Safely extracts a numeric metric from a Stormglass hour data block.
    Stormglass formats parameters as nested dicts of sources (e.g. {"sg": 1.2, "noaa": 1.4}).
    This helper retrieves the 'sg' source value or falls back to any available source.
    """
    if key not in hour_data:
        return 0.0
    val = hour_data[key]
    if isinstance(val, dict):
        if "sg" in val:
            return float(val["sg"])
        elif val:
            # Fallback to the first available source
            return float(next(iter(val.values())))
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def get_llm():
    """
    Returns a configured LangChain Chat Model.
    Dynamically selects between ChatGoogleGenerativeAI and ChatOpenAI based on environment variables.
    """
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Use gemini-2.5-flash as the state-of-the-art default
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError("No LLM API key found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY.")

def fetch_weather(state: AgentState) -> dict:
    """
    Fetcher Node: Queries the Stormglass.io weather/point endpoint.
    """
    lat = float(os.getenv("SURF_LAT", "33.6839"))
    lng = float(os.getenv("SURF_LNG", "-118.0122"))
    api_key = os.getenv("STORMGLASS_KEY")
    
    if not api_key:
        logger.warning("STORMGLASS_KEY is not set. Fetcher will fail unless mocked.")
        
    try:
        forecast = fetch_stormglass_data(lat, lng, api_key)
        return {"forecast_data": forecast}
    except Exception as e:
        logger.error(f"Failed to fetch weather: {e}")
        raise e

def judge_conditions(state: AgentState) -> dict:
    """
    Judge Node:
    1. Hard Filter (Deterministic): If wavePeriod < 5 OR windSpeed > 10, sets decision = 'STAY'.
    2. LLM Reasoner: If filter passes, uses LLM with structured output to decide and draft reasoning.
    """
    forecast = state.get("forecast_data")
    if not forecast or "hours" not in forecast or not forecast["hours"]:
        return {
            "decision": "STAY",
            "reasoning": "No valid hourly forecast data available to judge."
        }
    
    # Analyze the closest forecast hour
    first_hour = forecast["hours"][0]
    
    wave_period = extract_metric(first_hour, "wavePeriod")
    wind_speed = extract_metric(first_hour, "windSpeed")
    wave_height = extract_metric(first_hour, "waveHeight")
    wind_direction = extract_metric(first_hour, "windDirection")
    
    logger.info(f"Judging nearest forecast conditions: wavePeriod={wave_period}s, windSpeed={wind_speed}m/s, waveHeight={wave_height}m, windDirection={wind_direction}°")
    
    # 1. Hard Filter Check
    if wave_period < 5 or wind_speed > 10:
        reasoning = (
            f"Deterministic STAY: Wave period is too short ({wave_period}s < 5s) "
            f"or wind speed is too strong ({wind_speed} m/s > 10 m/s) for safe, clean surfing."
        )
        logger.info(f"Hard filter triggered: {reasoning}")
        return {
            "decision": "STAY",
            "reasoning": reasoning
        }
    
    # 2. LLM Reasoner Check (Filter passed)
    logger.info("Hard filter passed. Consulting LLM Reasoner for the final decision...")
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SurfDecision)
        
        prompt = (
            "You are SurfSentry, an expert surf forecasting AI agent.\n"
            "Analyze these marine forecast metrics and decide if they are suitable/favorable for surfing. "
            "If conditions are favorable (e.g. good wave height, clean winds, decent period), set decision to 'GO' "
            "and craft a highly engaging, enthusiastic surf report for the user's Telegram.\n"
            "If conditions are not favorable (e.g. wave height is flat/low, or wind is unfavorable), set decision to 'STAY' "
            "and craft a polite, brief explanation of why surfing is not recommended.\n\n"
            "METRICS TO ANALYZE:\n"
            f"- Wave Height: {wave_height} meters\n"
            f"- Wave Period: {wave_period} seconds\n"
            f"- Wind Speed: {wind_speed} m/s\n"
            f"- Wind Direction: {wind_direction} degrees\n"
        )
        
        response = structured_llm.invoke(prompt)
        logger.info(f"LLM decision: {response.decision}, reasoning length: {len(response.reasoning)}")
        
        return {
            "decision": response.decision.upper(),
            "reasoning": response.reasoning
        }
    except Exception as e:
        logger.error(f"Error during LLM reasoning node execution: {e}")
        # Soft fallback if LLM fails (e.g. key issue)
        return {
            "decision": "GO",
            "reasoning": (
                f"LLM Reasoner unavailable. The deterministic filters passed successfully!\n"
                f"Current marine metrics:\n"
                f"• Wave Height: {wave_height}m\n"
                f"• Wave Period: {wave_period}s\n"
                f"• Wind Speed: {wind_speed} m/s\n"
                f"• Wind Direction: {wind_direction}°"
            )
        }

def send_notification(state: AgentState) -> dict:
    """
    Notifier Node: Sends Telegram notification if decision == 'GO'.
    """
    decision = state.get("decision")
    reasoning = state.get("reasoning", "")
    
    if decision == "GO":
        token = os.getenv("TG_TOKEN")
        chat_id = os.getenv("TG_CHAT_ID")
        
        # Prepare the HTML message content
        message = (
            f"🏄‍♂️ <b>SURF SENTRY CHECK: GO!</b> 🏄‍♂️\n\n"
            f"{reasoning}\n\n"
            f"<i>Have a legendary session! 🌊</i>"
        )
        
        try:
            send_telegram_message(token, chat_id, message)
            logger.info("Telegram notification sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            raise e
            
    return {}
