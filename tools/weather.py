"""OpenWeatherMap tool — gracefully degrades when API key is absent."""
from __future__ import annotations

import requests
from langchain_core.tools import tool

_OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


@tool
def get_weather(location: str) -> str:
    """
    Get current weather conditions for a city.

    Returns temperature (°C), humidity, wind speed, and a short description.
    Requires OPENWEATHER_API_KEY in .env; returns a helpful message if absent.

    Args:
        location: City name, e.g. 'London' or 'New York,US' or 'Tokyo,JP'

    Returns:
        A weather summary string or an error/fallback message.
    """
    # Import here to avoid circular import at module load
    from config import settings

    if not settings.openweather_api_key:
        return (
            f"Weather tool is not configured. "
            f"Set OPENWEATHER_API_KEY in your .env file to enable weather lookups. "
            f"(Query was: '{location}')"
        )

    params = {
        "q": location,
        "appid": settings.openweather_api_key,
        "units": "metric",
    }
    try:
        resp = requests.get(_OWM_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        city = data.get("name", location)
        country = data.get("sys", {}).get("country", "")
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        description = data["weather"][0]["description"].capitalize()

        return (
            f"Weather in {city}, {country}: {description}. "
            f"Temperature: {temp}°C (feels like {feels_like}°C). "
            f"Humidity: {humidity}%. Wind: {wind_speed} m/s."
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"Location '{location}' not found. Try a different city name."
        return f"Weather API error: {exc}"
    except requests.RequestException as exc:
        return f"Network error fetching weather: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Unexpected weather response format: {exc}"
