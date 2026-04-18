import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

DEMAND_PROFILE_24H = [
    0.6, 0.55, 0.5, 0.5, 0.55, 0.65, 0.8, 0.9, 0.95, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65
]
SOLAR_PROFILE_24H = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,0.0
]
WIND_PROFILE_24H = [0.5] * 24

class StochasticProfile:
    def __init__(self, base_profile: list[float], noise_volatility: float, rng: np.random.Generator):
        """
        Manages stochastic generation for a given variable (wind, solar, demand).
        """
        self.base_profile = np.array(base_profile)
        self.noise_volatility = noise_volatility
        self.rng = rng

    def get_actual_and_forecast(self, current_hour: int, horizon: int = 12) -> tuple[float, np.ndarray]:
        """
        Returns the actual value at time T, and noisy forecasts from T+1 to T+horizon.
        """
        # Get hour indices (modulo 24 to loop over multiple days)
        hours = [(current_hour + i) % 24 for i in range(horizon + 1)]
        trend = self.base_profile[hours]
        
        # Uncertainty multiplier: Error grows with the forecast horizon
        uncertainty_growth = np.linspace(0.5, 3.0, horizon + 1)
        
        # Standard deviation of the noise is proportional to the trend and horizon
        noise_std = self.noise_volatility * trend * uncertainty_growth
        
        # Draw noise from a normal distribution
        noise = self.rng.normal(loc=0.0, scale=noise_std)
        
        # Apply noise and prevent negative values
        values = np.clip(trend + noise, 0.0, None)
        
        actual_value = values[0]
        forecast = values[1:]
        
        return actual_value, forecast


def _build_output_dict(
    demand_values: np.ndarray,
    solar_values: np.ndarray,
    wind_values: np.ndarray,
) -> dict:
    """Formats series where index 0 is actual and 1..N are forecasts into the shared contract."""
    return {
        "actual": {
            "demand": float(demand_values[0]),
            "solar": float(solar_values[0]),
            "wind": float(wind_values[0]),
        },
        "forecast": {
            "demand": np.asarray(demand_values[1:], dtype=float),
            "solar": np.asarray(solar_values[1:], dtype=float),
            "wind": np.asarray(wind_values[1:], dtype=float),
        },
    }

class StochasticEngine:
    def __init__(self, seed: int | None = None):
        """Core engine centralizing all grid uncertainties."""
        self.rng = np.random.default_rng(seed)

        # Initialize profiles with their base volatility levels
        self.demand = StochasticProfile(DEMAND_PROFILE_24H, noise_volatility=0.05, rng=self.rng)
        self.solar = StochasticProfile(SOLAR_PROFILE_24H, noise_volatility=0.15, rng=self.rng)
        self.wind = StochasticProfile(WIND_PROFILE_24H, noise_volatility=0.25, rng=self.rng)

    def step_weather_and_demand(self, current_hour: int, horizon: int = 12) -> dict:
        """Generates actuals and forecasts for the current turn."""
        actual_demand, forecast_demand = self.demand.get_actual_and_forecast(current_hour, horizon)
        actual_solar, forecast_solar = self.solar.get_actual_and_forecast(current_hour, horizon)
        actual_wind, forecast_wind = self.wind.get_actual_and_forecast(current_hour, horizon)

        demand_values = np.concatenate(([actual_demand], forecast_demand))
        solar_values = np.concatenate(([actual_solar], forecast_solar))
        wind_values = np.concatenate(([actual_wind], forecast_wind))
        return _build_output_dict(demand_values, solar_values, wind_values)

class OpenMeteoEngine:
    """
    API-based drop-in replacement for StochasticEngine.

    It exposes the same main method:
        step_weather_and_demand(current_hour, horizon=12)

    and returns the exact same output schema as StochasticEngine.
    """

    def __init__(
        self,
        latitude: float = 48.8566,
        longitude: float = 2.3522,
        timezone: str = "auto",
        demand_profile: list[float] | None = None,
        request_timeout_seconds: int = 10,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.request_timeout_seconds = request_timeout_seconds
        self.demand_profile = np.array(demand_profile or DEMAND_PROFILE_24H, dtype=float)

    def step_weather_and_demand(self, current_hour: int, horizon: int = 12) -> dict:
        """
        Open-Meteo-backed forecast.

        Output format:
        {
            "actual": {"demand": float, "solar": float, "wind": float},
            "forecast": {"demand": np.ndarray, "solar": np.ndarray, "wind": np.ndarray}
        }
        """
        api_data = self._fetch_open_meteo_hourly(self.latitude, self.longitude, self.timezone)
        hourly = api_data["hourly"]

        times = hourly["time"]
        wind_speed = np.array(hourly["wind_speed_10m"], dtype=float)
        shortwave_radiation = np.array(hourly["shortwave_radiation"], dtype=float)
        temperature = np.array(hourly["temperature_2m"], dtype=float)

        if len(times) == 0:
            raise ValueError("Open-Meteo returned empty hourly forecast.")

        hours_of_day = np.array([datetime.fromisoformat(t).hour for t in times], dtype=int)
        matching_indices = np.where(hours_of_day == (current_hour % 24))[0]
        start_idx = int(matching_indices[0]) if len(matching_indices) > 0 else 0

        indices = [(start_idx + i) % len(times) for i in range(horizon + 1)]
        selected_hours = hours_of_day[indices]

        # Normalize weather magnitudes to game scale [0, 1].
        solar_values = np.clip(shortwave_radiation[indices] / 1000.0, 0.0, 1.0)
        wind_values = np.clip(wind_speed[indices] / 20.0, 0.0, 1.0)

        # Keep demand in same scale profile, adjusted by thermal comfort deviation.
        base_demand = self.demand_profile[selected_hours]
        comfort_temp = 20.0
        weather_load = 1.0 + 0.015 * np.abs(temperature[indices] - comfort_temp)
        demand_values = np.clip(base_demand * weather_load, 0.0, None)

        return _build_output_dict(demand_values, solar_values, wind_values)

    def _fetch_open_meteo_hourly(self, latitude: float, longitude: float, timezone: str = "auto") -> dict:
        """Fetches hourly weather data from Open-Meteo used by the API-based forecast."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "forecast_days": 2,
            "timezone": timezone,
            "wind_speed_unit": "ms",
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"

        with urlopen(url, timeout=self.request_timeout_seconds) as response:
            if response.status != 200:
                raise RuntimeError(f"Open-Meteo API request failed with status {response.status}.")
            payload = response.read().decode("utf-8")
            return json.loads(payload)


def _print_output_summary(title: str, formatted_output: dict):
    """Prints a compact summary useful for local smoke tests."""
    print(f"\n{title}")
    print("actual:", formatted_output["actual"])
    for key in ["demand", "solar", "wind"]:
        arr = np.asarray(formatted_output["forecast"][key], dtype=float)
        print(f"forecast[{key}] len={len(arr)} first3={arr[:3].round(3).tolist()}")


if __name__ == "__main__":
    # In-place test: Open-Meteo engine only.
    latitude = 48.8566
    longitude = 2.3522
    timezone = "auto"
    current_hour = datetime.now().hour
    horizon = 12

    engine = OpenMeteoEngine(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )

    output = engine.step_weather_and_demand(current_hour=current_hour, horizon=horizon)
    _print_output_summary(
        f"Engine=openmeteo hour={current_hour} horizon={horizon} lat={latitude} lon={longitude}",
        output,
    )

