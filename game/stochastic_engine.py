import numpy as np

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

class StochasticEngine:
    def __init__(self, seed: int = None):
        """Core engine centralizing all grid uncertainties."""
        self.rng = np.random.default_rng(seed)
        
        # Sample 24h demand profile
        demand_profile = [0.6, 0.55, 0.5, 0.5, 0.55, 0.65, 0.8, 0.9, 0.95, 0.9, 0.85, 0.85,
                          0.85, 0.85, 0.85, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65]
        
        # Sample solar profile
        solar_profile = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0,
                         1.0, 0.9, 0.8, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Wind is more constant but highly volatile
        wind_profile = [0.5] * 24 

        # Initialize profiles with their base volatility levels
        self.demand = StochasticProfile(demand_profile, noise_volatility=0.05, rng=self.rng)
        self.solar = StochasticProfile(solar_profile, noise_volatility=0.15, rng=self.rng)
        self.wind = StochasticProfile(wind_profile, noise_volatility=0.25, rng=self.rng)

    def step_weather_and_demand(self, current_hour: int, horizon: int = 12):
        """Generates actuals and forecasts for the current turn."""
        actual_demand, forecast_demand = self.demand.get_actual_and_forecast(current_hour, horizon)
        actual_solar, forecast_solar = self.solar.get_actual_and_forecast(current_hour, horizon)
        actual_wind, forecast_wind = self.wind.get_actual_and_forecast(current_hour, horizon)
        
        return {
            "actual": {"demand": actual_demand, "solar": actual_solar, "wind": actual_wind},
            "forecast": {"demand": forecast_demand, "solar": forecast_solar, "wind": forecast_wind}
        }