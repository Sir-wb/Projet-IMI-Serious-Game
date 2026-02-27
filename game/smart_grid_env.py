import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .grid_model import Grid, PowerPlant, Consumer
from .stochastic_engine import StochasticEngine

class SmartGridEnv(gym.Env):
    """Gymnasium environment for the Smart Grid serious game."""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        self.num_controllable_plants = 3 
        self.forecast_horizon = 12
        self.max_steps = 24 
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_controllable_plants,), dtype=np.float32
        )

        # Observation size: 3 (Controllable) + 2 (Renewables) + 3 (Current weather) + 36 (Forecasts) + 1 (Step)
        obs_size = self.num_controllable_plants + 2 + 3 + (3 * self.forecast_horizon) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment at the beginning of a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        
        self.engine = StochasticEngine(seed=seed)
        self.grid = Grid() 
        
        # 1. Controllable Plants (Indices 0, 1, 2)
        self.grid.add_plant(PowerPlant("Gas Plant", "gas", p_max=100.0, p_min=0.0, cost_per_mw=150.0, co2_per_mw=0.8, ramp_rate=100.0))
        self.grid.add_plant(PowerPlant("Coal Plant", "coal", p_max=150.0, p_min=0.0, cost_per_mw=50.0, co2_per_mw=1.2, ramp_rate=50.0))
        self.grid.add_plant(PowerPlant("Nuclear Plant", "nuclear", p_max=300.0, p_min=150.0, cost_per_mw=20.0, co2_per_mw=0.01, ramp_rate=20.0))
        
        # 2. Renewable Plants (Indices 3, 4) - Driven by weather, huge ramp rate to follow weather instantly
        self.solar_capacity = 150.0
        self.wind_capacity = 100.0
        self.grid.add_plant(PowerPlant("Solar Farm", "solar", p_max=self.solar_capacity, p_min=0.0, cost_per_mw=0.0, co2_per_mw=0.0, ramp_rate=999.0))
        self.grid.add_plant(PowerPlant("Wind Farm", "wind", p_max=self.wind_capacity, p_min=0.0, cost_per_mw=0.0, co2_per_mw=0.0, ramp_rate=999.0))
        
        self.city_base_load = 400.0
        self.grid.add_consumer(Consumer("Main City", base_load=self.city_base_load))
        
        # Initialize turn 0 states
        self.current_weather = self.engine.step_weather_and_demand(self.current_step, self.forecast_horizon)
        self._update_stochastic_elements()
        
        return self._get_obs(), self._get_info()

    def _update_stochastic_elements(self):
        """Applies current weather to renewables and demand."""
        actual_demand = self.current_weather["actual"]["demand"] * self.city_base_load
        self.grid.consumers[0].update_load(actual_demand)
        
        actual_solar = self.current_weather["actual"]["solar"] * self.solar_capacity
        actual_wind = self.current_weather["actual"]["wind"] * self.wind_capacity
        
        self.grid.plants[3].step_power(actual_solar)
        self.grid.plants[4].step_power(actual_wind)

    def step(self, action):
        """Advances the simulation by one hour."""
        self.current_step += 1
        total_financial_cost = 0.0
        total_co2_emissions = 0.0
        
        # Apply actions ONLY to controllable plants
        for i in range(self.num_controllable_plants):
            plant = self.grid.plants[i]
            target_power = action[i] * plant.p_max
            cost, co2 = plant.step_power(target_power)
            total_financial_cost += cost
            total_co2_emissions += co2
            
        # Update weather and apply it to renewables/consumers
        self.current_weather = self.engine.step_weather_and_demand(self.current_step, self.forecast_horizon)
        self._update_stochastic_elements()
        
        balance = self.grid.step_balance()
        
        reward = -((0.1 * total_financial_cost) + (0.1 * total_co2_emissions) + (0.5 * balance["wasted_energy"]) + (100.0 * balance["unmet_demand"]))
        
        info = self._get_info()
        info["is_blackout"] = balance["is_blackout"]
        
        return self._get_obs(), float(reward), False, self.current_step >= self.max_steps, info

    def _get_obs(self):
        """Constructs the flat observation vector."""
        obs = [p.current_power / p.p_max for p in self.grid.plants]
        obs.extend([self.current_weather["actual"]["wind"], self.current_weather["actual"]["solar"], self.current_weather["actual"]["demand"]])
        obs.extend(self.current_weather["forecast"]["wind"])
        obs.extend(self.current_weather["forecast"]["solar"])
        obs.extend(self.current_weather["forecast"]["demand"])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Packages rich data for the UI rendering, including renewable forecasts and grid status."""
        return {
            "forecast_data": {
                "demand_forecast": (self.current_weather["forecast"]["demand"] * self.city_base_load).tolist(),
                "solar_forecast": (self.current_weather["forecast"]["solar"] * self.solar_capacity).tolist(),
                "wind_forecast": (self.current_weather["forecast"]["wind"] * self.wind_capacity).tolist()
            },
            "total_generation": sum(p.current_power for p in self.grid.plants),
            "current_demand": self.grid.consumers[0].current_load,
            "solar_power": self.grid.plants[3].current_power,
            "wind_power": self.grid.plants[4].current_power,
            "plants_status": [
                {"name": p.name, "type": p.plant_type, "power": p.current_power, "p_max": p.p_max} 
                for p in self.grid.plants
            ]
        }