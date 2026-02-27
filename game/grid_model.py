class PowerPlant:
    def __init__(self, name: str, plant_type: str, p_max: float, p_min: float, cost_per_mw: float, co2_per_mw: float, ramp_rate: float):
        """
        Represents an electrical power source.
        1 turn = 1 hour of production.
        """
        self.name = name
        self.plant_type = plant_type # e.g., 'nuclear', 'gas', 'wind', 'solar'
        self.p_max = p_max
        self.p_min = p_min
        self.cost_per_mw = cost_per_mw
        self.co2_per_mw = co2_per_mw
        self.ramp_rate = ramp_rate   # Max MW variation per hour
        
        self.current_power = 0.0
        self.is_on = False

    def step_power(self, target_power: float) -> tuple[float, float]:
        """
        Attempts to reach the target power while respecting physical constraints.
        Returns the financial and ecological costs generated during this hour.
        """
        # Constraint 1: Absolute limits
        if target_power > 0 and target_power < self.p_min:
            target_power = 0.0  # Turn off if below minimum technical threshold
        target_power = min(target_power, self.p_max)
        
        # Constraint 2: Inertia / Ramp Rate
        delta = target_power - self.current_power
        if abs(delta) > self.ramp_rate:
            # Clamp the variation to the plant's physical limits
            delta = self.ramp_rate if delta > 0 else -self.ramp_rate
            
        self.current_power += delta
        self.is_on = self.current_power > 0
        
        # Calculate hourly costs
        hourly_cost = self.current_power * self.cost_per_mw
        hourly_co2 = self.current_power * self.co2_per_mw
        
        return hourly_cost, hourly_co2

class Consumer:
    def __init__(self, name: str, base_load: float):
        """Represents a city or industrial area."""
        self.name = name
        self.base_load = base_load
        self.current_load = base_load

    def update_load(self, new_load: float):
        """Updates the current demand (called by the stochastic engine)."""
        self.current_load = new_load

class Grid:
    def __init__(self):
        """The network grid that balances supply and demand at each step."""
        self.plants: list[PowerPlant] = []
        self.consumers: list[Consumer] = []

    def add_plant(self, plant: PowerPlant):
        self.plants.append(plant)

    def add_consumer(self, consumer: Consumer):
        self.consumers.append(consumer)

    def step_balance(self) -> dict:
        """
        Calculates the grid balance at the end of the hour.
        Returns a dictionary with essential metrics for reward calculation.
        """
        total_prod = sum(p.current_power for p in self.plants)
        total_demand = sum(c.current_load for c in self.consumers)
        
        delta = total_prod - total_demand
        
        # If delta < 0, we lack energy (Partial or total blackout)
        is_blackout = delta < 0
        unmet_demand = abs(delta) if is_blackout else 0.0
        wasted_energy = delta if delta > 0 else 0.0
        
        return {
            "total_production": total_prod,
            "total_demand": total_demand,
            "delta": delta,
            "is_blackout": is_blackout,
            "unmet_demand": unmet_demand,
            "wasted_energy": wasted_energy
        }