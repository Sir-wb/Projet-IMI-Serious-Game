import pygame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import csv
import os
from datetime import datetime

# Use non-interactive backend for Matplotlib
matplotlib.use("Agg")

class Slider:
    """A UI component to control the target power of a power plant."""
    def __init__(self, x, y, w, h, index, name, plant_type, p_max, ramp_rate):
        self.rect = pygame.Rect(x, y, w, h)
        self.index = index
        self.name = name
        self.plant_type = plant_type
        self.p_max = p_max
        self.ramp_rate = ramp_rate
        
        self.current_power = 0.0
        self.value = 0.0  
        self.is_dragging = False

    def update_state(self, current_power):
        self.current_power = current_power

    def draw(self, screen, font, small_font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect, border_radius=5)
        
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, int(self.rect.w * self.value), self.rect.h)
        color = (200, 50, 50) if "gas" in self.plant_type else (50, 50, 50) if "coal" in self.plant_type else (50, 200, 50)
        pygame.draw.rect(screen, color, fill_rect, border_radius=5)
        
        pygame.draw.rect(screen, (100, 100, 100), self.rect, 2, border_radius=5)

        target_mw = self.value * self.p_max
        actual_delta = max(-self.ramp_rate, min(self.ramp_rate, target_mw - self.current_power))
        next_turn_mw = self.current_power + actual_delta

        text_name = font.render(f"{self.name} ({self.plant_type.upper()})", True, (0, 0, 0))
        stats_string = f"Current: {self.current_power:.0f} MW  |  Next Hour: {next_turn_mw:.0f} MW  |  Target: {target_mw:.0f} MW"
        text_stats = small_font.render(stats_string, True, (40, 40, 40))
        text_percent = font.render(f"{int(self.value * 100)} %", True, (0, 0, 0))
        
        screen.blit(text_name, (self.rect.x, self.rect.y - 40))
        screen.blit(text_stats, (self.rect.x, self.rect.y - 20))
        screen.blit(text_percent, (self.rect.x + self.rect.w + 15, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_dragging = True
                self._update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self._update_value(event.pos[0])

    def _update_value(self, mouse_x):
        rel_x = mouse_x - self.rect.x
        self.value = max(0.0, min(1.0, rel_x / self.rect.w))

class HumanUI:
    def __init__(self, screen_width=1280, screen_height=720):
        pygame.init()
        self.width = screen_width
        self.height = screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Smart Grid Simulator - Control Room")
        self.clock = pygame.time.Clock()
        self.sliders = []
        self.turn_validated = False
        
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 22, bold=True)
        
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/human_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, mode='w', newline='') as f: 
            csv.writer(f).writerow(["step", "actions", "score", "blackout"])
            
        # --- OPTIMIZATION: Pre-allocate Matplotlib Figure & Canvas ---
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
    def setup_sliders(self, plants_info):
        self.sliders = [Slider(50, 520 + (i*70), 400, 25, i, p['name'], p['type'], p['p_max'], p['ramp_rate']) 
                        for i, p in enumerate(plants_info[:3])] 

    def update_plants_state(self, plants_info):
        for i, plant in enumerate(plants_info[:3]): 
            self.sliders[i].update_state(plant['current_power'])

    def log_decision(self, step, actions, score, blackout):
        with open(self.log_file, mode='a', newline='') as f: 
            csv.writer(f).writerow([step, str(actions), score, blackout])

    def _render_matplotlib_to_pygame(self, forecast_data: dict, current_generation: float) -> pygame.Surface:
        """Draws the 12h forecast graph using cached figures for high performance."""
        # OPTIMIZATION: Clear the existing axes instead of creating a new figure
        self.ax.clear()
        
        x = np.arange(1, 13)
        demand_forecast = np.array(forecast_data.get("demand_forecast", [0]*12))
        solar_forecast = np.array(forecast_data.get("solar_forecast", [0]*12))
        wind_forecast = np.array(forecast_data.get("wind_forecast", [0]*12))
        
        uncertainty = demand_forecast * np.linspace(0.05, 0.40, 12)
        
        self.ax.plot(x, demand_forecast, label="Demand Forecast", color='green', linewidth=2)
        self.ax.fill_between(x, demand_forecast - uncertainty, demand_forecast + uncertainty, color='green', alpha=0.1)
        
        self.ax.plot(x, solar_forecast, label="Solar Forecast", color='#FFA500', linestyle='--', linewidth=2)
        self.ax.plot(x, wind_forecast, label="Wind Forecast", color='blue', linestyle='--', linewidth=2)
        
        self.ax.axhline(y=current_generation, color='red', linestyle='-', linewidth=2, label=f"Current Output")
        
        self.ax.set_title("12h Forecasts (Demand & Renewables)")
        self.ax.set_xlabel("Upcoming Hours")
        self.ax.set_ylabel("Power (MW)")
        self.ax.legend(loc="upper left", fontsize='small') 
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()

        # Draw on the pre-allocated canvas
        self.canvas.draw()
        surf = pygame.image.frombuffer(self.canvas.buffer_rgba(), self.canvas.get_width_height(), "RGBA")
        return surf

    def handle_events(self):
        """Processes Pygame events (mouse clicks, dragging, enter key)."""
        self.turn_validated = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: 
                self.turn_validated = True
            for slider in self.sliders: 
                slider.handle_event(event)

    def get_current_actions(self) -> list:
        return [s.value for s in self.sliders]

    def render_frame(self, step, info):
        self.screen.fill((230, 235, 240))
        
        pygame.draw.line(self.screen, (150, 150, 150), (640, 0), (640, 450), 2)
        pygame.draw.line(self.screen, (150, 150, 150), (0, 450), (1280, 450), 2)

        # --- ZONE 1: GRID & TABLE ---
        self.screen.blit(self.title_font.render("Grid Status & Totals", True, (50, 50, 50)), (20, 20))
        
        next_controllable_gen = 0.0
        for slider in self.sliders:
            target_mw = slider.value * slider.p_max
            actual_delta = max(-slider.ramp_rate, min(slider.ramp_rate, target_mw - slider.current_power))
            next_controllable_gen += (slider.current_power + actual_delta)
            
        expected_next_gen = next_controllable_gen + info.get('solar_power', 0) + info.get('wind_power', 0)

        table_rect = pygame.Rect(20, 60, 320, 130)
        pygame.draw.rect(self.screen, (255, 255, 255), table_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 100), table_rect, 2, border_radius=8)
        
        demand_color = (200, 0, 0) if info["total_generation"] < info["current_demand"] else (0, 150, 0)
        next_gen_color = (0, 100, 200) 
        
        self.screen.blit(self.font.render(f"Current Generation: {info['total_generation']:.0f} MW", True, (0, 0, 0)), (35, 75))
        self.screen.blit(self.font.render(f"Current Demand: {info['current_demand']:.0f} MW", True, demand_color), (35, 105))
        
        pygame.draw.line(self.screen, (220, 220, 220), (35, 135), (325, 135), 1)
        self.screen.blit(self.font.render(f"Expected Next Gen: {expected_next_gen:.0f} MW", True, next_gen_color), (35, 150))

       # --- NETWORK TOPOLOGY RENDERING ---
        # Confined to the bottom-right space of Zone 1 to avoid the info table
        city_pos = (540, 330)
        
        # Positions mathematically adjusted to stay within Y: 220-440 and X: 50-600
        plant_positions = [
            (120, 250),  # Gas (Top Left)
            (80, 330),   # Coal (Middle Left)
            (120, 410),  # Nuclear (Bottom Left)
            (320, 240),  # Solar (Top Center)
            (320, 420)   # Wind (Bottom Center)
        ]
        
        plant_colors = {
            "gas": (200, 80, 80), "coal": (100, 100, 100), "nuclear": (100, 200, 100),
            "solar": (255, 200, 50), "wind": (100, 180, 255)
        }

        if "plants_status" in info:
            # 1. Draw Transmission Lines first (so they render under the nodes)
            for i, p in enumerate(info["plants_status"]):
                start_pos = plant_positions[i]
                power_ratio = p["power"] / max(1.0, p["p_max"])
                
                if p["power"] > 0:
                    thickness = max(2, int(power_ratio * 12)) 
                    line_color = (255, 0, 0) if info.get("is_blackout") else (50, 200, 50)
                    pygame.draw.line(self.screen, line_color, start_pos, city_pos, thickness)
                else:
                    pygame.draw.line(self.screen, (200, 200, 200), start_pos, city_pos, 2)

            # 2. Draw City Node (Centered labels)
            pygame.draw.circle(self.screen, (200, 150, 100), city_pos, 45)
            pygame.draw.circle(self.screen, (50, 50, 50), city_pos, 45, 3)
            
            city_lbl = self.font.render("CITY", True, (255, 255, 255))
            city_rect = city_lbl.get_rect(center=city_pos)
            self.screen.blit(city_lbl, city_rect)
            
            # 3. Draw Plant Nodes and Centered Labels
            for i, p in enumerate(info["plants_status"]):
                pos = plant_positions[i]
                color = plant_colors.get(p["type"], (150, 150, 150))
                
                pygame.draw.circle(self.screen, color, pos, 25)
                pygame.draw.circle(self.screen, (50, 50, 50), pos, 25, 2)
                
                # Using get_rect(center=...) ensures perfect alignment without overlap
                name_lbl = self.small_font.render(p["name"], True, (50, 50, 50))
                name_rect = name_lbl.get_rect(center=(pos[0], pos[1] - 35))
                self.screen.blit(name_lbl, name_rect)

                mw_lbl = self.small_font.render(f"{p['power']:.0f} MW", True, (0, 0, 0))
                mw_rect = mw_lbl.get_rect(center=(pos[0], pos[1] + 35))
                self.screen.blit(mw_lbl, mw_rect)

        
        # --- ZONE 2: FORECAST GRAPH ---
        graph_surface = self._render_matplotlib_to_pygame(info.get("forecast_data", {}), info["total_generation"])
        self.screen.blit(graph_surface, (650, 20))

        # --- ZONE 3: CONTROLS & RENEWABLES ---
        self.screen.blit(self.title_font.render("Power Plants Control Panel", True, (50, 50, 50)), (20, 460))
        for slider in self.sliders: 
            slider.draw(self.screen, self.font, self.small_font)

        ren_rect = pygame.Rect(650, 500, 250, 100)
        pygame.draw.rect(self.screen, (220, 240, 220), ren_rect, border_radius=5)
        self.screen.blit(self.font.render("Weather Generation (Auto):", True, (50, 100, 50)), (660, 510))
        self.screen.blit(self.font.render(f"☀️ Solar: {info.get('solar_power', 0):.0f} MW", True, (200, 150, 0)), (660, 540))
        self.screen.blit(self.font.render(f"💨 Wind: {info.get('wind_power', 0):.0f} MW", True, (0, 100, 200)), (660, 570))

        self.screen.blit(self.title_font.render(f"Turn: {step}/24", True, (0, 0, 0)), (950, 520))
        self.screen.blit(self.font.render("[ PRESS ENTER TO VALIDATE ]", True, (100, 100, 100)), (950, 570))

        pygame.display.flip()
        self.clock.tick(30)

    def show_final_score(self, final_score):
        """Displays a dedicated screen at the end of the game with the final score."""
        self.screen.fill((230, 235, 240))
        
        game_over_text = self.title_font.render("SIMULATION COMPLETE", True, (50, 50, 50))
        score_text = self.title_font.render(f"Final Score: {final_score:.0f}", True, (200, 50, 50) if final_score < -5000 else (50, 150, 50))
        instruction_text = self.font.render("Closing in 5 seconds...", True, (100, 100, 100))
        
        self.screen.blit(game_over_text, (self.width//2 - 120, self.height//2 - 50))
        self.screen.blit(score_text, (self.width//2 - 100, self.height//2))
        self.screen.blit(instruction_text, (self.width//2 - 80, self.height//2 + 50))
        
        pygame.display.flip()
        pygame.time.wait(5000)