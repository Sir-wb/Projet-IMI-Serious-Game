import gymnasium as gym
import pygame
import numpy as np

# Adjust imports based on your folder structure
from game.smart_grid_env import SmartGridEnv
from game.human_ui import HumanUI

def play_human():
    """Main loop to run the Smart Grid environment in human-playable mode."""
    print("Initializing Smart Grid Simulator - Control Room...")

    env = SmartGridEnv(render_mode="human")
    ui = HumanUI(screen_width=1280, screen_height=720)

    obs, info = env.reset()
    
    plants_info = [
        {
            "name": p.name, 
            "type": p.plant_type, 
            "p_max": p.p_max, 
            "ramp_rate": p.ramp_rate
        } 
        for p in env.grid.plants
    ]
    ui.setup_sliders(plants_info)

    terminated = False
    truncated = False
    current_score = 0.0

    while not (terminated or truncated):
        ui.handle_events()

        dynamic_plants_state = [{"current_power": p.current_power} for p in env.grid.plants]
        ui.update_plants_state(dynamic_plants_state)

        ui.render_frame(env.current_step, info)

        if ui.turn_validated:
            action_list = ui.get_current_actions()
            action = np.array(action_list, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            current_score += reward

            ui.log_decision(env.current_step, action_list, current_score, info.get("is_blackout", False))
            print(f"Turn: {env.current_step}/24 | Total Gen: {info.get('total_generation', 0):.0f} MW | Reward: {reward:.0f}")

    print(f"Episode finished! Final Score: {current_score:.0f}")
    ui.show_final_score(current_score)
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_human()