import os
import sys
import time
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Dynamically add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(project_root))

from game.smart_grid_env import SmartGridEnv
from game.human_ui import HumanUI

def evaluate_agent():
    print("Loading trained AI and normalization statistics...")

    # FIX: Use project_root to locate the models folder, not current_dir
    models_dir = os.path.join(os.path.abspath(project_root), "models", "ppo")
    model_path = os.path.join(models_dir, "ppo_smartgrid_100k")
    norm_path = os.path.join(models_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip") or not os.path.exists(norm_path):
        print(f"Error: Files not found at {models_dir}")
        return

    # 1. Create the base environment
    env = SmartGridEnv(render_mode=None) 
    
    # 2. Wrap it with DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load the normalization stats and freeze them (CRITICAL)
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # 4. Load the trained model onto the CPU
    model = PPO.load(model_path, env=vec_env, device="cpu")

    # 5. Initialize the Pygame UI
    ui = HumanUI()
    obs = vec_env.reset()
    
    # Extract the underlying environment to access physical grid parameters
    base_env = vec_env.envs[0]
    plants_info = [
        {
            "name": p.name, 
            "type": p.plant_type, 
            "p_max": p.p_max,
            "p_min": p.p_min,
            "ramp_rate": p.ramp_rate
        } 
        for p in base_env.grid.plants
    ]
    ui.setup_sliders(plants_info)

    print("Starting AI Evaluation Phase. Watch the dashboard...")

    done = False
    running = True
    current_step = 0
    cumulative_reward = 0.0

    while running:
        # Keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:
            # Ask the AI for the best action
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment forward
            obs, rewards, dones, infos = vec_env.step(action)
            
            done = dones[0]
            info = infos[0]

            # Accumulate the REAL (unnormalized) reward
            real_reward = vec_env.get_original_reward()[0]
            cumulative_reward += real_reward

            # Inject the AI's chosen actions into the UI sliders
            raw_action = action[0] 
            for i, slider in enumerate(ui.sliders):
                slider.value = float(raw_action[i])
                slider._drag_needs_update = True
                slider.update_state(base_env.grid.plants[i].current_power)

            # Render the frame
            ui.render_frame(current_step, info)
            current_step += 1

            if done:
                print(f"\n--- EVALUATION COMPLETE ---")
                print(f"Final Cumulative Score: {cumulative_reward:.2f}")
                print("The simulation has ended. You can now close the window.")
                
                # Draw a dark overlay with the final score
                overlay = pygame.Surface((ui.width, ui.height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180)) 
                ui.screen.blit(overlay, (0, 0))
                
                font = pygame.font.SysFont("Arial", 36, bold=True)
                text_surf = font.render(f"SIMULATION COMPLETE | Final Score: {cumulative_reward:.0f}", True, (255, 255, 255))
                text_rect = text_surf.get_rect(center=(ui.width // 2, ui.height // 2))
                ui.screen.blit(text_surf, text_rect)
                pygame.display.flip()
            else:
                # Pause briefly so a human can observe the AI's strategy
                time.sleep(0.5)

    pygame.quit()
    print("Evaluation closed.")

if __name__ == "__main__":
    evaluate_agent()