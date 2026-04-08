import os
import sys
import time
import json
import argparse
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime

# Dynamically add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(project_root))

from game.smart_grid_env import SmartGridEnv
from game.human_ui import HumanUI

def build_reward_weights(args):
    return {
        "w_finance": args.w_finance,
        "w_co2": args.w_co2,
        "w_waste": args.w_waste,
        "w_blackout": args.w_blackout,
    }


def build_exploit_report(steps, cumulative_reward, blackout_count, totals, blackout_threshold):
    blackout_frequency = (blackout_count / max(1, steps))
    warnings = []
    if blackout_frequency > blackout_threshold:
        warnings.append(
            f"Blackout frequency {blackout_frequency:.1%} exceeded threshold {blackout_threshold:.1%}."
        )
    if totals["unmet_demand"] > 0 and totals["financial_cost"] < 1e-6:
        warnings.append("Potential exploit: unmet demand accumulated with near-zero controllable production cost.")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "steps": steps,
        "cumulative_score": cumulative_reward,
        "blackout_count": blackout_count,
        "blackout_frequency": blackout_frequency,
        "totals": totals,
        "blackout_frequency_threshold": blackout_threshold,
        "warnings": warnings,
    }


def evaluate_agent(args):
    print("Loading trained AI and normalization statistics...")
    reward_weights = build_reward_weights(args)
    print(f"Using reward weights: {reward_weights}")

    # FIX: Use project_root to locate the models folder, not current_dir
    models_dir = os.path.join(os.path.abspath(project_root), "models", "ppo")
    model_path = os.path.join(models_dir, "ppo_smartgrid_100k")
    norm_path = os.path.join(models_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip") or not os.path.exists(norm_path):
        print(f"Error: Files not found at {models_dir}")
        return

    # 1. Create the base environment
    env = SmartGridEnv(render_mode=None, reward_weights=reward_weights)
    
    # 2. Wrap it with DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Load the normalization stats and freeze them (CRITICAL)
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # 4. Load the trained model onto the CPU
    model = PPO.load(model_path, env=vec_env, device="cpu")

    # 5. Initialize the Pygame UI
    ui = HumanUI(log_prefix="ai_game")
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
    blackout_count = 0
    totals = {
        "financial_cost": 0.0,
        "co2_emissions": 0.0,
        "wasted_energy": 0.0,
        "unmet_demand": 0.0,
    }

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
            if info.get("is_blackout", False):
                blackout_count += 1

            components = info.get("reward_components", {})
            totals["financial_cost"] += float(components.get("financial_cost", 0.0))
            totals["co2_emissions"] += float(components.get("co2_emissions", 0.0))
            totals["wasted_energy"] += float(components.get("wasted_energy", 0.0))
            totals["unmet_demand"] += float(components.get("unmet_demand", 0.0))

            # Inject the AI's chosen actions into the UI sliders
            raw_action = action[0] 
            for i, slider in enumerate(ui.sliders):
                slider.value = float(raw_action[i])
                slider._drag_needs_update = True
                slider.update_state(base_env.grid.plants[i].current_power)

            # Render the frame
            ui.render_frame(current_step, info)
            ui.log_decision(base_env.current_step, raw_action.tolist(), cumulative_reward, info.get("is_blackout", False))
            current_step += 1

            if done:
                print(f"\n--- EVALUATION COMPLETE ---")
                print(f"Final Cumulative Score: {cumulative_reward:.2f}")
                report = build_exploit_report(
                    steps=current_step,
                    cumulative_reward=float(cumulative_reward),
                    blackout_count=blackout_count,
                    totals={k: float(v) for k, v in totals.items()},
                    blackout_threshold=args.blackout_threshold,
                )
                os.makedirs("logs", exist_ok=True)
                report_path = args.report_path or os.path.join(
                    "logs",
                    f"exploit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(f"Exploit report saved to: {report_path}")
                print(f"Blackout Frequency: {report['blackout_frequency']:.1%}")
                if report["warnings"]:
                    for warning in report["warnings"]:
                        print(f"WARNING: {warning}")
                else:
                    print("No exploit warnings triggered.")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO smart-grid policy with UI playback.")
    parser.add_argument("--w-finance", type=float, default=0.1, help="Reward weight for financial cost.")
    parser.add_argument("--w-co2", type=float, default=0.1, help="Reward weight for CO2 emissions.")
    parser.add_argument("--w-waste", type=float, default=0.5, help="Reward weight for wasted energy.")
    parser.add_argument("--w-blackout", type=float, default=100.0, help="Reward weight for unmet demand (blackout).")
    parser.add_argument(
        "--blackout-threshold",
        type=float,
        default=0.15,
        help="Exploit warning threshold for blackout frequency.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="",
        help="Optional custom path for exploit report JSON.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    evaluate_agent(parse_args())