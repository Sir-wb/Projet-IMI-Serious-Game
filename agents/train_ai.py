import os
import sys
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(project_root))
from game.smart_grid_env import SmartGridEnv

from game.smart_grid_env import SmartGridEnv

def train_agent():
    print("Initializing headless Smart Grid Environment for RL Training...")

    # --- Hardware Detection ---
    if torch.cuda.is_available():
        device = "cuda"
        print("Hardware detected: NVIDIA GPU (CUDA). Utilizing dedicated PC GPU...")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Hardware detected: Apple Silicon (MPS). Utilizing Mac GPU...")
    else:
        device = "cpu"
        print("Warning !!! : No compatible GPU detected. Defaulting to CPU...")

    # 1. Create the environment in headless mode for maximum speed
    env = SmartGridEnv(render_mode=None)
    
    # 2. Vectorize the environment
    vec_env = DummyVecEnv([lambda: env])
    
    # 3. Normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Setup directories for models and logs
    models_dir = "models/ppo"
    log_dir = "logs/tensorboard"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 4. Initialize the PPO Algorithm with explicit device routing
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, device=device)

    # 5. Train the Agent
    timesteps = 100000 
    print(f"Starting training for {timesteps} timesteps on {device.upper()}...")
    model.learn(total_timesteps=timesteps, tb_log_name="PPO_SmartGrid_Run1")

    # 6. Save the model and the normalization statistics
    model_path = os.path.join(models_dir, "ppo_smartgrid_100k")
    norm_path = os.path.join(models_dir, "vec_normalize.pkl")
    
    model.save(model_path)
    vec_env.save(norm_path)
    
    print(f"Training complete. Model saved to {model_path}.zip")
    print(f"Normalization statistics saved to {norm_path}")

if __name__ == "__main__":
    train_agent()