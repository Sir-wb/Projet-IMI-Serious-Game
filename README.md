## Project Roadmap & To-Do List

This project is structured into 5 distinct phases to ensure mathematical robustness and optimize the computational load for Reinforcement Learning (RL) training.

### Phase 1: Physical Engine & Human MVP (Completed)
- [x] **Grid Physics (`grid_model.py`):** Implement absolute limits and physical inertia (`ramp_rate`) for gas, coal, and nuclear plants.
- [x] **Stochastic Engine (`stochastic_engine.py`):** Generate 12-hour forecasts with a mathematically expanding "cone of uncertainty" for demand and renewables (solar/wind).
- [x] **Gymnasium Environment (`smart_grid_env.py`):** Vectorize the observation space natively in NumPy (`np.float32`) for maximum RL performance.
- [x] **Human Dashboard (`human_ui.py` & `play.py`):** Build an optimized, memory-efficient Pygame/Matplotlib interface to log human decision-making under uncertainty.

### Phase 2: Reinforcement Learning Setup (AI Training)
- [ ] **Headless Training Script (`train_ai.py`):** Instantiate the environment without the Pygame UI to maximize frames-per-second (FPS) during training on the RTX 4070/Mac GPU
- [ ] **Environment Wrapping:** Apply Stable Baselines 3 utilities (`DummyVecEnv`, `VecNormalize`) to strictly normalize observations and rewards for faster neural network convergence.
- [ ] **Algorithm Implementation:** Deploy a continuous-action RL algorithm (PPO or SAC) adapted to our `Box(0.0, 1.0)` action space.
- [ ] **TensorBoard Integration:** Configure the training loop to output logs for monitoring the learning curve, reward progression, and episode lengths.

### Phase 3: AI Evaluation and Visualization
- [ ] **Evaluation Script (`evaluate_ai.py`):** Write a script to load the trained model (e.g., `ppo_smart_grid.zip`).
- [ ] **Visual AI Playthrough:** Connect the trained agent to the `human_ui.py` dashboard to visually observe the AI adjusting sliders and reacting to the uncertainty cone in real-time.
- [ ] **AI Data Logging:** Ensure the AI's decisions are logged into CSV files using the exact same format as the human player for 1:1 statistical comparison.

### Phase 4: Reward Shaping and Fine-Tuning
- [ ] **The "Exploit" Verification:** Review the AI's learned behavior to ensure it balances the grid rather than exploiting mathematical loopholes (e.g., choosing total blackout over expensive gas costs).
- [ ] **Hyperparameter Tuning:** Adjust the weights in the reward function (`w_finance`, `w_co2`, `w_waste`, `w_blackout`) based on initial AI behavior to force complex, realistic trade-offs.

### Phase 5: Data Analysis (Phimeca Objective)
- [ ] **Human vs. AI Comparison:** Extract and analyze the CSV logs to compare risk management strategies. 
- [ ] **Behavioral Profiling:** Determine if the human player over-reacts to the uncertainty cone (overproducing/wasting energy) compared to the AI's optimized edge-riding strategy.