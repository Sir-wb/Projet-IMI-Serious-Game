# PHIMECA — Serious Game & Reinforcement Learning Project
**Tutors:** Valentin Pibernus (`pibernus@phimeca.com`) · Sylvain Girard (`girard@phimeca.com`)  
**Team size:** 2–3 students  
**Partner:** [Phimeca Engineering](https://www.phimeca.com)

---

## Context & Problem Statement

Many industrial problems boil down to **decision-making under uncertainty**. Experts build complex models and use available measurements to make the best possible decision — but this is hard because:

- The problem may be highly non-linear or high-dimensional
- Multiple conflicting objectives must be balanced simultaneously
- Measurements arrive sequentially or are incomplete
- The model and measurements carry strong uncertainty and/or come from stochastic phenomena

The core challenge: **experts often don't know how to integrate randomness into their decisions**, leading to over-conservative or under-conservative choices. Without proper cost-risk trade-offs, decisions are rarely optimal.

**Our goal:** Design an approach to integrate uncertainty into decision-making, using a serious game as the laboratory.

---

## Our Project: Smart Grid Energy Management

We chose an **electrical grid balancing simulation** as our serious game scenario. A human (or AI) operator must manage a power grid over 12-hour episodes by controlling the output of:

- ⚫ **Coal** plants — cheap but slow to ramp, high CO₂
- ☢️ **Nuclear** plants — very stable, nearly zero CO₂, extremely slow to adjust
- 🔥 **Gas** plants — expensive but fast to respond, useful for peaks
- ☀️ **Solar** & 💨 **Wind** — free and clean, but stochastic and unpredictable

The operator sees a **12-hour forecast cone** (expanding uncertainty over time) for both demand and renewable production, and must constantly balance supply vs. demand while managing financial cost, CO₂ emissions, energy waste, and blackout risk.

---

## Architecture Overview

| File | Role |
|---|---|
| `grid_model.py` | Physical engine — absolute limits, inertia (`ramp_rate`) for each plant type |
| `stochastic_engine.py` | Generates expanding uncertainty cones for demand & renewables |
| `smart_grid_env.py` | Gymnasium-compatible RL environment (NumPy `float32` observation space) |
| `human_ui.py` & `play.py` | Pygame/Matplotlib human dashboard for interactive play & data logging |
| `train_ai.py` | Headless training script (no UI) for GPU-accelerated RL |
| `evaluate_ai.py` | Loads trained model and runs evaluation episodes |

---

## Project Phases

### ✅ Phase 1 — Physical Engine & Human MVP *(Completed)*

- Grid physics with inertia constraints (ramp rates) for gas, coal, and nuclear
- Stochastic forecast engine with a mathematically expanding cone of uncertainty
- Vectorized Gymnasium environment in NumPy for RL performance
- Playable human dashboard with decision logging to CSV

### 🔄 Phase 2 — Reinforcement Learning Setup

- **Headless training** (`train_ai.py`): environment runs without UI for maximum FPS on GPU
- **Environment wrapping**: Stable Baselines 3 `DummyVecEnv` + `VecNormalize` for observation and reward normalization
- **Algorithm**: PPO or SAC on a continuous `Box(0.0, 1.0)` action space
- **TensorBoard**: monitor reward curves, episode lengths, and learning progress

### 🔄 Phase 3 — AI Evaluation & Visualization

- Evaluation script loads a trained model (`ppo_smart_grid.zip`) and runs test episodes
- Plug the AI agent back into the human UI to visually observe its behavior in real-time
- AI decisions logged to CSV in the **exact same format** as human logs → 1:1 comparison

### 🔄 Phase 4 — Reward Shaping & Fine-Tuning

- Audit the AI's learned policy for exploits (e.g., preferring full blackout over costly gas usage)
- Tune reward weights (`w_finance`, `w_co2`, `w_waste`, `w_blackout`) to enforce realistic trade-offs

### 🔄 Phase 5 — Human vs. AI Analysis *(Phimeca Objective)*

- Extract and statistically compare CSV logs: human players vs. trained AI agent
- **Key research question:** Do humans over-react to the uncertainty cone (over-producing, wasting energy) compared to the AI's optimized strategy?
- Profile risk perception: does the human's awareness and score improve with practice?

---

## Reward Function

The scalar reward at each timestep balances four competing objectives:

$$
r = -\left( w_{\text{finance}} \cdot C_{\text{finance}} + w_{\text{co2}} \cdot C_{\text{co2}} + w_{\text{waste}} \cdot C_{\text{waste}} + w_{\text{blackout}} \cdot C_{\text{blackout}} \right)
$$

Tuning these weights is central to Phase 4 — they encode the value judgment about what constitutes a "good" grid operator decision.

---

## Key Technical Choices

| Choice | Rationale |
|---|---|
| **Gymnasium** standard API | Plug-and-play compatibility with Stable Baselines 3 and future algorithms |
| **Continuous action space** `Box(0, 1)` | More realistic than discrete; maps naturally to "slider" metaphor in UI |
| **PPO / SAC** | Both handle continuous actions well; PPO is more stable, SAC is more sample-efficient |
| **VecNormalize** | Critical for NN convergence — raw MW values vary wildly in scale |
| **Same CSV format** for human & AI | Enables direct statistical comparison without data transformation |

---

## Deliverables for Phimeca

1. **Playable serious game** usable by non-expert humans
2. **Trained RL agent** capable of balancing the grid autonomously
3. **Comparative analysis report**: human decision-making vs. AI strategy under uncertainty
4. **Behavioral insights**: quantify over/under-conservatism in human operators

---

## References

- Last year's project: [PHIMECA Serious Game 2024–25](https://cermics.enpc.fr/~gontierd/RACA/ProjetsDepartement/2425/PHIMECA_Serious_Game.pdf)
- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
