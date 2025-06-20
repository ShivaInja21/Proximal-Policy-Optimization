# PPO with Visual Input for Pendulum Control 🎮🧠

This project implements a **Proximal Policy Optimization (PPO)** agent using **convolutional neural networks (CNNs)** to control a pendulum in OpenAI's `Pendulum-v1` environment. Unlike traditional state-based methods, this agent learns directly from **grayscale image observations**, showcasing the integration of **deep reinforcement learning** and **computer vision**.

## 🧩 Key Features

- **CNN-based Actor-Critic Networks**  
  Learns from visual input using stacked image frames as state representations.

- **Proximal Policy Optimization (PPO)**  
  Implements clipped surrogate objective for stable policy updates.

- **Generalized Advantage Estimation (GAE)**  
  Enhances advantage calculation by reducing variance while maintaining bias control.

- **Frame Rendering and Stacking**  
  Converts pendulum angles into 64x64 pixel images with frame stacking to capture temporal information.

- **Replay Buffer and Sampling**  
  Custom replay memory for storing and sampling trajectory batches.

- **Training Modes**  
  Supports 4 configurations for ablation study:
  - PPO with Clipping and GAE  
  - PPO with Clipping only  
  - PPO with GAE only  
  - PPO without Clipping or GAE

- **Logging & Visualization**  
  Plots and saves:
  - Cumulative rewards per epoch  
  - Policy loss  
  - Value loss (log scale)

## 📊 Example Results

Plots are saved in the `training_plots/` directory after training:
- `reward_plot.png`
- `policy_loss_plot.png`
- `value_loss_plot.png`

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/ppo-pendulum-visual.git
cd ppo-pendulum-visual
pip install -r requirements.txt
