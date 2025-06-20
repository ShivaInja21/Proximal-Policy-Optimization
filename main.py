import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from Modules import ConvPolicyNet, ConvValueNet, render_pendulum_image
from torch_misc import gather_experiences, calculate_advantages, optimize_ppo, ReplayMemory
import os

# Define hyperparameters as a dictionary
config = {
    "num_epochs": 30,
    "steps_per_cycle": 4000,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "clip_bound": 0.2,
    "policy_learning_rate": 3e-4,
    "value_fn_learning_rate": 1e-3,
    "neuron_layers": [64, 64],
    "memory_capacity": 10000,
    "batch_capacity": 64,
}

# Environment setup
env = gym.make('Pendulum-v1')
action_dim = env.action_space.shape[0]
observation_dim = (3, 64, 64)

# Initialize models and optimizers
policy_net = ConvPolicyNet(observation_dim, action_dim, config["neuron_layers"])
value_net = ConvValueNet(observation_dim, config["neuron_layers"])
policy_optimizer = Adam(policy_net.parameters(), lr=config["policy_learning_rate"])
value_optimizer = Adam(value_net.parameters(), lr=config["value_fn_learning_rate"])
replay_memory = ReplayMemory(config["memory_capacity"])

# Define a function to log training progress
def log_progress(training_results):
    training_progress, policy_loss_logs, value_loss_logs = training_results
    
    # Ensure the output directory exists
    output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    epochs = config["num_epochs"]

    # Reward plot
    plt.figure(figsize=(10, 6))
    for config_name, reward_curve in training_progress.items():
        plt.plot(range(1, epochs + 1), reward_curve, label=config_name)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_plot.png"), bbox_inches='tight')
    plt.close()

    # Policy loss plot
    plt.figure(figsize=(10, 6))
    for config_name, loss_curve in policy_loss_logs.items():
        plt.plot(range(1, epochs + 1), loss_curve, label=config_name)
    plt.xlabel("Epoch")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "policy_loss_plot.png"), bbox_inches='tight')
    plt.close()

    # Value loss plot (log scale)
    plt.figure(figsize=(10, 6))
    for config_name, loss_curve in value_loss_logs.items():
        normalized_losses = np.log(np.array(loss_curve) + 1e-5)
        plt.plot(range(1, epochs + 1), normalized_losses, label=config_name)
    plt.xlabel("Epoch")
    plt.ylabel("Log(Value Loss)")
    plt.title("Value Loss Curves (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_loss_plot.png"), bbox_inches='tight')
    plt.close()

# Define training logic
def train(config, env, policy_net, value_net, policy_optimizer, value_optimizer, replay_memory):
    training_progress = {}
    policy_loss_logs = {}
    value_loss_logs = {}

    # Setup training configurations
    training_settings = {
        "with_clipping": (True, True),
        "without_clipping": (False, True),
        "with_GAE": (True, True),
        "without_GAE": (True, False),
    }

    for config_name, (clipping, gae) in training_settings.items():
        rewards_log, policy_loss_log, value_loss_log = [], [], []
        
        for epoch in range(config["num_epochs"]):
            episodes = gather_experiences(env, policy_net, config["steps_per_cycle"], render_pendulum_image, frame_stack_count=3)

            # Store experiences in memory
            for episode in episodes:
                replay_memory.store(episode)

            # Sample batch from memory
            batch = replay_memory.sample(config["batch_capacity"])
            batch_states, batch_actions, advantage_scores, target_returns, prev_log_probs = calculate_advantages(
                batch, value_net, config["discount_factor"], config["gae_lambda"] if gae else 0.0
            )

            # Optimize policy network using PPO
            policy_loss, _ = optimize_ppo(
                policy_net, value_net, policy_optimizer, value_optimizer,
                batch_states, batch_actions, advantage_scores, target_returns, prev_log_probs,
                config["clip_bound"] if clipping else 1e6
            )

            # Update value network
            value_optimizer.zero_grad()
            value_loss = ((value_net(batch_states) - target_returns) ** 2).mean()
            value_loss.backward()
            value_optimizer.step()

            # Log training progress
            mean_reward = np.mean([ep[2] for ep in episodes])
            rewards_log.append(mean_reward)
            policy_loss_log.append(policy_loss)
            value_loss_log.append(value_loss.item())

            print(f"{config_name} - Cycle {epoch + 1}: Mean Reward: {mean_reward:.2f}")

        training_progress[config_name] = rewards_log
        policy_loss_logs[config_name] = policy_loss_log
        value_loss_logs[config_name] = value_loss_log

    return training_progress, policy_loss_logs, value_loss_logs

# Run training and visualization
training_results = train(config, env, policy_net, value_net, policy_optimizer, value_optimizer, replay_memory)
log_progress(training_results)
