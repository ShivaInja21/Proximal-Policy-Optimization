import torch
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]

def gather_experiences(env, policy_net, num_steps, frame_renderer, frame_stack_count):
    observation = env.reset()
    if isinstance(observation, tuple) and len(observation) == 2:
        observation = observation[0]
    print("Initial observation:", observation)

    if isinstance(observation, dict):
        observation = observation.get('observation', observation)

    observation = np.array(observation)

    angle = np.arctan2(observation[1], observation[0])
    initial_frame = frame_renderer(angle)
    frame_stack = deque([initial_frame] * frame_stack_count, maxlen=frame_stack_count)
    experiences = []

    for _ in range(num_steps):
        stacked_state = np.stack(frame_stack, axis=0)
        state_tensor = torch.as_tensor(stacked_state, dtype=torch.float32).unsqueeze(0)
        action, log_prob = policy_net.generate_action(state_tensor)

        step_result = env.step(action)

        # Modify unpacking to account for the extra returned value (info)
        if len(step_result) == 4:
            next_observation, reward, done, info = step_result
        elif len(step_result) == 5:
            next_observation, reward, done, info, extra = step_result
        else:
            raise ValueError(f"Unexpected number of values returned from env.step(): {len(step_result)}")

        if isinstance(next_observation, dict):
            next_observation = next_observation.get('observation', next_observation)
        next_observation = np.array(next_observation)

        if not isinstance(next_observation, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(next_observation)}")

        angle = np.arctan2(next_observation[1], next_observation[0])
        next_frame = frame_renderer(angle)
        frame_stack.append(next_frame)

        experiences.append((stacked_state, action, reward, log_prob.item()))

        if done:
            observation = env.reset()
            if isinstance(observation, dict):
                observation = observation.get('observation', observation)
            observation = np.array(observation)

            if not isinstance(observation, np.ndarray):
                raise TypeError(f"Expected a NumPy array, got {type(observation)}")

            angle = np.arctan2(observation[1], observation[0])
            initial_frame = frame_renderer(angle)
            frame_stack = deque([initial_frame] * frame_stack_count, maxlen=frame_stack_count)

    return experiences


def optimize_ppo(policy_net, value_net, policy_optimizer, value_optimizer, batch_states, batch_actions,
                 batch_advantages, batch_returns, old_log_probs, clip_epsilon):
    total_policy_loss = 0
    total_value_loss = 0

    for _ in range(10):
        policy_optimizer.zero_grad()
        mean, std_dev = policy_net(batch_states)
        action_dist = torch.distributions.Normal(mean, std_dev)
        new_log_probs = action_dist.log_prob(batch_actions).sum(axis=-1)
        prob_ratios = torch.exp(new_log_probs - old_log_probs)
        clipped_ratios = torch.clamp(prob_ratios, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -(torch.min(prob_ratios * batch_advantages, clipped_ratios * batch_advantages)).mean()
        policy_loss.backward()
        policy_optimizer.step()
        total_policy_loss += policy_loss.item()

        if value_optimizer is not None:
            value_optimizer.zero_grad()
            value_loss = ((value_net(batch_states) - batch_returns) ** 2).mean()
            value_loss.backward()
            value_optimizer.step()
            total_value_loss += value_loss.item()

    average_policy_loss = total_policy_loss / 10
    average_value_loss = total_value_loss / 10 if value_optimizer is not None else None
    return average_policy_loss, average_value_loss


def calculate_advantages(experiences, value_net, gamma, lam):
    states, actions, rewards, log_probs = zip(*experiences)
    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions = torch.as_tensor(np.array(actions), dtype=torch.float32)
    rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).view(-1)
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)

    values = value_net(states).detach().view(-1)
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    advantages = torch.zeros_like(rewards)
    adv_value = 0.0
    for t in reversed(range(len(deltas))):
        adv_value = deltas[t] + gamma * lam * adv_value
        advantages[t] = adv_value

    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    returns = normalized_rewards + gamma * advantages
    return states, actions, advantages, returns, log_probs
