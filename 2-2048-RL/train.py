import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dqn_agent import DQN, ReplayBuffer, select_action
from rl_env import Game2048Env

# Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99
BATCH_SIZE = 32
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3
MAX_EPISODES = 1000

# Initialize environment, model, and optimizer
env = Game2048Env()
policy_net = DQN(input_size=16)
target_net = DQN(input_size=16)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer()

# Epsilon for exploration
epsilon = EPSILON_START

# For live plotting
rewards = []
plt.ion()  # Interactive mode on

# Training loop
for episode in range(MAX_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    timestep = 0

    while not done:
        action = select_action(state, policy_net, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if len(replay_buffer) > BATCH_SIZE:
            states, actions, rewards_batch, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards_batch = torch.FloatTensor(np.array(rewards_batch))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            q_all_values = policy_net(states)
            q_values = q_all_values.gather(1, actions.unsqueeze(1))
            next_q_values = target_net(next_states).max(1)[0].detach()
            expected_q_values = rewards_batch + GAMMA * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values.squeeze(), expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timestep += 1

    rewards.append(total_reward)

    print(f"Episode: {episode}, Total Reward: {total_reward}")
    print(f"State: {state}")
    print(f"Action taken: {action}, Reward: {reward}")
    print(f"Q-values: {q_all_values}")

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Save model periodically
    if episode % 100 == 0:
        torch.save(policy_net.state_dict(), f"model_{episode}.pth")

    # Live plot
    if episode % 10 == 0:
        plt.clf()
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.plot(rewards, label="Total Reward")
        plt.legend()
        plt.pause(0.01)

plt.ioff()
plt.show()
