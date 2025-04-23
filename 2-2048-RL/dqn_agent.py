import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, output_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
		
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state).argmax().item()