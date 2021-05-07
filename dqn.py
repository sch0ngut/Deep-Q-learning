import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, terminal_transition):
        """
        Pushes a transition to the replay memory.
        terminal_transition stores whether a transition is terminal (0) or not (1)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, terminal_transition)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.eps = self.eps_start
        self.eps_delta = (self.eps_start - self.eps_end) / self.anneal_length
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observations, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        n_observations = observations.shape[0]
        actions = self.forward(observations).argmax(dim=1).int()
        if not exploit:
            exploration_vec = torch.bernoulli(self.eps * torch.ones(n_observations)).int()
            actions += exploration_vec * (np.random.choice(self.n_actions) - actions)
        self.eps = np.max((self.eps_end, self.eps - self.eps_delta))
        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory
    observations, actions, next_observations, rewards, term_trans = memory.sample(target_dqn.batch_size)
    observations = torch.stack(observations).squeeze().to(device)
    n_obs = observations.shape[0]
    actions = torch.stack(actions).reshape(n_obs, 1).to(device)
    next_observations = torch.stack(next_observations).squeeze().to(device)
    rewards = torch.stack(rewards).to(device)
    term_trans = torch.stack(term_trans).int().squeeze().to(device)

    # Compute the current estimates of the Q-values for each state-action pair (s,a).
    q_values = dqn.forward(observations).gather(1, actions)

    # Compute the Q-value targets
    q_value_targets = rewards + target_dqn.gamma * term_trans * target_dqn.forward(next_observations).max(dim=1).values

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
