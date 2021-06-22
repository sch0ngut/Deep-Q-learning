import random
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

    def push(self, obs, action, next_obs, reward, terminal):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, terminal)
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
        self.eps = self.eps_start
        self.anneal_length = env_config["anneal_length"]
        self.eps_step = (self.eps_start - self.eps_end) / self.anneal_length
        self.n_actions = env_config["n_actions"]

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, ), padding=(0, 0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    @staticmethod
    def map_action(x):
        a = x.item()
        if a == 0:
            return 2
        elif a == 1:
            return 3
        else:
            return -1

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        max_action = torch.argmax(self.forward(observation), dim=1)
        if not exploit:
            # possible random action shuffle
            rand_action = torch.randint_like(max_action, 0, self.n_actions).to(device)
            # create a 1D tensor which is a mask for which actions should be taken randomly
            rand_mask = (torch.rand(rand_action.size()) <= self.eps).int().to(device)
            # change the epsilon value after every frame is seen
            self.eps = max(self.eps_end, self.eps - self.eps_step)
            return (1 - rand_mask) * max_action + rand_mask * rand_action
        return max_action


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and concatenate so that there are four tensors in total:
    # observations, actions, next observations and rewards. Special care is taken for terminal transitions
    obs, action, next_obs, reward, term = memory.sample(dqn.batch_size)
    obs = torch.stack(obs).squeeze().to(device)
    n_obs = obs.shape[0]
    action = torch.stack(action).reshape(n_obs, 1).to(device)
    next_obs = torch.stack(next_obs).squeeze().to(device)
    reward = torch.stack(reward).squeeze().to(device)
    term = torch.stack(term).int().to(device)
    
    # Compute the current estimates of the Q-values for each state-action
    q_values = dqn.forward(obs).gather(1, action)
    
    # Compute the Q-value targets.
    q_value_targets = reward + dqn.gamma * term * torch.max(target_dqn.forward(next_obs), dim=1).values
    
    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
