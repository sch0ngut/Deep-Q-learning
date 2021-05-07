import argparse
import copy

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    target_dqn = copy.deepcopy(dqn)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    i_step = 0  # Step counter
    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)

        while not done:
            i_step += 1

            # Get action from DQN.
            obs_old = obs
            action = dqn.act(obs_old).item()

            # Act in the true environment.
            obs, reward, done, info = env.step(action)

            # Preprocess incoming observations.
            obs = preprocess(obs, env=args.env).unsqueeze(0)

            # Add the transition to the replay memory.
            memory.push(obs_old, torch.tensor(action), obs, torch.tensor(reward), torch.tensor(not done))

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if i_step % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if i_step % env_config["target_update_frequency"] == 0:
                target_dqn = copy.deepcopy(dqn)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')

    # Close environment after training is completed.
    env.close()
