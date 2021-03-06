import argparse
import gym
import torch
from gym.wrappers import AtariPreprocessing
import matplotlib.pyplot as plt
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--path_prefix', type=str, help='Path prefix to store DQN model.', default='')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--plot_episode_returns', dest='plot_episode_returns', action='store_true',
                    help='Save a plot with episode returns during course of training', default=True)
parser.set_defaults(plot_episode_returns=True)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'Pong-v0': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=True)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create target network and set parameters to the same as dqn.
    target = DQN(env_config=env_config).to(device)
    target.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    obs_stack_size = env_config['observation_stack_size']

    if args.plot_episode_returns:
        episodes = []
        episode_returns = []

    step = 0
    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)
        
        while not done:
            step += 1

            with torch.no_grad():
                action = dqn.act(obs_stack)
                # Act in the true environment.
                obs_next, reward, done, info = env.step(dqn.map_action(action))

                # Preprocess incoming observation.
                obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
                obs_next_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
                
                # Add the transition to the replay memory.
                memory.push(obs_stack, action, obs_next_stack, torch.tensor(reward), torch.tensor((not done)))
                obs_stack = obs_next_stack

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            if args.plot_episode_returns:
                episodes.append(episode)
                episode_returns.append(mean_return)
                plt.plot(episodes, episode_returns)
                plt.xlabel('episodes')
                plt.ylabel('mean return')
                plt.title('Training progress')
                plt.savefig(f'{args.path_prefix}plots/training_progress.png', dpi=1000)

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'{args.path_prefix}models/{args.env}_test.pt')

    # Close environment after training is completed.
    env.close()
