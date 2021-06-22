# Deep-Q-learning
An implementation of the Deep-Q algorithm for Reinforcement Learning problems as described in the papers [Playing Atari with Deep Reinforcement Learning] and [Human-level control through deep reinforcement learning].

[Playing Atari with Deep Reinforcement Learning]: https://arxiv.org/abs/1312.5602
[Human-level control through deep reinforcement learning]: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

The algorithm has proven to be able to generate a policy that solves most Atari games. In this implementation we focus on the game Pong.

## Evaluation

A trained policy that achieves an approximate average return of 5.8 is provided in the `models` folder. You can run it using
```
python evaluate.py --env Pong-v0 --path models/Pong-v0_best.pt
```
Note that there is some volatility in the performance over several runs.To get an estimate of the expected mean return you can additionally set e.g. `n_eval_episodes=10`. You can also specify `render` if you want to see the agent in action.

## Training
Model training can be performed by running
```
python train.py --env Pong-v0
```
However, it should be noted that the model training is time-consuming. If you don't have a GPU at hand, I recommend performing the model training using Google Colab and make use of their GPUs. A notebook with instructions on how to do so is provided in `DeepQLearning.ipynb`. 
______________________

The code is the result of a project as part of the course  *Reinforcement Learning* that I took at Uppsala University in the spring 2021.
