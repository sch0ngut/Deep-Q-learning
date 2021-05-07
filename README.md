# Deep-Q-learning
An implementation of the Deep-Q algorithm for Reinforcement Learning problems as described in the papers [Playing Atari with Deep Reinforcement Learning] and [Human-level control through deep reinforcement learning]

[Playing Atari with Deep Reinforcement Learning]: https://arxiv.org/abs/1312.5602
[Human-level control through deep reinforcement learning]: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

As of now the agent is able to solve the [CartPole-v0]. Ideally the agent should be able to solve most Atari games such as Pong and Breakout (in process)

[CartPole-v0]: https://gym.openai.com/envs/CartPole-v0/

Training and evaluation can be performed using
```
python train.py --env CartPole-v0
python evaluate.py --env CartPole-v0 --path models/CartPole-v0_best.pt
```

The code is the result of a project as part of the course in Reinforcement Learning that I took at Uppsala University in the spring 2021
