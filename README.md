# Optimal Convergence Rate for Average-Reward Reinforcement Learning with General Parametrized Actor and Neural Critic

## Required environments:
- see requirements.txt file

## How to run our algorithm

```bash
cd arnac_dd
python main.py --drop_num W --seed X --algorithm Y --env Z
```
- W is the drop number for which we choose from [1, 3, 5];

- X is the random seed for which we choose from [500, 1000, 1500];

- Y is the algorithm for which we set as ARNACDD;

- Z is the environment which can be one of [Hopper-v4, HalfCheetah-v4].

## How to run the baselines

```bash
cd pg_travel/mujoco
python main.py --seed X --algorithm Y --env Z
```

- X is the random seed for which we choose from [500, 1000, 1500];

- Y is one of the baselines: [PG, NPG, TRPO, PPO];

- Z is the environment which we chose as HalfCheetah-v4.

[PG, NPG, TRPO, PPO] correspond to the following policy gradient (PG) algorithms:
* Vanilla Policy Gradient: R. Sutton, et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NIPS 2000.
* Truncated Natural Policy Gradient: S. Kakade, "A Natural Policy Gradient", NIPS 2002.
* Trust Region Policy Optimization: J. Schulman, et al., "Trust Region Policy Optimization", ICML 2015.
* Proximal Policy Optimization: J. Schulman, et al., "Proximal Policy Optimization Algorithms", arXiv, https://arxiv.org/pdf/1707.06347.pdf.


## Reference
We referenced the codes from:
* [pg_travel](https://github.com/reinforcement-learning-kr/pg_travel)
