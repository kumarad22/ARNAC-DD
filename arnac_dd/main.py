import os
import gymnasium as gym
import torch
import argparse
import pandas as pd
import numpy as np
from collections import deque
import torch.optim as optim
from model import Actor, Critic
from utils.utils import get_action, save_checkpoint, get_param_range
from utils.running_state import ZFilter
from hparams import HyperParams as hp
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='ARNACDD')
parser.add_argument('--env', type=str, default="Hopper-v4", help='Mujoco environment name')
parser.add_argument('--load_model', type=str, default=None, help="Checkpoint to load")
parser.add_argument('--render', action="store_true", help="Render environment")

parser.add_argument('--logdir', type=str, default='logs', help='TensorBoard logs directory')

parser.add_argument('--seed', type=int, default=500, help='Random seed')
parser.add_argument('--drop_num', type=int, default=1, help='Drop number for Data Drop')
parser.add_argument('--iter', type=int, default=15000, help='No. of outer loops (epochs)')
args = parser.parse_args()

# ----------------------------
# Select training algorithm
# ----------------------------
if args.algorithm == "ARNACDD":
    from agent.nacdd import train_model
else:
    raise NotImplementedError(f"Algorithm {args.algorithm} not implemented.")

if __name__ == "__main__":
    # ----------------------------
    # Device setup
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hp.drop_num = args.drop_num

    # ----------------------------
    # Initialize environment & seeding
    # ----------------------------
    env = gym.make(args.env)
    obs, info = env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print(f"State size: {num_inputs}, Action size: {num_actions}")

    # ----------------------------
    # Experiment setup
    # ----------------------------
    exp_id = f"{args.algorithm}_{args.env}_dropnum_{hp.drop_num}_seed_{args.seed}_Adam_2nd"
    args.logdir = os.path.join(args.logdir, exp_id)

    model_path = os.path.join("save_model", exp_id)
    
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # TensorBoard logger
    writer = SummaryWriter(args.logdir)

    # CSV logger
    csv_log_path = os.path.join(args.logdir, 'training_log.csv')
    csv_columns = ['epoch', 'score', 'avg_reward_per_step', 'num_episodes']
    log_df = pd.read_csv(csv_log_path) if os.path.exists(csv_log_path) else pd.DataFrame(columns=csv_columns)

    # ----------------------------
    # Create actor and critic
    # ----------------------------
    actor = Actor(num_inputs, num_actions).to(device)
    critic = Critic(num_inputs).to(device)

    # running_state = ZFilter((num_inputs,), clip=5)
    running_state = ZFilter((num_inputs,), clip=5, device=device)

    # ----------------------------
    # Load pre-trained checkpoint
    # ----------------------------
    if args.load_model is not None:
        ckpt = torch.load(args.load_model, map_location=device)
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']
        print(f"Loaded checkpoint: {args.load_model}, Zfilter N {running_state.rs.n}")

    # ----------------------------
    # Optimizers
    # ----------------------------
    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)
    param_range = get_param_range(critic, hp.max_param_size)

    # ----------------------------
    # Main training loop
    # ----------------------------
    episodes = 0
    outer_loops = args.iter

    for epoch in trange(outer_loops, desc="Epochs"):
        actor.eval(), critic.eval()

        # ==================================================
        # 1) Collect critic training samples
        # ==================================================
        critic_memory = deque()
        steps, current_average = 0, 0
        total_critic_steps = int(2048 * hp.drop_num)

        with tqdm(total=total_critic_steps, desc="Collect critic", leave=False) as pbar_critic:
            while steps < total_critic_steps:
                episodes += 1
                state, _ = env.reset()
                state = running_state(state)
                for _ in range(10000):
                    steps += 1
                    pbar_critic.update(1)
                    mu, std, _ = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = get_action(mu, std)[0]
                    next_state, reward, done1, done2, _ = env.step(action)
                    next_state = running_state(next_state)
                    mask = 0 if (done1 or done2) else 1
                    current_average += (reward - current_average) / steps
                    critic_memory.append([state, action, reward, mask, current_average])
                    state = next_state
                    if done1 or done2:
                        break

        # ==================================================
        # 2) Collect actor training samples
        # ==================================================
        actor_memory = deque()
        steps, current_average = 0, 0
        total_actor_steps = int(2048 * hp.drop_num)

        with tqdm(total=total_actor_steps, desc="Collect actor", leave=False) as pbar_actor:
            while steps < total_actor_steps:
                state, _ = env.reset()
                state = running_state(state)
                for _ in range(10000):
                    steps += 1
                    pbar_actor.update(1)
                    mu, std, _ = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = get_action(mu, std)[0]
                    next_state, reward, done1, done2, _ = env.step(action)
                    next_state = running_state(next_state)
                    mask = 0 if (done1 or done2) else 1
                    current_average += (reward - current_average) / steps
                    actor_memory.append([state, action, reward, mask, current_average])
                    state = next_state
                    if done1 or done2:
                        break

        # ==================================================
        # 3) Evaluation
        # ==================================================
        steps, scores = 0, []
        total_eval_steps = 2048

        with tqdm(total=total_eval_steps, desc="Evaluation", leave=False) as pbar_eval:
            while steps < total_eval_steps:
                state, _ = env.reset()
                state = running_state(state)
                score = 0
                for _ in range(10000):
                    if args.render:
                        env.render()
                    steps += 1
                    pbar_eval.update(1)
                    mu, std, _ = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = mu.detach().cpu().numpy()[0]
                    next_state, reward, done1, done2, _ = env.step(action)
                    next_state = running_state(next_state)
                    score += reward
                    state = next_state
                    if done1 or done2:
                        break
                scores.append(score)

        score_avg = float(np.mean(scores))
        avg_reward_per_step = score_avg / steps
        num_episodes = len(scores)

        print(f"[Eval] Episodes: {num_episodes}, Score: {score_avg:.2f}, "
              f"Avg reward/step: {avg_reward_per_step:.4f}")

        # TensorBoard logging
        writer.add_scalar('log/score', score_avg, epoch)
        writer.add_scalar('log/avg_reward_per_step', avg_reward_per_step, epoch)

        # CSV logging
        new_row = pd.DataFrame([{
            'epoch': epoch,
            'score': score_avg,
            'avg_reward_per_step': avg_reward_per_step,
            'num_episodes': num_episodes
        }])
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        if epoch % 100 == 0 or epoch == outer_loops - 1:
            log_df.to_csv(csv_log_path, index=False)

        # ==================================================
        # 4) Train actor & critic
        # ==================================================
        actor.train(), critic.train()
        train_model(actor, critic, actor_memory, critic_memory,
                    actor_optim, critic_optim, param_range, device)

        # ==================================================
        # 5) Checkpointing
        # ==================================================
        if epoch % 1000 == 0:
            ckpt_path = os.path.join(model_path, f"ckpt_{int(score_avg)}.pth.tar")
            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n': running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': vars(args),
                'score': int(score_avg)
            }, filename=ckpt_path)
