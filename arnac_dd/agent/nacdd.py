import numpy as np
import torch
from utils.utils import *
from hparams import HyperParams as hp
from model import Actor

# progress bars
from tqdm import trange, tqdm

# automatic device selection
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gae(rewards, masks, values, r_bar=None, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    # ensure r_bar is a tensor on the correct device
    r_bar_tensor = torch.as_tensor(r_bar, dtype=torch.float32, device=device)

    running_returns = 0.0
    previous_value = 0.0
    running_advants = 0.0
    eps = 1e-8

    # Average-reward (differential) GAE:
    for t in reversed(range(len(rewards))):
        diff_reward = rewards[t] - r_bar_tensor[t]

        # Monte Carlo estimate of returns:
        running_returns = diff_reward + running_returns * masks[t]

        # Temporal Difference Error for Average-Reward Formulation:
        running_tderror = diff_reward + previous_value * masks[t] - values.data[t]

        # Generalized Advantage Estimation:
        running_advants = running_tderror + hp.gamma * hp.lamda * running_advants * masks[t]

        previous_value = values.data[t]

        returns[t] = running_returns
        advants[t] = running_advants

    # normalize advantages
    advants = (advants - advants.mean()) / (advants.std() + eps)
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    mu, std, logstd = actor(states)
    new_policy = log_density(actions, mu, std, logstd)
    advants = advants.unsqueeze(1)

    surrogate = advants * torch.exp(new_policy - old_policy)
    surrogate = surrogate.mean()

    return surrogate


def train_critic(critic, states, returns, advants, critic_optim, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    criterion = torch.nn.MSELoss()
    n = len(states)
    if n == 0:
        return

    # All tensors are already on GPU, no conversion needed
    arr = torch.randperm(n, device=device)

    for epoch in trange(5, desc="critic epochs", leave=False):
        for i in tqdm(range(max(1, n // hp.batch_size)), desc="critic batches", leave=False):
            batch_index = arr[i * hp.batch_size: (i + 1) * hp.batch_size]
            if len(batch_index) == 0:
                continue

            inputs = states[batch_index]
            target1 = returns[batch_index].unsqueeze(1)
            target2 = advants[batch_index].unsqueeze(1)

            critic_optim.zero_grad()

            # avoiding mixed precision here for stability
            values = critic(inputs)
            loss = criterion(values, target1 + target2)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            
            critic_optim.step()


def fisher_vector_product(actor, states, p, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    p = p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states).mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p


def conjugate_gradient(actor, states, b, nsteps=10, residual_tol=1e-10, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    x = torch.zeros(b.size(), device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for i in trange(nsteps, desc="CG iters", leave=False):
        _Avp = fisher_vector_product(actor, states, p, device=device)
        denom = torch.dot(p, _Avp)
        if denom.abs() < 1e-12:
            break
        alpha = rdotr / denom
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def train_model(actor, critic, actor_memory, critic_memory, actor_optim, critic_optim, param_range, device=None):
    device = DEFAULT_DEVICE if device is None else (torch.device(device) if isinstance(device, str) else device)

    # =======================================================
    # step 1: train the critic - FULLY ON GPU
    # =======================================================
    states, actions, rewards, masks, current_average = split_memory(critic_memory, device=device)

    # Convert to GPU immediately
    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    
    values = critic(states_tensor)
    returns, advants = get_gae(rewards, masks, values, r_bar=current_average, device=device)

    # Data drop on GPU using tensor indexing
    drop_indices = torch.arange(0, len(states_tensor), hp.drop_num, device=device)
    states_dropped = states_tensor[drop_indices]
    returns_dropped = returns[drop_indices] 
    advants_dropped = advants[drop_indices]

    used_samples = len(states_dropped)
    print(f"[train_model] Critic: using {used_samples} samples after Data Drop (drop_num={hp.drop_num})")

    train_critic(critic, states_dropped, returns_dropped, advants_dropped, critic_optim, device=device)

    with torch.no_grad():
        for name, param in critic.named_parameters():
            if "bias" in name:
                continue
            param.data.clamp_(param_range[0][name], param_range[1][name])

    # =======================================================
    # step 2: train the actor - FULLY ON GPU
    # =======================================================
    states, actions, rewards, masks, current_average = split_memory(actor_memory, device=device)

    # Convert everything to GPU at once
    states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)

    values = critic(states_tensor)
    returns, advants = get_gae(rewards, masks, values, r_bar=current_average, device=device)

    # Data drop on GPU for all tensors
    drop_indices = torch.arange(0, len(states_tensor), hp.drop_num, device=device)
    states_dropped = states_tensor[drop_indices]
    actions_dropped = actions_tensor[drop_indices]
    returns_dropped = returns[drop_indices]
    advants_dropped = advants[drop_indices]

    used_samples_actor = len(states_dropped)
    print(f"[train_model] Actor: using {used_samples_actor} samples after Data Drop (drop_num={hp.drop_num})")

    # ==========================================================
    # step 3: find gradient of loss and hessian of kl - ALL ON GPU
    # ==========================================================
    mu, std, logstd = actor(states_dropped)
    old_policy = log_density(actions_dropped, mu, std, logstd)

    loss = surrogate_loss(actor, advants_dropped, states_dropped, old_policy.detach(), actions_dropped, device=device)
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)
    step_dir = conjugate_gradient(actor, states_dropped, loss_grad.data, nsteps=10, device=device)

    params = flat_params(actor)
    shs = 0.5 * (step_dir * fisher_vector_product(actor, states_dropped, step_dir, device=device)).sum(0, keepdim=True)
    
    shs_clamped = torch.clamp(shs / hp.max_kl, min=1e-8)
    step_size = 1 / torch.sqrt(shs_clamped)[0]
    full_step = step_size * step_dir

    # =======================================================
    # step 4: do backtracking line search - ALL ON GPU
    # =======================================================
    old_actor = Actor(actor.num_inputs, actor.num_outputs).to(device)
    update_model(old_actor, params)
    expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

    flag = False
    fraction = 1.0
    for i in tqdm(range(10), desc="line search", leave=False):
        new_params = params + fraction * full_step
        update_model(actor, new_params)
        new_loss = surrogate_loss(actor, advants_dropped, states_dropped, old_policy.detach(), actions_dropped, device=device)
        loss_improve = new_loss - loss

        kl = kl_divergence(new_actor=actor, old_actor=old_actor, states=states_dropped)
        kl = kl.mean()

        expected_improve_scaled = expected_improve * fraction
        improve_ratio = loss_improve / (expected_improve_scaled + 1e-8)

        print(f'kl: {kl.item():.6f}  loss improve: {loss_improve.item():.6f}  '
              f'expected improve: {expected_improve_scaled.item():.6f}  '
              f'improve ratio: {improve_ratio.item():.6f}  line search: {i}')

        if kl < hp.max_kl and improve_ratio > 0.5:
            flag = True
            break

        fraction *= 0.5

    if not flag:
        params = flat_params(old_actor)
        update_model(actor, params)
        print('policy update does not improve the surrogate')