import torch
import math
import numpy as np
import os

def get_param_range(q_network, max_param_size):
    param_min, param_max = {}, {}
    with torch.no_grad():
        for name, param in q_network.named_parameters():
            param_min[name] = param.data.clone() - max_param_size
            param_max[name] = param.data.clone() + max_param_size
    return param_min, param_max

def split_memory(memory, device=None):
    if not memory:
        return tuple()
    
    num_entry = len(memory[0])
    # Pre-allocate lists
    entry_lists = [[] for _ in range(num_entry)]
    
    # Batch process tensors more efficiently
    for m in memory:
        for i in range(num_entry):
            entry_lists[i].append(m[i])
    
    # Convert entire batches at once
    results = []
    for i in range(num_entry):
        entries = entry_lists[i]
        if isinstance(entries[0], torch.Tensor):
            # Batch move all tensors to CPU at once
            batch_tensor = torch.stack([e.detach() for e in entries])
            results.append(batch_tensor.cpu().numpy())
        else:
            results.append(np.array(entries))
    
    return tuple(results)

def get_action(mu, std, device=None):
    action = torch.normal(mu, std)
    return action.detach().cpu().numpy()

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_2pi = torch.log(2 * torch.tensor(math.pi, device=var.device))
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * log_2pi - logstd
    return log_density.sum(1, keepdim=True)

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        if grad is not None:
            grad_flatten.append(grad.view(-1))
    return torch.cat(grad_flatten) if grad_flatten else torch.tensor([], device=grads[0].device)

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        if hessian is not None:
            hessians_flatten.append(hessian.contiguous().view(-1))
    return torch.cat(hessians_flatten).data if hessians_flatten else torch.tensor([])

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)

def update_model(model, new_params):
    index = 0
    for param in model.parameters():
        param_length = param.numel()
        new_param = new_params[index:index + param_length].view(param.shape)
        param.data.copy_(new_param)
        index += param_length

def kl_divergence(new_actor, old_actor, states, device=None):
    # Use model's device instead of global device
    model_device = next(new_actor.parameters()).device
    states = torch.as_tensor(states, dtype=torch.float32, device=model_device)
    
    mu, std, logstd = new_actor(states)
    mu_old, std_old, logstd_old = old_actor(states)
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)