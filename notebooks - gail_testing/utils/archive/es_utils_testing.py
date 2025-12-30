import math
import torch
import torch.nn as nn
from utils.env_utils import *
from marl_aquarium.aquarium_v0 import parallel_env


def discriminator_reward(prey_tensors, prey_discriminator):
    clip_len, n_prey, neigh, feat = prey_tensors.shape
    prey_flat = prey_tensors.view(clip_len * n_prey, neigh, feat)
    prey_scores = prey_discriminator(prey_flat)
    return prey_scores.mean()


def gradient_estimate(theta, rewards_norm, dim, epsilons, sigma, lr, num_perturbations):
    grad = torch.zeros(dim, device=theta.device)
    for eps, reward in zip(epsilons, rewards_norm):
        grad += eps * reward
    theta_est = theta + (lr / (2 * sigma**2 * num_perturbations)) * grad
    return theta_est


def make_exp_states(batch, agent, neigh):
    random_states = torch.rand(batch, agent, neigh, 4) * 2 - 1

    dx  = random_states[..., 0].mean(-1)
    rvx = random_states[..., 2].mean(-1)
    actions = torch.sigmoid(12 * dx + 4 * rvx)

    actions = actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, neigh, 1)

    expert_states = torch.cat([random_states, actions], dim=-1)
    shaped = expert_states.reshape(batch * agent, neigh, 5)
    return shaped


def make_gen_states(batch=100, agent=32, neigh=31, prey_policy=None, deterministic=True):

    states = torch.rand(batch * agent, neigh, 4) * 2 - 1
    prey_states = states[..., :4]

    prey_actions = prey_policy.forward(prey_states, deterministic=deterministic)
    prey_actions = prey_actions.view(prey_states.shape[0], 1, 1).repeat(1, neigh, 1)

    expert_states = torch.cat([prey_states, prey_actions], dim=-1)
    shaped_states = expert_states.reshape(batch * agent, neigh, 5)

    return shaped_states.float()


def get_test_rollouts(prey_policy=None, clip_length=30, fixed_states=None, deterministic=True):
    prey_tensors = []

    for i in range(clip_length):
        states = fixed_states[i]

        prey_states = states[..., :4]

        prey_actions = prey_policy.forward(prey_states, deterministic=deterministic)
        prey_actions = prey_actions.view(prey_states.shape[0], 1, 1).repeat(1, prey_states.shape[1], 1)

        expert_states = torch.cat([prey_states, prey_actions], dim=-1)
        prey_tensors.append(expert_states)

    prey_tensor = torch.stack(prey_tensors, dim=0)
    return prey_tensor.float()


def apply_perturbation(args):
    (theta, eps, prey_policy, prey_disc, pert_clip_length, module_name, fixed_states, deterministic) = args

    module = getattr(prey_policy, module_name)

    nn.utils.vector_to_parameters(theta + eps, module.parameters())
    states_pos = get_test_rollouts(prey_policy, clip_length=pert_clip_length, fixed_states=fixed_states, deterministic=deterministic)
    #r_pos = discriminator_reward(states_pos, prey_disc)
    r_pos = fake_reward(states_pos)

    nn.utils.vector_to_parameters(theta - eps, module.parameters())
    states_neg = get_test_rollouts(prey_policy, clip_length=pert_clip_length, fixed_states=fixed_states, deterministic=deterministic)
    #r_neg = discriminator_reward(states_neg, prey_disc)
    r_neg = fake_reward(states_neg)

    nn.utils.vector_to_parameters(theta, module.parameters())

    prey_reward_diff = (r_pos - r_neg).detach().item()
    return prey_reward_diff



def fake_reward(prey_tensors):
    # prey_tensors: (clip_len, n_prey, neigh, 5)  -> last dim: (x,y,vx,vy,a)
    clip_len, n_prey, neigh, feat = prey_tensors.shape
    prey_flat = prey_tensors.view(clip_len * n_prey, neigh, feat)

    states = prey_flat[..., :4]          # (B, neigh, 4)
    a_star = f_target(states)            # (B,)
    a_pred = prey_flat[:, 0, 4]          # (B,) action is repeated over neigh

    mse = (a_pred - a_star).pow(2).mean()
    return -mse   


def f_target(states):  # states: (B, neigh, 4)
    dx  = states[..., 0].mean(dim=-1)
    rvx = states[..., 2].mean(dim=-1)
    a = torch.sigmoid(12 * dx + 4 * rvx)
    return a  # (B,)

def eval_policy(prey_policy, B=4096, neigh=31, device="cpu"):
    states = (torch.rand(B, neigh, 4, device=device) * 2 - 1)
    a_star = f_target(states)

    out = prey_policy.forward(states)
    a_pred = out[0] if isinstance(out, (tuple, list)) else out   # <- ggf. Index anpassen!
    a_pred = a_pred.view(-1)

    err = a_pred - a_star
    mae = err.abs().mean().item()
    mse = (err**2).mean().item()

    # correlation
    xs = a_star - a_star.mean()
    ys = a_pred - a_pred.mean()
    corr = (xs*ys).mean() / (xs.std(unbiased=False)*ys.std(unbiased=False) + 1e-8)
    corr = corr.item()

    return {"MAE": mae, "MSE": mse, "corr": corr}