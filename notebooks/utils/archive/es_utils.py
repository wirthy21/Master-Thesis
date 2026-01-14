import torch
import torch.nn as nn
from utils.env_utils import *
import custom_marl_aquarium
from marl_aquarium.aquarium_v0 import parallel_env


def discriminator_reward(prey_tensors, prey_discriminator):
    clip_len, n_prey, neigh, feat = prey_tensors.shape
    prey_flat = prey_tensors.view(clip_len * n_prey, neigh, feat)

    prey_scores = prey_discriminator(prey_flat)

    return prey_scores.mean()


def gradient_estimate(theta, rewards_norm, dim, epsilons, sigma, lr, num_perturbations):
    grad = torch.zeros(dim, device=theta.device)
    for eps, r in zip(epsilons, rewards_norm):
        grad += eps * r

    theta_est = theta + (lr /  (2 * sigma**2 * num_perturbations)) * grad
    return theta_est


def apply_perturbation(args):
    (module, theta, eps, prey_policy, prey_disc, pert_clip_length, use_walls, start_frame_pool) = args
    env = parallel_env(use_walls=use_walls)
    positions = start_frame_pool.sample(n=1)

    # positive
    env.reset(options=positions)
    nn.utils.vector_to_parameters(theta + eps, module.parameters())
    states_prey_pos = parallel_get_rollouts(env, prey_policy, clip_length=pert_clip_length)
    reward_prey_pos = discriminator_reward(states_prey_pos, prey_disc)

    # negative
    env.reset(options=positions)
    nn.utils.vector_to_parameters(theta - eps, module.parameters())
    states_prey_neg = parallel_get_rollouts(env, prey_policy, clip_length=pert_clip_length)
    reward_prey_neg = discriminator_reward(states_prey_neg, prey_disc)

    # reset
    nn.utils.vector_to_parameters(theta, module.parameters())

    prey_reward_diff = (reward_prey_pos - reward_prey_neg).detach().item()

    return prey_reward_diff

