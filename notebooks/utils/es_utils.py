import torch
import torch.nn as nn
from utils.env_utils import *
from marl_aquarium.aquarium_v0 import parallel_env


def discriminator_reward(pred_tensors, prey_tensors, pred_discriminator, prey_discriminator, eps=1e-8):
    clip_len, n_pred, neigh, feat = pred_tensors.shape
    pred_flat = pred_tensors.view(clip_len * n_pred, neigh, feat)

    _, n_prey, _, _ = prey_tensors.shape
    prey_flat = prey_tensors.view(clip_len * n_prey, neigh, feat)

    pred_scores = pred_discriminator(pred_flat)
    prey_scores = prey_discriminator(prey_flat)

    return pred_scores.mean(), prey_scores.mean()


def gradient_estimate(theta, rewards_norm, dim, epsilons, sigma, lr, num_pertubations):
    grad = torch.zeros(dim, device=theta.device)
    for eps, r in zip(epsilons, rewards_norm):
        grad += eps * r

    theta_est = theta + (lr /  (2 * sigma**2 * num_pertubations)) * grad
    return theta_est


def apply_perturbation(args):
    (module, theta, eps, pred_count, prey_count, action_count, pred_policy, prey_policy, pred_disc, prey_disc, role, pert_clip_length, use_walls, start_frame_pool) = args
    env = parallel_env(predator_count=pred_count, prey_count=prey_count, action_count=action_count, use_walls=use_walls)
    positions = start_frame_pool.sample(n=1)

    if role == "predator":
        #positive
        env.reset(options=positions)
        nn.utils.vector_to_parameters(theta + eps, module.parameters())
        states_pred_pos, _ = parallel_get_rollouts(env, pred_count, prey_count, action_count, pred_policy, prey_policy, clip_length=pert_clip_length)
        reward_pred_pos, _ = discriminator_reward(states_pred_pos, _, pred_disc, prey_disc)

        # negative
        env.reset(options=positions)
        nn.utils.vector_to_parameters(theta - eps, module.parameters())
        states_pred_neg, _ = parallel_get_rollouts(env, pred_count, prey_count, action_count, pred_policy, prey_policy, clip_length=pert_clip_length)
        reward_pred_neg, _ = discriminator_reward(states_pred_neg, _, pred_disc, prey_disc)

        # reset
        nn.utils.vector_to_parameters(theta, module.parameters())

        pred_reward_diff = (reward_pred_pos - reward_pred_neg).detach().item()

        return pred_reward_diff

    else:
        # positive
        env.reset(options=positions)
        nn.utils.vector_to_parameters(theta + eps, module.parameters())
        _, states_prey_pos = parallel_get_rollouts(env, pred_count, prey_count, action_count, pred_policy, prey_policy, clip_length=pert_clip_length)
        _, reward_prey_pos = discriminator_reward(_, states_prey_pos, pred_disc, prey_disc)

        # negative
        env.reset(options=positions)
        nn.utils.vector_to_parameters(theta - eps, module.parameters())
        _, states_prey_neg = parallel_get_rollouts(env, pred_count, prey_count, action_count, pred_policy, prey_policy, clip_length=pert_clip_length)
        _, reward_prey_neg = discriminator_reward(_, states_prey_neg, pred_disc, prey_disc)

        # reset
        nn.utils.vector_to_parameters(theta, module.parameters())

        prey_reward_diff = (reward_prey_pos - reward_prey_neg).detach().item()

        return prey_reward_diff

