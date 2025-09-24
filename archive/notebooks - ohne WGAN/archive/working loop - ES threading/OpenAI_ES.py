import numpy as np
import torch
import torch.nn as nn
from utils.env_utils import get_rollouts
from marl_aquarium import aquarium_v0


def discriminator_reward(pred_tensors, prey_tensors, pred_discriminator, prey_discriminator):
    pred_scores = pred_discriminator.forward(pred_tensors)
    prey_scores = prey_discriminator.forward(prey_tensors)
    reward_pred = torch.log(pred_scores + 1e-8) - torch.log(1 - pred_scores + 1e-8)
    reward_prey = torch.log(prey_scores + 1e-8) - torch.log(1 - prey_scores + 1e-8)
    return reward_pred.mean(), reward_prey.mean()


def perturbation(self, theta, eps, sign="positive", network="pairwise"):
    if network == "pairwise":
        if sign == "positive":
            nn.utils.vector_to_parameters(theta + eps, self.pairwise.parameters())
        else:
            nn.utils.vector_to_parameters(theta - eps, self.pairwise.parameters())

    else:
        if sign == "positive":
            nn.utils.vector_to_parameters(theta + eps, self.attention.parameters())
        else:
            nn.utils.vector_to_parameters(theta - eps, self.attention.parameters())


def normalize(rewards_diff):
    reward_stacked = torch.stack(rewards_diff)
    reward_normalized = (reward_stacked - reward_stacked.mean()) / (reward_stacked.std() + 1e-8)
    return reward_normalized


def gradient_estimate(theta, rewards_norm, dim, epsilons, sigma, lr, num_pertubations):
    grad = torch.zeros(dim)
    for eps, r in zip(epsilons, rewards_norm):
        grad += eps * r

    theta_est = theta + (lr /  (2 * sigma**2 * num_pertubations)) * grad
    return theta_est


def apply_perturbation(args):
    (module, theta, eps, pred_policy, prey_policy, pred_disc, prey_disc) = args
     
    env = aquarium_v0.env(predator_count=1,prey_count=32,action_count=360)

    # positive
    nn.utils.vector_to_parameters(theta + eps, module.parameters())
    states_p, _ = get_rollouts(env, pred_policy, prey_policy, num_frames=9)
    r_pos, _ = discriminator_reward(states_p, _, pred_disc, prey_disc)
    pos = r_pos.mean().item()

    # negative
    nn.utils.vector_to_parameters(theta - eps, module.parameters())
    states_n, _ = get_rollouts(env, pred_policy, prey_policy, num_frames=9)
    r_neg, _ = discriminator_reward(states_n, _, pred_disc, prey_disc)
    neg = r_neg.mean().item()

    # reset
    nn.utils.vector_to_parameters(theta, module.parameters())
    return (r_pos - r_neg).detach(), pos, neg

