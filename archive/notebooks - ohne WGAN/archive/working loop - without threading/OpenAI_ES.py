import numpy as np
import torch
import torch.nn as nn


def discriminator_reward(pred_tensors, prey_tensors, pred_discriminator, prey_discriminator):
    pred_scores = pred_discriminator.forward(pred_tensors)
    prey_scores = prey_discriminator.forward(prey_tensors)
    reward_pred = torch.log(pred_scores + 1e-8) - torch.log(1 - pred_scores + 1e-8)
    reward_prey = torch.log(prey_scores + 1e-8) - torch.log(1 - prey_scores + 1e-8)
    return reward_pred.mean(), reward_prey.mean()


def pertubation(self, theta, eps, sign="positive", network="pairwise"):
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