import numpy as np
import torch
import torch.nn as nn
from marl_aquarium import aquarium_v0
from .env_utils import get_rollouts

def discriminator_reward(prob, eps):
    return torch.log(prob + eps) - torch.log(1 - prob + eps)


def pertubation(env, pred_policy, prey_policy, pred_discriminator, prey_discriminator, theta_pred, theta_prey, eps, sign="positive"):
    randomized_seed = np.random.randint(0, 10000) #protect overfitting to a specific seed
    
    if sign == "positive":
        nn.utils.vector_to_parameters(theta_pred + eps, pred_policy.parameters())
        nn.utils.vector_to_parameters(theta_prey + eps, prey_policy.parameters())
        env.reset(seed=randomized_seed)
        pred_tensors, prey_tensors = get_rollouts(env, pred_policy, prey_policy, num_frames=9, render=False)
        pred_scores = pred_discriminator.forward(pred_tensors)
        prey_scores = prey_discriminator.forward(prey_tensors)
        reward_pred = discriminator_reward(pred_scores, eps)
        reward_prey = discriminator_reward(prey_scores, eps)
        return reward_pred, reward_prey
    else:
        nn.utils.vector_to_parameters(theta_pred - eps, pred_policy.parameters())
        nn.utils.vector_to_parameters(theta_prey - eps, prey_policy.parameters())
        env.reset(seed=randomized_seed)
        pred_tensors, prey_tensors = get_rollouts(env, pred_policy, prey_policy, num_frames=9, render=False)
        pred_scores = pred_discriminator.forward(pred_tensors)
        prey_scores = prey_discriminator.forward(prey_tensors)
        reward_pred = discriminator_reward(pred_scores, eps)
        reward_prey = discriminator_reward(prey_scores, eps)
        return reward_pred, reward_prey


def normalize(rewards_diff):
    reward_stacked = torch.stack(rewards_diff)
    reward_normalized = (reward_stacked - reward_stacked.mean()) / (reward_stacked.std() + 1e-8)
    return reward_normalized


def gradient_estimate(theta_pred, theta_prey, rewards_norm_pred, rewards_norm_prey, dim, epsilons, sigma, lr_pred, lr_prey, num_pertubations):
    grad_pred = torch.zeros(dim)
    grad_prey = torch.zeros(dim)
    for eps, rp, ry in zip(epsilons, rewards_norm_pred, rewards_norm_prey):
        grad_pred += eps * rp
        grad_prey += eps * ry

    theta_est_pred = theta_pred + (lr_pred /  (2 * sigma**2 * num_pertubations)) * grad_pred
    theta_est_prey = theta_prey + (lr_prey / (2 * sigma**2 * num_pertubations)) * grad_prey
    return theta_est_pred, theta_est_prey


def openai_es(env, pred_policy, prey_policy, pred_discriminator, prey_discriminator, num_generations, num_pertubations, sigma, lr_pred, lr_prey, gamma):
    theta_pred = nn.utils.parameters_to_vector(pred_policy.parameters())
    theta_prey = nn.utils.parameters_to_vector(prey_policy.parameters())
    dim = theta_pred.numel() #same like prey

    for generation in range(num_generations):
        epsilons = []
        rewards_diff_pred = []
        rewards_diff_prey = []

        for i in range(num_pertubations):
            eps = torch.randn(dim) * sigma

            epsilons.append(eps)

            # positive perturbation
            reward_pred_pos, reward_prey_pos = pertubation(env, 
                                                           pred_policy, prey_policy, 
                                                           pred_discriminator, prey_discriminator, 
                                                           theta_pred, theta_prey, 
                                                           eps, sign="positive")

            # negative perturbation
            reward_pred_neg, reward_prey_neg = pertubation(env, 
                                                           pred_policy, prey_policy, 
                                                           pred_discriminator, prey_discriminator, 
                                                           theta_pred, theta_prey, 
                                                           eps, sign="negative")

            # save reward differences
            rewards_diff_pred.append((reward_pred_pos - reward_pred_neg).detach())
            rewards_diff_prey.append((reward_prey_pos - reward_prey_neg).detach())

        # reset inital parameters
        nn.utils.vector_to_parameters(theta_pred, pred_policy.parameters())
        nn.utils.vector_to_parameters(theta_prey, prey_policy.parameters())
            
        rewards_norm_pred = normalize(rewards_diff_pred)
        rewards_norm_prey = normalize(rewards_diff_prey)

        theta_est_pred, theta_est_prey = gradient_estimate(theta_pred, theta_prey, 
                                                           rewards_norm_pred, rewards_norm_prey, 
                                                           dim, epsilons, sigma, lr_pred, lr_prey, num_pertubations)

        # apply new parameters to the policies
        theta_pred = theta_est_pred.clone()
        theta_prey = theta_est_prey.clone()
        nn.utils.vector_to_parameters(theta_pred, pred_policy.parameters())
        nn.utils.vector_to_parameters(theta_prey, prey_policy.parameters())

        print(f"Generation: {generation} | Sigma: {sigma:.4f} | lr_pred: {lr_pred:.4f} | lr_prey: {lr_prey:.4f}")

        sigma *= gamma
        lr_pred *= gamma
        lr_prey *= gamma

    return theta_pred, theta_prey