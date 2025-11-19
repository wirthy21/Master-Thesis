import sys, os
sys.path.insert(0, os.path.abspath('..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing import Pool, set_start_method
from models.ModularNetworks import PairwiseInteraction, Attention


class PreyPolicy(nn.Module):
    def __init__(self, features=4, gain=0.0):
        super(PreyPolicy, self).__init__()
        self.pred_gain = nn.Parameter(torch.tensor(gain))

        self.prey_pairwise = PairwiseInteraction(features)
        self.prey_attention = Attention(features)

        self.pred_pairwise = PairwiseInteraction(features)
        self.pred_attention = Attention(features)

    def forward(self, states):
        agents, neigh, feat = states.shape                  # Shape: (32,32,4)

        device = states.device
        dtype  = states.dtype

        ##### Predator #####
        pred_states = states[:, 0, :]                               # Shape: (32,1,4)
        mu_pred, sigma_pred = self.pred_pairwise(pred_states)       # mu=32, simga=32
        sampled_pred_action = Normal(mu_pred, sigma_pred).sample()  # actions=32
        pred_actions = torch.tanh(sampled_pred_action) * math.pi    # Value Range [-pi:pi]

        pred_weight_logits = self.pred_attention(pred_states)       # weights=32
        pred_weights = torch.softmax(pred_weight_logits, dim=1)          

        ##### Prey #####
        prey_states = states[:, 1:, :]                                      # Shape: (32,31,4)
        prey_states_flat   = prey_states.reshape(agents * (neigh-1), feat)  # Shape: (32*31,4)

        mu_prey, sigma_prey = self.prey_pairwise(prey_states_flat)          # mu=32*31, simga=32*31
        sampled_prey_action = Normal(mu_prey, sigma_prey).sample()          # actions=32*31
        prey_actions = (torch.tanh(sampled_prey_action) * math.pi).view(agents, neigh - 1, 1)

        prey_weight_logits = self.prey_attention(prey_states_flat)
        prey_weight_logits = prey_weight_logits.view(agents, neigh-1)       # [A, N-1]
        prey_weights = torch.softmax(prey_weight_logits, dim=1).view(agents, neigh-1, 1)

        ##### Action Aggregation #####

        # Ensure gain is positive
        pred_gain = torch.sigmoid(self.pred_gain)

        # Aggregation of Predator Actions per Prey
        pred_action_per_prey = (pred_actions * pred_weights).sum(dim=1)

        # Aggregation of Prey Actions per Prey
        prey_actions_nei = prey_actions.squeeze(-1)
        prey_weights_nei = prey_weights.squeeze(-1)
        prey_action_per_prey = (prey_actions_nei * prey_weights_nei).sum(dim=1)

        final_action = pred_gain * pred_action_per_prey + (1.0 - pred_gain) * prey_action_per_prey

        ##### Logging #####

        mu_full = torch.zeros(agents, neigh, 1, device=device, dtype=dtype)
        sigma_full = torch.zeros_like(mu_full)
        weights_full = torch.zeros_like(mu_full)

        # Predator index 0
        mu_full[:, 0, :] = mu_pred 
        sigma_full[:, 0, :] = sigma_pred 
        weights_full[:, 0, :] = pred_weights

        # Prey indices 1..
        mu_full[:, 1:, :] = mu_prey.view(agents, neigh-1, 1)
        sigma_full[:, 1:, :] = sigma_prey.view(agents, neigh-1, 1)
        weights_full[:, 1:, :] = prey_weights

        # fürs Logging: [A, N]
        mu_log = mu_full.squeeze(-1)
        sigma_log = sigma_full.squeeze(-1)
        weights_log = weights_full.squeeze(-1)

        return final_action, mu_log, sigma_log, weights_log, pred_gain


    def update(self, role, network,
            pred_count, prey_count, action_count,
            pred_policy, prey_policy,
            pred_discriminator, prey_discriminator, 
            num_perturbations, generation,
            lr_pred_policy, lr_prey_policy,
            sigma, gamma, clip_length, use_walls, start_frame_pool):

        if role == "predator":
            module = self.pred_pairwise if network == "pairwise" else self.pred_attention
            lr = lr_pred_policy
        else:
            module = self.prey_pairwise if network == "pairwise" else self.prey_attention
            lr = lr_prey_policy

        theta = nn.utils.parameters_to_vector(module.parameters()).detach().clone()
        dim = theta.numel()

        # Use spawn safely
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

        pool = Pool(processes=min(num_perturbations, os.cpu_count()))
        metrics = []

        epsilons = [torch.randn(dim, device=theta.device) * sigma for _ in range(num_perturbations)]

        tasks = [(module, theta, eps,
                pred_count, prey_count, action_count,
                pred_policy, prey_policy,
                pred_discriminator, prey_discriminator, role,
                clip_length, use_walls, start_frame_pool)
                for eps in epsilons]

        results = pool.map(apply_perturbation, tasks)

        stacked_diffs = torch.tensor(results, device=theta.device)
        diff_min = stacked_diffs.min().item()
        diff_max = stacked_diffs.max().item()
        mean = stacked_diffs.mean()
        std  = stacked_diffs.std(unbiased=False) + 1e-8
        normed = (stacked_diffs - mean) / std

        theta_new = gradient_estimate(theta, normed, dim, epsilons, sigma, lr, num_perturbations)

        grad_norm = (theta_new - theta).norm().item()

        metrics.append({"generation": generation,
                        "avg_reward_diff": mean.item(),
                        "diff_min": diff_min,
                        "diff_max": diff_max,
                        "diff_std": std.item(),
                        "sigma": sigma,
                        "lr": lr,
                        "grad_norm": grad_norm})

        # update
        nn.utils.vector_to_parameters(theta_new, module.parameters())
        theta = theta_new.detach().clone()
        
        pool.close()
        pool.join()
        return metrics
    

    # ref: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    def set_parameters(self, init=True):
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()