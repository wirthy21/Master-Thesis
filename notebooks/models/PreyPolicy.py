import sys, os
sys.path.insert(0, os.path.abspath('..'))

import math
import torch
import scipy.stats
import torch.nn as nn
from utils.es_utils import *
from utils.env_utils import *
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing import Pool, set_start_method
from models.ModularNetworks import PairwiseInteraction, Attention, PredatorInteraction


class PreyPolicy(nn.Module):
    def __init__(self, features=4, gain=0.0):
        super(PreyPolicy, self).__init__()

        self.prey_pairwise = PairwiseInteraction(features)
        self.prey_attention = Attention(features)

        self.pred_pairwise = PredatorInteraction(features)

    def forward(self, states, pred_attention_weights=None):
        agents, neigh, feat = states.shape                  # Shape: (32,32,4)

        device = states.device
        dtype  = states.dtype

        ##### Predator #####
        pred_states = states[:, 0, :]                               # Shape: (32,4)
        mu_pred, sigma_pred = self.pred_pairwise(pred_states)       # mu=32, simga=32
        sampled_pred_action = Normal(mu_pred, sigma_pred).sample()  # actions=32
        pred_actions = torch.tanh(sampled_pred_action) * math.pi    # Value Range [-pi:pi]
        pred_action_flat = pred_actions.squeeze(-1)

        if pred_attention_weights is not None:
            pred_gain = pred_attention_weights.view(agents)
        else:
            pred_gain = torch.full((agents,), 1/33, device=states.device, dtype=states.dtype) # treat every action equal

        ##### Prey #####
        prey_states = states[:, 1:, :]                                      # Shape: (32,31,4)
        prey_states_flat   = prey_states.reshape(agents * (neigh-1), feat)  # Shape: (32*31,4)

        mu_prey, sigma_prey = self.prey_pairwise(prey_states_flat)          # mu=32*31, simga=32*31
        sampled_prey_action = Normal(mu_prey, sigma_prey).rsample()          # actions=32*31
        prey_actions = (torch.tanh(sampled_prey_action) * math.pi).view(agents, neigh - 1, 1)

        prey_weight_logits = self.prey_attention(prey_states_flat)
        prey_weight_logits = prey_weight_logits.view(agents, neigh-1)       # [A, N-1]
        prey_weights = torch.softmax(prey_weight_logits, dim=1).view(agents, neigh-1, 1)

        ##### Action Aggregation #####

        # Aggregation of Prey Actions per Prey
        prey_actions_nei = prey_actions.squeeze(-1)
        prey_weights_nei = prey_weights.squeeze(-1)
        prey_action_per_prey = (prey_actions_nei * prey_weights_nei).sum(dim=1)

        final_action = pred_gain * pred_action_flat + (1.0 - pred_gain) * prey_action_per_prey

        ##### Logging #####

        mu_full = torch.zeros(agents, neigh, 1, device=device, dtype=dtype)
        sigma_full = torch.zeros_like(mu_full)
        weights_full = torch.zeros_like(mu_full)

        # Predator index 0
        mu_full[:, 0, :] = mu_pred 
        sigma_full[:, 0, :] = sigma_pred 
        weights_full[:, 0, :] = 1 # pred_weight always 1, due to 1:1 relationship

        # Prey indices 1..
        mu_full[:, 1:, :] = mu_prey.view(agents, neigh-1, 1)
        sigma_full[:, 1:, :] = sigma_prey.view(agents, neigh-1, 1)
        weights_full[:, 1:, :] = prey_weights

        # fürs Logging: [A, N]
        mu_log = mu_full.squeeze(-1)
        sigma_log = sigma_full.squeeze(-1)
        weights_log = weights_full.squeeze(-1)

        #if torch.rand(1).item() < 0.002:
        #    print("\n[DEBUG|PreyPolicy.forward]")
        #    print(f"  final_action mean/std: {final_action.mean().item():.3f} / {final_action.std().item():.3f}")
        #    print(f"  prey_action_per_prey mean/std: {prey_action_per_prey.mean().item():.3f} / {prey_action_per_prey.std().item():.3f}")
        #    print(f"  mu_prey mean/std: {mu_prey.mean().item():.3f} / {mu_prey.std().item():.3f}")
        #    print(f"  sigma_prey mean/std: {sigma_prey.mean().item():.3f} / {sigma_prey.std().item():.3f}")
        #    print(f"  prey_weights min/max: {prey_weights.min().item():.3f} / {prey_weights.max().item():.3f}")
        #    print(f"  pred_gain min/mean/max: {pred_gain.min().item():.3f} / {pred_gain.mean().item():.3f} / {pred_gain.max().item():.3f}")


        return final_action, prey_action_per_prey, mu_log, sigma_log, weights_log, pred_gain


    def update(self, role, network,
               pred_policy, prey_policy,
               pred_discriminator, prey_discriminator, 
               num_perturbations, generation, lr,
               sigma, clip_length, use_walls, start_frame_pool):

        if network == "prey_pairwise":
            module = self.prey_pairwise
        elif network == "prey_attention":
            module = self.prey_attention
        elif network == "pred_pairwise":
            module = self.pred_pairwise

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

        tasks = [(module, theta, eps, pred_policy, prey_policy, pred_discriminator, prey_discriminator, 
                  role, clip_length, use_walls, start_frame_pool) for eps in epsilons]

        results = pool.map(apply_perturbation, tasks)

        stacked_results = torch.tensor(results, device=theta.device)
        ranked_diffs = scipy.stats.rankdata(stacked_results)
        diff_min = stacked_results.min().item()
        diff_max = stacked_results.max().item()
        diff_mean = stacked_results.mean().item()
        diff_std  = stacked_results.std().item()
        normed = (ranked_diffs - diff_mean) / (diff_std + 1e-8) # normalize to stabilise variance

        theta_new = gradient_estimate(theta, normed, dim, epsilons, sigma, lr, num_perturbations)

        grad_norm = (theta_new - theta).norm().item()

        metrics.append({"generation": generation,
                        "reward_min": diff_min,
                        "reward_max": diff_max,
                        "reward_mean": diff_mean,
                        "reward_std": diff_std,
                        "sigma": sigma,
                        "lr": lr,
                        "grad_norm": grad_norm})

        #if generation % 5 == 0:
        #    print(f"\n[DEBUG|{role}|{network}] grad_norm: {grad_norm:.3f} | r_mean: {diff_mean:.3f} | r_std:  {diff_std:.3f} | r_max:  {diff_max:.3f}, r_min:  {diff_min:.3f}")

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