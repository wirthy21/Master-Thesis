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
from models.ModularNetworks import PairwiseInteraction, Attention


class PredatorPolicy(nn.Module):
    def __init__(self, features=4):
        super(PredatorPolicy, self).__init__()
        self.pairwise = PairwiseInteraction(features)
        self.attention = Attention(features)

    def forward(self, states):
        agents, neigh, feat = states.shape                   # Shape: (1,32,4)
        states_flat = states.reshape(agents * neigh, feat)   # Shape: (32,4)

        # Sample Action from PI-Distribution
        mu, sigma = self.pairwise(states_flat)               # mu=32, simga=32
        sampled_action = Normal(mu, sigma).rsample()
        actions = (torch.tanh(sampled_action) * math.pi).view(agents, neigh) # Value Range [-pi:pi]

        # Attention Weights
        weight_logits = self.attention(states_flat).view(agents, neigh)
        weights = torch.softmax(weight_logits, dim=1)

        # Action Calculation
        action = (actions * weights).sum(dim=1)

        #if torch.rand(1).item() < 0.002:
        #    print("\n[DEBUG|PredPolicy.forward]")
        #    print(f"  actions mean/std: {actions.mean().item():.3f} / {actions.std().item():.3f}")
        #    print(f"  action(agg) mean/std: {action.mean().item():.3f}")
        #    print(f"  mu mean/std: {mu.mean().item():.3f} / {mu.std().item():.3f}")
        #    print(f"  sigma mean/std: {sigma.mean().item():.3f} / {sigma.std().item():.3f}")
        #    print(f"  weights min/max: {weights.min().item():.3f} / {weights.max().item():.3f}")

        return action, mu, sigma, weights
    

    def update(self, role, network,
            pred_policy, prey_policy,
            pred_discriminator, prey_discriminator, 
            num_perturbations, generation, lr,
            sigma, clip_length, use_walls, start_frame_pool):

        module = self.pairwise if network == "pairwise" else self.attention
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
        diff_std = stacked_results.std().item()
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