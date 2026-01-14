import sys, os
sys.path.insert(0, os.path.abspath('..'))

import math
import torch
import random
import scipy.stats
import torch.nn as nn
#from utils.es_utils import *
from utils.es_utils_testing import *
from utils.env_utils import *
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing import Pool, set_start_method
from models.ModularNetworks import PairwiseInteraction, Attention


class PreyOnlyPolicy(nn.Module):
    def __init__(self, features=4):
        super(PreyOnlyPolicy, self).__init__()

        self.prey_pairwise = PairwiseInteraction(features)
        self.prey_attention = Attention(features)

    def forward_alt(self, states, deterministic=False):
        agents, neigh, feat = states.shape                  # Shape: (32,32,4)
        prey_states_flat = states.reshape(agents * neigh, feat)

        mu_prey, sigma_prey = self.prey_pairwise(prey_states_flat)          # mu=32*31, simga=32*31
        mu_mat = mu_prey.view(agents, neigh)
        mu_agent = mu_mat.mean(dim=1)  
        action = torch.sigmoid(mu_agent)  
        return action
    

    def forward(self, states, deterministic=True):
        agents, neigh, feat = states.shape                  # Shape: (32,32,4)
        prey_states_flat = states.reshape(agents * neigh, feat)

        mu_prey, sigma_prey = self.prey_pairwise(prey_states_flat)          # mu=32*31, simga=32*31
        sampled_prey_action = Normal(mu_prey, sigma_prey).rsample()          # actions=32*31
        
        if deterministic:
            prey_actions = torch.sigmoid(mu_prey).view(agents, neigh, 1)    # [A, N-1, 1]
        else:
            prey_actions = torch.sigmoid(sampled_prey_action).view(agents, neigh, 1)    # [A, N-1, 1]
        
        prey_weight_logits = self.prey_attention(prey_states_flat)
        prey_weight_logits = prey_weight_logits.view(agents, neigh)       # [A, N-1]
        prey_weights = torch.softmax(prey_weight_logits, dim=1).view(agents, neigh, 1)

        prey_actions_nei = prey_actions.squeeze(-1)
        prey_weights_nei = prey_weights.squeeze(-1)
        weighted_prey_actions = (prey_actions_nei * prey_weights_nei).sum(dim=1)

        if random.random() < 0.005:
            print("[POLICY FORWARD]")
            print("  mu     :", mu_prey.min().item(), mu_prey.max().item())
            print("  sigma  :", sigma_prey.min().item(), sigma_prey.max().item())
            print("  weights:", prey_weights.min().item(), prey_weights.max().item())
            print("  action :", prey_actions.min().item(), prey_actions.max().item())

        return weighted_prey_actions
    

    def update_single(self, network, policy, discriminator, num_perturbations, generation, lr, sigma, clip_length, use_walls, start_frame_pool):

        if network == "prey_pairwise":
            module = self.prey_pairwise
        elif network == "prey_attention":
            module = self.prey_attention

        
    


    def update(self, role, network, prey_policy, prey_discriminator, num_perturbations, generation, lr, sigma, clip_length, use_walls, start_frame_pool):

        if network == "prey_pairwise":
            module = self.prey_pairwise
        elif network == "prey_attention":
            module = self.prey_attention

        theta = nn.utils.parameters_to_vector(module.parameters()).detach().clone()
        dim = theta.numel()

        # Use spawn safely
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

        pool = Pool(processes=min(num_perturbations, os.cpu_count()))
        metrics = []

        # σϵi​ ∼ N(0,σ2I) like Wu
        epsilons = [torch.randn(dim, device=theta.device) * sigma for _ in range(num_perturbations)]

        #(module, theta, eps, prey_policy, prey_disc, pert_clip_length, use_walls, start_frame_pool)
        tasks = [(module, theta, eps, prey_policy, prey_discriminator, clip_length, use_walls, start_frame_pool) for eps in epsilons]

        results = pool.map(apply_perturbation, tasks)

        stacked_results = torch.tensor(results, device=theta.device)

        print("  reward diff stats:")
        print("    min / max :", stacked_results.min().item(), stacked_results.max().item())
        print("    mean / std:", stacked_results.mean().item(), stacked_results.std().item())

        ranked_diffs = scipy.stats.rankdata(stacked_results)
        diff_min = stacked_results.min().item()
        diff_max = stacked_results.max().item()
        diff_mean = stacked_results.mean().item()
        diff_std  = stacked_results.std().item()
        normed = (ranked_diffs - diff_mean) / (diff_std + 1e-8) # normalize to stabilise variance

        print("  normed rewards:")
        print("    min / max :", normed.min().item(), normed.max().item())
        print("    mean / std:", normed.mean().item(), normed.std().item())

        theta_new = gradient_estimate(theta, normed, dim, epsilons, sigma, lr, num_perturbations)

        grad_norm = (theta_new - theta).norm().item()

        delta = theta_new - theta

        print(f"[GEN {generation}] {network}")
        print("  theta norm       :", theta.norm().item())
        print("  delta norm       :", delta.norm().item())
        print("  lr / sigma^2     :", lr / (sigma**2))
        print("  max |delta|      :", delta.abs().max().item())
        print("  mean |delta|     :", delta.abs().mean().item())

        metrics.append({"generation": generation,
                        "reward_min": diff_min,
                        "reward_max": diff_max,
                        "reward_mean": diff_mean,
                        "reward_std": diff_std,
                        "sigma": sigma,
                        "lr": lr,
                        "grad_norm": grad_norm})

        # update
        nn.utils.vector_to_parameters(theta_new, module.parameters())
        theta = theta_new.detach().clone()
        
        pool.close()
        pool.join()
        return metrics
    

    def update_testing(self, network, prey_policy, prey_discriminator, num_perturbations, generation, lr, sigma, clip_length, deterministic=True, policy_prey_batch=None):

        if network == "prey_pairwise":
            module = self.prey_pairwise
        elif network == "prey_attention":
            module = self.prey_attention

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

        neigh = policy_prey_batch.shape[1]
        need = clip_length * 32

        flat = policy_prey_batch[:need]                      # (clip_len*32, neigh, 5)
        fixed_states = flat.view(clip_length, 32, neigh, 5)[..., :4] \
                        .detach().cpu().contiguous()
        module_name = network  # "prey_pairwise" oder "prey_attention"
        tasks = [(theta, eps, prey_policy, prey_discriminator, clip_length, module_name, fixed_states, deterministic) for eps in epsilons]

        results = pool.map(apply_perturbation, tasks)

        stacked_results = torch.tensor(results, device=theta.device)

        print("  reward diff stats:")
        print("    min / max :", stacked_results.min().item(), stacked_results.max().item())
        print("    mean / std:", stacked_results.mean().item(), stacked_results.std().item())

        ranks = torch.tensor(scipy.stats.rankdata(stacked_results, method="average"), device=theta.device, dtype=torch.float32)
        diff_min = stacked_results.min().item()
        diff_max = stacked_results.max().item()
        diff_mean = stacked_results.mean().item()
        diff_std  = stacked_results.std().item()
        #normed = (ranks - 0.5 * (len(ranks) + 1)) / len(ranks)
        normed = (stacked_results - stacked_results.mean()) / (stacked_results.std() + 1e-8)

        print("  normed rewards:")
        print("    min / max :", normed.min().item(), normed.max().item())
        print("    mean / std:", normed.mean().item(), normed.std().item())

        theta_new = gradient_estimate(theta, normed, dim, epsilons, sigma, lr, num_perturbations)

        grad_norm = (theta_new - theta).norm().item()

        delta = theta_new - theta

        print(f"[GEN {generation}] {network}")
        print("  theta norm       :", theta.norm().item())
        print("  delta norm       :", delta.norm().item())
        print("  lr / sigma^2     :", lr / (sigma**2))
        print("  max |delta|      :", delta.abs().max().item())
        print("  mean |delta|     :", delta.abs().mean().item())

        metrics.append({"generation": generation,
                        "reward_min": diff_min,
                        "reward_max": diff_max,
                        "reward_mean": diff_mean,
                        "reward_std": diff_std,
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



