import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing import Pool, set_start_method
from ModularNetworks import PairwiseInteraction, Attention


class PredatorPolicy(nn.Module):
    def __init__(self, features=4):
        super(PredatorPolicy, self).__init__()
        self.features = features
        self.pairwise = PairwiseInteraction(features)
        self.attention = Attention(features)

    def forward_pred(self, states):
        agents, neigh, feat = states.shape                   # Shape: (1,32,4)
        states_flat = states.reshape(agents * neigh, feat)   # Shape: (32,4)

        # Sample Action from PI-Distribution
        mu, sigma = self.pairwise(states_flat)               # mu=32, simga=32
        sampled_action = Normal(mu, sigma).sample()
        actions = torch.tanh(sampled_action).view(agents, neigh) # Value Range [-1:1]

        # Attention Weights
        weight_logits = self.attention(states_flat).view(agents, neigh)
        weights = torch.softmax(weight_logits, dim=1)

        # Action Calculation
        action = (actions * weights).sum(dim=1)
        return action, mu, sigma, weights
    

    def update(self, role, network,
            pred_count, prey_count, action_count,
            pred_policy, prey_policy,
            pred_discriminator, prey_discriminator, 
            num_perturbations, generation,
            lr_pred_policy, lr_prey_policy,
            sigma, gamma, clip_length, use_walls, start_frame_pool):

        module = self.pairwise if network == "pairwise" else self.attention
        theta = nn.utils.parameters_to_vector(module.parameters()).detach().clone()
        dim = theta.numel()
        lr = lr_pred_policy if role == "predator" else lr_prey_policy

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

        theta_new = gradient_estimate(theta, normed, dim,
                                    epsilons, sigma, lr, num_perturbations)

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