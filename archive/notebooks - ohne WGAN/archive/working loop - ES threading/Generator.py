import os
import torch
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Function
from torch.distributions import Normal
from marl_aquarium import aquarium_v0
from utils.OpenAI_ES import *
from utils.env_utils import get_rollouts
from multiprocessing import Pool, set_start_method


class PairwiseInteraction(nn.Module):
    def __init__(self):
        super(PairwiseInteraction, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = Function.relu(self.fc1(x))
        x = Function.relu(self.fc2(x))
        x = self.fc3(x)

        mu, sigma_logit = x[:, :1], x[:, 1:]
        sigma = Function.softplus(sigma_logit) + 1e-6  # softplus always > 0, but with small values division by 0 possible (=added term).
        return mu, sigma
    
    

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 30)
        self.f3_prey = nn.Linear(30, 1) # seperated attention for prey and predator
        self.f3_pred = nn.Linear(30, 1)

    def forward(self, prey_states, pred_states=None, role="prey"):
        w_prey = Function.relu(self.fc1(prey_states))
        w_prey = Function.relu(self.fc2(w_prey))
        w_prey = self.f3_prey(w_prey)

        if role == "prey":
            w_pred = Function.relu(self.fc1(pred_states))
            w_pred = Function.relu(self.fc2(w_pred))
            w_pred = self.f3_pred(w_pred)
            return w_prey, w_pred
        else:
            return w_prey

    

class GeneratorPolicy(nn.Module):
    def __init__(self):
        super(GeneratorPolicy, self).__init__()
        self.pairwise = PairwiseInteraction()
        self.attention = Attention()
        

    def forward_prey(self, states):
        # PREY
        seq, agent, neigh, feat = states.shape
        prey_states = states[:, :, 1:, :]
        n_prey = prey_states.size(2)
        prey_flat = prey_states.reshape(seq * agent * n_prey, feat)

        #pairwise prey
        mu_prey, sigma_prey = self.pairwise(prey_flat)
        dist_prey = Normal(mu_prey, sigma_prey)
        raw_prey = dist_prey.rsample()
        a_prey_flat = Function.tanh(raw_prey) * math.pi
        a_prey = a_prey_flat.view(seq, agent, n_prey)

        # attention prey
        h_prey = Function.relu(self.attention.fc1(prey_flat))
        h_prey = Function.relu(self.attention.fc2(h_prey))
        w_prey_flat = self.attention.f3_prey(h_prey)
        w_prey = w_prey_flat.view(seq, agent, n_prey)

        ##############################################

        # PREDATOR
        pred_states = states[:, :, 0, :]
        pred_flat = pred_states.reshape(seq * agent, feat)

        # pairwise predator
        mu_pred, sigma_pred = self.pairwise(pred_flat)
        dist_pred = Normal(mu_pred, sigma_pred)
        raw_pred = dist_pred.rsample()
        a_pred_flat = Function.tanh(raw_pred) * math.pi
        a_pred = a_pred_flat.view(seq, agent)

        # attention predator
        h_pred = Function.relu(self.attention.fc1(pred_flat))
        h_pred = Function.relu(self.attention.fc2(h_pred))
        w_pred_flat = self.attention.f3_pred(h_pred)
        w_pred = w_pred_flat.view(seq, agent, 1)

        ##############################################

        # action and weights calculation
        logits = torch.cat([w_pred, w_prey], dim=2)
        weights = Function.softmax(logits, dim=2)
        w_pred_col = weights[:, :, 0]
        w_prey_mat = weights[:, :, 1:]

        action_prey = (a_prey * w_prey_mat).sum(dim=2) + a_pred * w_pred_col

        return action_prey
        

    def forward_pred(self, states):
        seq, agent, neigh, feat = states.shape
        states_flat = states.reshape(seq * agent * neigh, feat)

        mu, sigma = self.pairwise(states_flat)
        dist = Normal(mu, sigma)
        raw = dist.rsample()
        ai = (torch.tanh(raw) * math.pi).view(seq, agent, neigh)

        w_flat = self.attention(states_flat, None, role="pred")
        weights = torch.softmax(w_flat.view(seq, agent, neigh), dim=2)

        action_pred = (ai * weights).sum(dim=2)
        return action_pred

    # ref: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    def set_parameters(self, init=True):
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


    def update(self, role, network, env,
            pred_policy, prey_policy,
            pred_discriminator, prey_discriminator, 
            num_generations, num_perturbations,
            lr_pred_policy, lr_prey_policy,
            sigma, gamma):

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

        for gen in range(num_generations):
            # sample perturbations
            epsilons = [torch.randn(dim, device=theta.device) * sigma for _ in range(num_perturbations)]

            tasks = [(module, theta, eps,
                    pred_policy, prey_policy,
                    pred_discriminator, prey_discriminator)
                    for eps in epsilons]

            results = pool.map(apply_perturbation, tasks)

            # unpack into arrays
            reward_diffs = [r[0] for r in results]
            stacked_diffs = torch.stack(reward_diffs)
            pos_rewards  = [r[1] for r in results]
            neg_rewards  = [r[2] for r in results]

            # normalize diffs and compute gradient estimate
            normed = normalize(reward_diffs)
            theta_new = gradient_estimate(theta, normed, dim,
                                        epsilons, sigma, lr, num_perturbations)

            grad_norm = (theta_new - theta).norm().item()
            metrics.append({
                "generation": gen,
                "avg_reward_diff": stacked_diffs.mean().item(),
                "best_positive_reward": max(pos_rewards),
                "sigma": sigma,
                "lr": lr,
                "grad_norm": grad_norm
            })

            # update
            nn.utils.vector_to_parameters(theta_new, module.parameters())
            theta = theta_new.detach().clone()
            sigma *= gamma
            lr    *= gamma

        pool.close()
        pool.join()
        return metrics