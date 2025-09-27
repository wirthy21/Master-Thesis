import os
import math
import torch
import torch.nn as nn
from utils.es_utils import *
from utils.env_utils import *
from utils.train_utils import *
import torch.nn.functional as F
from torch.distributions import Normal
from multiprocessing import Pool, set_start_method

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseInteraction(nn.Module):
    def __init__(self, features):
        super(PairwiseInteraction, self).__init__()
        self.fc1 = nn.Linear(features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)  # mu, sigma

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        mu, sigma_logit = x[:, :1], x[:, 1:]
        sigma = F.softplus(sigma_logit) + 1e-6
        return mu, sigma


class Attention(nn.Module):
    def __init__(self, features):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)

        self.f6_prey = nn.Linear(32, 1)  # logits for prey-neighbor weights
        self.f6_pred = nn.Linear(32, 1)  # logits for predator weight(s)

    def forward(self, states, pred_states=None, role="prey"):
        w_prey = F.relu(self.fc1(states))
        w_prey = F.relu(self.fc2(w_prey))
        w_prey = F.relu(self.fc3(w_prey))
        w_prey = F.relu(self.fc4(w_prey))
        w_prey = F.relu(self.fc5(w_prey))
        w_prey = self.f6_prey(w_prey)  # (..., 1)

        if role == "prey":
            w_pred = F.relu(self.fc1(pred_states))
            w_pred = F.relu(self.fc2(w_pred))
            w_pred = F.relu(self.fc3(w_pred))
            w_pred = F.relu(self.fc4(w_pred))
            w_pred = F.relu(self.fc5(w_pred))
            w_pred = self.f6_pred(w_pred)  # (..., 1)
            return w_prey, w_pred
        else:
            return w_prey


    

class GeneratorPolicy(nn.Module):
    def __init__(self, features=4):
        super(GeneratorPolicy, self).__init__()
        self.features = features
        self.pairwise = PairwiseInteraction(features)
        self.attention = Attention(features)
        

    '''def forward_prey(self, states):
        states = states.float()
        agent, neigh, feat = states.shape

        # --- PREY ---
        prey_states = states[:, 1:, :]                       # (A, N_prey, F)
        n_prey = prey_states.size(1)
        prey_flat = prey_states.reshape(agent * n_prey, feat)  # (A*N_prey, F)

        # Pairwise für Prey
        mu_prey, sigma_prey = self.pairwise(prey_flat)       # (A*N_prey, 1) jeweils
        dist_prey = Normal(mu_prey, sigma_prey)
        raw_prey = dist_prey.rsample()
        a_prey_flat = torch.tanh(raw_prey) * math.pi         # (A*N_prey, 1)
        a_prey = a_prey_flat.view(agent, n_prey)             # (A, N_prey)

        # --- PREDATOR ---
        pred_states = states[:, 0, :]                        # (A, F)
        pred_flat = pred_states.reshape(agent, feat)

        # Pairwise für Predator
        mu_pred, sigma_pred = self.pairwise(pred_flat)       # (A, 1)
        dist_pred = Normal(mu_pred, sigma_pred)
        raw_pred = dist_pred.rsample()
        a_pred = (torch.tanh(raw_pred) * math.pi).view(agent)  # (A,)

        # --- Attention-Gewichte (nutzt die tiefe Attention.forward) ---
        # Gibt (A*N_prey,1) und (A,1) zurück
        w_prey_flat, w_pred_flat = self.attention(prey_flat, pred_flat, role="prey")
        w_prey = w_prey_flat.view(agent, n_prey)             # (A, N_prey)
        w_pred = w_pred_flat.view(agent, 1)                  # (A, 1)

        # --- Kombination ---
        logits = torch.cat([w_pred, w_prey], dim=1)          # (A, 1+N_prey)
        weights = F.softmax(logits, dim=1)                   # (A, 1+N_prey)
        w_pred_col = weights[:, 0]                           # (A,)
        w_prey_mat = weights[:, 1:]                          # (A, N_prey)

        action_prey = (a_prey * w_prey_mat).sum(dim=1) + a_pred * w_pred_col  # (A,)
        return action_prey'''

    def forward_prey(self, states):
        states = states.float()
        agent, neigh, feat = states.shape

        # --- PREY ---
        prey_states = states[:, 1:, :]                          # (A, N_prey, F)
        n_prey = prey_states.size(1)
        prey_flat = prey_states.reshape(agent * n_prey, feat)   # (A*N_prey, F)

        # Pairwise für Prey
        mu_prey, sigma_prey = self.pairwise(prey_flat)          # (A*N_prey, 1)
        dist_prey = Normal(mu_prey, sigma_prey)
        raw_prey = dist_prey.rsample()
        a_prey_flat = torch.tanh(raw_prey) * math.pi            # (A*N_prey, 1)
        a_prey = a_prey_flat.view(agent, n_prey)                # (A, N_prey)

        # --- PREDATOR ---
        pred_states = states[:, 0, :]                           # (A, F)
        pred_flat = pred_states.reshape(agent, feat)

        # Pairwise für Predator
        mu_pred, sigma_pred = self.pairwise(pred_flat)          # (A, 1)
        dist_pred = Normal(mu_pred, sigma_pred)
        raw_pred = dist_pred.rsample()
        a_pred = (torch.tanh(raw_pred) * math.pi).view(agent)   # (A,)

        # --- Attention-Gewichte ---
        w_prey_flat, w_pred_flat = self.attention(prey_flat, pred_flat, role="prey")
        w_prey = w_prey_flat.view(agent, n_prey)                # (A, N_prey)
        w_pred = w_pred_flat.view(agent, 1)                     # (A, 1)

        # --- Kombination ---
        logits = torch.cat([w_pred, w_prey], dim=1)             # (A, 1+N_prey)
        weights = F.softmax(logits, dim=1)                      # (A, 1+N_prey)
        w_pred_col = weights[:, 0]                              # (A,)
        w_prey_mat = weights[:, 1:]                             # (A, N_prey)

        action_prey = (a_prey * w_prey_mat).sum(dim=1) + a_pred * w_pred_col  # (A,)
        return action_prey



    def forward_pred(self, states):
        states = states.float()
        agent, neigh, feat = states.shape
        states_flat = states.reshape(agent * neigh, feat)

        # Pairwise
        mu, sigma = self.pairwise(states_flat)               # (A*N,1)
        dist = Normal(mu, sigma)
        raw = dist.rsample()
        ai = (torch.tanh(raw) * math.pi).view(agent, neigh)  # (A, N)

        # Attention (nur Prey-Gewichte nötig, Rolle != "prey")
        w_flat = self.attention(states_flat, None, role="pred")  # (A*N,1)
        weights = torch.softmax(w_flat.view(agent, neigh), dim=1)  # (A, N)

        action_pred = (ai * weights).sum(dim=1)              # (A,)
        return action_pred

    # ref: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    def set_parameters(self, init=True):
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


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