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


class PairwiseInteraction(nn.Module):
    def __init__(self):
        super(PairwiseInteraction, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        mu, sigma_logit = x[:, :1], x[:, 1:]
        sigma = F.softplus(sigma_logit) + 1e-6  # softplus always > 0, but with small values division by 0 possible (=added term).
        return mu, sigma
    
    

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 30)
        self.f3_prey = nn.Linear(30, 1) # seperated attention for prey and predator
        self.f3_pred = nn.Linear(30, 1)

    def forward(self, prey_states, pred_states=None, role="prey"):
        w_prey = F.relu(self.fc1(prey_states))
        w_prey = F.relu(self.fc2(w_prey))
        w_prey = self.f3_prey(w_prey)

        if role == "prey":
            w_pred = F.relu(self.fc1(pred_states))
            w_pred = F.relu(self.fc2(w_pred))
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
        states = states.float()
        # PREY
        agent, neigh, feat = states.shape
        prey_states = states[:, 1:, :]
        n_prey = prey_states.size(1)
        prey_flat = prey_states.reshape(agent * n_prey, feat)

        #pairwise prey
        mu_prey, sigma_prey = self.pairwise(prey_flat)
        dist_prey = Normal(mu_prey, sigma_prey)
        raw_prey = dist_prey.rsample()
        a_prey_flat = F.tanh(raw_prey) * math.pi
        a_prey = a_prey_flat.view(agent, n_prey)

        # attention prey
        h_prey = F.relu(self.attention.fc1(prey_flat))
        h_prey = F.relu(self.attention.fc2(h_prey))
        w_prey_flat = self.attention.f3_prey(h_prey)
        w_prey = w_prey_flat.view(agent, n_prey)

        ##############################################

        # PREDATOR
        pred_states = states[:, 0, :]
        pred_flat = pred_states.reshape(agent, feat)

        # pairwise predator
        mu_pred, sigma_pred = self.pairwise(pred_flat)
        dist_pred = Normal(mu_pred, sigma_pred)
        raw_pred = dist_pred.rsample()
        a_pred_flat = F.tanh(raw_pred) * math.pi
        a_pred = a_pred_flat.view(agent)

        # attention predator
        h_pred = F.relu(self.attention.fc1(pred_flat))
        h_pred = F.relu(self.attention.fc2(h_pred))
        w_pred_flat = self.attention.f3_pred(h_pred)
        w_pred = w_pred_flat.view(agent, 1)

        ##############################################

        # action and weights calculation
        logits = torch.cat([w_pred, w_prey], dim=1)
        weights = F.softmax(logits, dim=1)
        w_pred_col = weights[:, 0]
        w_prey_mat = weights[:, 1:]

        action_prey = (a_prey * w_prey_mat).sum(dim=1) + a_pred * w_pred_col

        return action_prey
        

    def forward_pred(self, states):
        states = states.float()
        agent, neigh, feat = states.shape
        states_flat = states.reshape(agent * neigh, feat)

        mu, sigma = self.pairwise(states_flat)
        dist = Normal(mu, sigma)
        raw = dist.rsample()
        ai = (torch.tanh(raw) * math.pi).view(agent, neigh) #(1, 32)

        w_flat = self.attention(states_flat, None, role="pred")
        weights = torch.softmax(w_flat.view(agent, neigh), dim=1)

        action_pred = (ai * weights).sum(dim=1)
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
        sigma *= gamma
        lr    *= gamma

        pool.close()
        pool.join()
        return metrics