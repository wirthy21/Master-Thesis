import os
import copy
import time
import torch
import pandas as pd
from torch import autograd
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split


# ref: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/

def gradient_penalty(discriminator, expert_traj, generated_traj):
        expert_traj = expert_traj.detach()
        generated_traj = generated_traj.detach()
        
        batch_size = expert_traj.size(0)

        eps_shape = [batch_size] + [1] * (expert_traj.dim() - 1)
        eps = torch.rand(*eps_shape, device=expert_traj.device)
        
        # Interpolation between real data and fake data.
        interpolation = eps * expert_traj + (1 - eps) * generated_traj
        interpolation.requires_grad_(True)
        
        # get logits for interpolated images
        interp_logits = discriminator(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=False,
            only_inputs=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
        


def compute_wasserstein_loss(expert_scores, policy_scores, lambda_gp, gp):
    loss = policy_scores.mean() - expert_scores.mean()
    loss_gp = lambda_gp * gp
    return loss, loss_gp