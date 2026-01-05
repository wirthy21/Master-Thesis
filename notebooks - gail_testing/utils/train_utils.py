import os
import copy
import time
import torch
import numpy as np
import pandas as pd
from torch import autograd
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.eval_utils import *
from torch.utils.data import TensorDataset, DataLoader, random_split


# ref: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead/

def gradient_penalty(discriminator, expert_traj, generated_traj):
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
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)


def compute_wasserstein_loss(expert_scores, policy_scores, lambda_gp, gp):
    loss = policy_scores.mean() - expert_scores.mean()
    loss_gp = loss + lambda_gp * gp
    return loss, loss_gp


def run_prey_policy(env, prey_policy, steps=30):

    metrics = []

    for frame in range(steps):
        global_state = env.state().item()
        prey_tensor, xs, ys, vxs, vys = get_eval_features(global_state)

        prey_states = prey_tensor[..., :4]
        prey_headings = prey_tensor[:, 0, 4]
        prey_headings_list = prey_headings.tolist()
        action_prey = prey_policy.forward(prey_states)
        dis_prey = continuous_to_discrete(action_prey, 360)

        # Action dictionary
        action_dict = {}
        for i, agent_name in enumerate(sorted([agent for agent in env.agents if agent.startswith("prey")])):
            action_dict[agent_name] = dis_prey[i]

        env.step(action_dict)

        # Log metrics
        metrics.append({
            "polarization": compute_polarization(vxs, vys),
            "angular_momentum": compute_angular_momentum(xs, ys, vxs, vys),
            "degree_of_sparsity": degree_of_sparsity(xs, ys),
            "distance_to_predator": distance_to_predator(xs, ys),
            "escape_alignment": escape_alignment(xs, ys, vxs, vys)})

    try:
        env.close()
    except:
        pass

    polarization_mean = np.mean([m["polarization"] for m in metrics])
    angular_momentum_mean = np.mean([m["angular_momentum"] for m in metrics])

    return polarization_mean, angular_momentum_mean


def save_models(path,
                prey_policy,
                prey_discriminator,
                optim_dis_prey,
                expert_buffer, generative_buffer,
                dis_metrics_prey,
                es_metrics_prey):

    save_dir = os.path.join(path, "final_output")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(prey_policy, os.path.join(save_dir, "prey_policy.pt"))

    torch.save(prey_discriminator, os.path.join(save_dir, "prey_disc.pt"))

    torch.save(optim_dis_prey, os.path.join(save_dir, "optim_prey.pt"))

    dis_metrics_prey = pd.DataFrame(dis_metrics_prey)

    dis_metrics_prey.to_csv(os.path.join(save_dir, "dis_metrics_prey.csv"), index=False)

    es_metrics_prey = pd.DataFrame(es_metrics_prey)

    es_metrics_prey.to_csv(os.path.join(save_dir, "es_metrics_prey.csv"), index=False)

    expert_buffer.save(save_dir, type="expert")
    generative_buffer.save(save_dir, type="generative")

    print("Models successfully saved!")
    print("Training done!")


def save_checkpoint(path, epoch,
                    prey_policy,
                    prey_discriminator,
                    optim_dis_prey,
                    expert_buffer, generative_buffer,
                    dis_metrics_prey,
                    es_metrics_prey):

    ckpt_path = os.path.join(path, "ckpt")
    ckpt_save = os.path.join(ckpt_path, f"ckpt_epoch{epoch+1:03d}")
    os.makedirs(ckpt_save, exist_ok=True)

    torch.save(prey_policy, os.path.join(ckpt_save, "prey_policy.pt"))

    torch.save(prey_discriminator, os.path.join(ckpt_save, "prey_disc.pt"))

    torch.save(optim_dis_prey, os.path.join(ckpt_save, "optim_prey.pt"))

    generative_buffer.save(ckpt_save, type="generative")

    torch.save(dis_metrics_prey, os.path.join(ckpt_save, "dis_metrics_prey.pt"))

    torch.save(es_metrics_prey, os.path.join(ckpt_save, "es_metrics_prey.pt"))

    print("Checkpoint successfully saved! \n ")
    

def remaining_time(num_generations, last_epoch_duration, current_generation):
    remaining_epochs = num_generations - current_generation - 1
    remaining = remaining_epochs * last_epoch_duration
    finish_ts = time.time() + remaining

    finish_struct = time.localtime(finish_ts)
    estimated_time = time.strftime("%d.%m.%Y %H:%M:%S", finish_struct)

    min = int(last_epoch_duration // 60)
    sec = int(last_epoch_duration % 60)
    epoch_str = f"{min}:{sec:02d}"
    return estimated_time, epoch_str
    

class EarlyStoppingWasserstein:
    def __init__(self, patience=10, start_es=50):
        self.patience = patience
        self.start_es = start_es
        self.best_dist = float("inf")
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, loss, generation, role="prey"):
        if generation < self.start_es:
            return False

        dist = abs(loss)

        if self.best_dist == float("inf"):
            self.best_dist = dist
            return False

        if dist < self.best_dist:
            self.best_dist = dist
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"[{role.upper()}] Early stopping triggered: No improvement for {self.patience} epochs.")
                self.early_stop = True

        return self.early_stop