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
from utils.vec_sim_utils import *
from utils.encoder_utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split


def gradient_estimate(theta, rewards_norm, epsilons, sigma, lr, num_perturbations,
                      rel_clip=0.01, abs_clip=None, min_clip=1e-12, return_metrics=True):

    grad = torch.zeros_like(theta)
    for eps, reward in zip(epsilons, rewards_norm):
        grad += eps * reward

    delta_raw = (lr / (2 * sigma**2 * num_perturbations)) * grad  
    delta_raw_norm = delta_raw.norm().clamp_min(1e-12)        

    theta_norm = theta.norm().clamp_min(1e-12)
    max_delta_norm = rel_clip * theta_norm

    if abs_clip is not None:
        max_delta_norm = torch.minimum(max_delta_norm, theta.new_tensor(abs_clip))
    max_delta_norm = torch.maximum(max_delta_norm, theta.new_tensor(min_clip))

    clip_ratio = (max_delta_norm / delta_raw_norm).clamp(max=1.0)
    delta = delta_raw * clip_ratio 

    theta_new = theta + delta

    if return_metrics:
        return theta_new, {
            "theta_norm": float(theta_norm.item()),
            "delta_raw_norm": float(delta_raw_norm.item()),
            "delta_norm": float(delta.norm().item()),
            "max_delta_norm": float(max_delta_norm.item()),
            "clip_ratio": float(clip_ratio.item()),
        }

    return theta_new



def discriminator_reward(discriminator, gen_tensor, mode="mean"):
    matrix = discriminator(gen_tensor)  # expected: (B, F-1, A) or (B, F-1, A, ...)

    # ensure batch is dim0
    if matrix.ndim == 0:
        raise RuntimeError("Discriminator returned scalar. It must return at least (B, ...)")

    if mode == "mean":
        # mean over all dims except batch
        reduce_dims = tuple(range(1, matrix.ndim))
        scores = matrix.mean(dim=reduce_dims)  # (B,)
        return scores

    if mode == "top":
        k = 0.2
        flat = matrix.flatten(1)  # (B, *)
        K = max(1, int(flat.size(1) * k))
        topk_vals = flat.topk(K, dim=1).values  # (B, K)
        return topk_vals.mean(dim=1)            # (B,)

    if mode == "avoid":
        d0 = 0.08
        temp = 0.02

        dx = gen_tensor[:, :-1, :, :, 0]
        dy = gen_tensor[:, :-1, :, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)  # (B, F-1, A, N)

        if discriminator.role == "prey":
            pred_dist = dist[:, :, :, 0]            # (B, F-1, A)
        else:
            pred_dist = dist[:, :, :, 1:].amin(-1)  # (B, F-1, A)

        weight = torch.sigmoid((d0 - pred_dist) / temp) + 0.05  # (B, F-1, A)

        # bring matrix to (B, F-1, A) if it has extra dims
        if matrix.ndim > 3:
            matrix_reduced = matrix.mean(dim=tuple(range(3, matrix.ndim)))  # (B, F-1, A)
        else:
            matrix_reduced = matrix

        # weighted mean per batch: sum over (F-1,A), keep batch
        num = (matrix_reduced * weight).sum(dim=(1, 2))  # (B,)
        den = weight.sum(dim=(1, 2)).clamp_min(1e-12)    # (B,)
        return num / den                                 # (B,)

    raise ValueError(f"Unknown mode: {mode}")



def optimize_es(pred_policy, prey_policy, 
                role, module, mode,
                discriminator, lr, 
                sigma, num_perturbations, 
                init_pos=None, device="cuda"):
    
    if role == "prey":
        network = prey_policy.pairwise if module == 'pairwise' else prey_policy.attention
    else:
        network = pred_policy.pairwise if module == 'pairwise' else pred_policy.attention

    theta = nn.utils.parameters_to_vector(network.parameters())

    pred_rollouts, prey_rollouts, epsilons = apply_perturbations(prey_policy, pred_policy, init_pos,
                                role=role, module=module, device=device,
                                sigma=sigma, num_perturbations=num_perturbations)
    
    if role == "prey":
        reward = discriminator_reward(discriminator, prey_rollouts, mode="top")
    else:
        reward = discriminator_reward(discriminator, pred_rollouts, mode="top")

    reward_pos = reward[:num_perturbations]
    reward_neg = reward[num_perturbations:]

    diffs = (reward_pos - reward_neg).detach()
    ranks = torch.argsort(torch.argsort(diffs)).float()
    ranks_norm = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

    theta_est, grad_metrics = gradient_estimate(theta, ranks_norm, epsilons, sigma, lr, num_perturbations)

    # if std is too small, do not update (Random Walk)
    if diffs.std(unbiased=False) < 1e-6:
        theta_est = theta

    nn.utils.vector_to_parameters(theta_est, network.parameters())
    
    return {"diff_min": round(diffs.min().item(), 6),
            "diff_max": round(diffs.max().item(), 6),
            "diff_mean": round(diffs.mean().item(), 6),
            "diff_std": round(diffs.std(unbiased=False).item(), 6),
            "delta_norm": round((theta_est - theta).norm().item(), 6),
            "clip_ratio": round(grad_metrics["clip_ratio"], 6),
            "delta_raw_norm": round(grad_metrics["delta_raw_norm"], 6),
            "max_delta_norm": round(grad_metrics["max_delta_norm"], 6)
        }



def pretrain_policy(policy, role, expert_data, 
                     batch_size=256, epochs=250, 
                     lr=1e-3, deterministic=True, 
                     patience=10, device='cuda'):

    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n, frames, agents, neigh, features = expert_data.shape
    expert_data = expert_data.reshape(n * frames * agents, neigh, features)
    
    if role == "prey":
        states  = expert_data[..., :5]
        actions = expert_data[:, 0, 5]
    else:
        states  = expert_data[..., :4]
        actions = expert_data[:, 0, 4]

    # make sure actions are float for MSE
    actions = actions.to(torch.float32)

    dataset = TensorDataset(states, actions)
    val_size = int(0.2 * len(dataset))  # 80/20 split
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    bad_epochs = 0
    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        policy.train()
        epoch_train_loss = 0.0
        train_count = 0

        for states, actions in train_loader:
            states = states.to(device=device)
            exp_actions = actions.to(device=device)

            est_actions = policy.forward(states, deterministic=deterministic).squeeze(-1)
            loss = F.mse_loss(est_actions, exp_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = exp_actions.size(0)
            epoch_train_loss += loss.item() * bs
            train_count += bs
            
        epoch_train_loss = epoch_train_loss / max(1, train_count)
        train_losses.append(float(epoch_train_loss))

        # Validation
        policy.eval()
        epoch_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device=device)
                exp_actions = actions.to(device=device)

                est_actions = policy.forward(states, deterministic=True).squeeze(-1)
                loss = F.mse_loss(est_actions, exp_actions)

                bs = exp_actions.size(0)
                epoch_val_loss += loss.item() * bs
                val_count += bs

        epoch_val_loss = epoch_val_loss / max(1, val_count)
        val_losses.append(float(epoch_val_loss))

        # Early stopping
        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            best_state = copy.deepcopy(policy.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 25 == 0:
            print(f"[{role.upper()}] Epoch {epoch}/{epochs} | train_loss:{epoch_train_loss:.6f} | val_loss:{epoch_val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"[{role.upper()}] Early stopping at epoch {epoch}.")
            break

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"[{role.upper()}] Pretrain Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return policy.load_state_dict(best_state)


def calculate_metrics(pred_policy=None, prey_policy=None, init_pool=None, 
                      pred_encoder=None, prey_encoder=None,
                      exp_pred_tensor=None, exp_prey_tensor=None,
                      pred_mmd_loss=None, prey_mmd_loss=None,
                      sinkhorn_loss=None, device=None):

    gen_pred_tensor, gen_prey_tensor = run_env_vectorized(prey_policy=prey_policy, 
                                                          pred_policy=pred_policy, 
                                                          n_prey=32, n_pred=1, max_steps=100,
                                                          init_pool=init_pool)
    
    mmd_list = []
    sinkhorn_list = []
    for i in range(100):
        expert_prey_batch = sample_data(exp_prey_tensor, batch_size=10, window_len=10).to(device)
        generative_prey_batch = sample_data(gen_prey_tensor, batch_size=10, window_len=10).to(device)

        expert_pred_batch = sample_data(exp_pred_tensor, batch_size=20, window_len=10).to(device)
        generative_pred_batch = sample_data(gen_pred_tensor, batch_size=20, window_len=10).to(device)


        # MMD Calculation
        with torch.no_grad():
            mmd_prey_metric = prey_mmd_loss.forward(expert_prey_batch, generative_prey_batch)
            mmd_pred_metric = pred_mmd_loss.forward(expert_pred_batch, generative_pred_batch)
        mmd_list.append((mmd_prey_metric.item(), mmd_pred_metric.item()))


        # Sinkhorn Calculation
        _, trans_exp_pred = pred_encoder(expert_pred_batch[...,:4])
        _, trans_gen_pred = pred_encoder(generative_pred_batch[...,:4])
        batch, frames, agents, dim = trans_exp_pred.shape
        pred_x = trans_exp_pred.reshape(batch * frames, agents, dim)
        pred_y = trans_gen_pred.reshape(batch * frames, agents, dim)

        _, trans_exp_prey = prey_encoder(expert_prey_batch[...,:5])
        _, trans_gen_prey = prey_encoder(generative_prey_batch[...,:5])
        batch, frames, agents, dim = trans_exp_prey.shape
        prey_x = trans_exp_prey.reshape(batch * frames, agents, dim)
        prey_y = trans_gen_prey.reshape(batch * frames, agents, dim)

        sinkhorn_pred = sinkhorn_loss(pred_x, pred_y)
        sinkhorn_prey = sinkhorn_loss(prey_x, prey_y)
        sinkhorn_list.append((sinkhorn_prey.mean().item(), sinkhorn_pred.mean().item()))


    mmd_pred_mean = np.mean([mmd[1] for mmd in mmd_list])
    mmd_pred_std = np.std([mmd[1] for mmd in mmd_list], ddof=1)

    mmd_prey_mean = np.mean([mmd[0] for mmd in mmd_list])
    mmd_prey_std = np.std([mmd[0] for mmd in mmd_list], ddof=1)

    sinkhorn_pred_mean = np.mean([sinkhorn[1] for sinkhorn in sinkhorn_list])
    sinkhorn_pred_std = np.std([sinkhorn[1] for sinkhorn in sinkhorn_list], ddof=1)

    sinkhorn_prey_mean = np.mean([sinkhorn[0] for sinkhorn in sinkhorn_list])
    sinkhorn_prey_std = np.std([sinkhorn[0] for sinkhorn in sinkhorn_list], ddof=1)

    return {
        "mmd_prey_mean": mmd_prey_mean,
        "mmd_prey_std": mmd_prey_std,
        "mmd_pred_mean": mmd_pred_mean,
        "mmd_pred_std": mmd_pred_std,
        "sinkhorn_prey_mean": sinkhorn_prey_mean,
        "sinkhorn_prey_std": sinkhorn_prey_std,
        "sinkhorn_pred_mean": sinkhorn_pred_mean,
        "sinkhorn_pred_std": sinkhorn_pred_std
    }


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



def save_hyperparameter(save_dir,
                        max_steps=None,
                        num_generations=None,
                        gamma=None,
                        lr_policy=None,
                        num_perturbations=None,
                        sigma=None,
                        deterministic=None,
                        dis_balance_factor=None,
                        noise=None,
                        lr_disc=None,
                        lambda_gp=None,
                        performance_eval=None):
    path = os.path.join(save_dir, "hyperparameters.txt")
    with open(path, "w") as f:
        f.write("# === Hyperparameters ===\n\n")

        f.write(f"# Expert\n")
        f.write(f"max_steps = {max_steps}\n\n")

        f.write(f"# Training\n")
        f.write(f"num_generations = {num_generations}\n")
        f.write(f"gamma = {gamma}\n\n")

        f.write(f"# Policy\n")
        f.write(f"lr_policy = {lr_policy}\n")
        f.write(f"num_perturbations = {num_perturbations}\n")
        f.write(f"sigma = {sigma}\n")
        f.write(f"deterministic = {deterministic}\n\n")

        f.write(f"# Discriminator\n")
        f.write(f"dis_balance_factor = {dis_balance_factor}\n")
        f.write(f"noise = {noise}\n")
        f.write(f"lr_disc = {lr_disc}\n")
        f.write(f"lambda_gp = {lambda_gp}\n\n")

        f.write(f"performance_eval = {performance_eval}\n\n")

    return path
