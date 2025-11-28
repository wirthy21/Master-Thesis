import os
import copy
import time
import torch
import pandas as pd
from torch import autograd
from utils.env_utils import *
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        

# Nutzt Jensen–Shannon‑Divergenz‑Loss
def compute_discriminator_loss(expert_scores, policy_scores, lambda_gp, gp, label_smoothing=False, smooth=0.1):
    #clipping
    clipped_expert_scores = torch.clamp(expert_scores, min=1e-8, max=1 - 1e-8) #necessary to avoid log(0) (and log(1-1))
    clipped_policy_scores = torch.clamp(policy_scores, min=1e-8, max=1 - 1e-8)

    if label_smoothing:
        y_exp = 1.0 - smooth
        y_pol = smooth

        loss_expert = -(y_exp * torch.log(clipped_expert_scores) + (1 - y_exp) * torch.log(1 - clipped_expert_scores)).mean()
        loss_policy = -(y_pol * torch.log(clipped_policy_scores) + (1 - y_pol) * torch.log(1 - clipped_policy_scores)).mean()
    else:
        loss_expert = -torch.log(clipped_expert_scores).mean()
        loss_policy = -torch.log(1 - clipped_policy_scores).mean()

    penalty = lambda_gp * gp

    discriminator_loss = loss_expert + loss_policy + penalty
    return discriminator_loss


def compute_wasserstein_loss(expert_scores, policy_scores, lambda_gp, gp):
    # Label-Smoothing not nessary for WGAN-GP
    loss = policy_scores.mean() - expert_scores.mean() + lambda_gp * gp
    return loss


def save_models(path,
                pred_policy, prey_policy,
                pred_discriminator, prey_discriminator,
                optim_dis_pred, optim_dis_prey,
                expert_buffer, generative_buffer,
                dis_metrics_pred, dis_metrics_prey,
                es_metrics_pred, es_metrics_prey):

    save_dir = os.path.join(path, "final_output")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(pred_policy, os.path.join(save_dir, "pred_policy.pt"))
    torch.save(prey_policy, os.path.join(save_dir, "prey_policy.pt"))

    torch.save(pred_discriminator, os.path.join(save_dir, "pred_dis.pt"))
    torch.save(prey_discriminator, os.path.join(save_dir, "prey_dis.pt"))

    torch.save(optim_dis_pred, os.path.join(save_dir, "optim_pred.pt"))
    torch.save(optim_dis_prey, os.path.join(save_dir, "optim_prey.pt"))

    dis_metrics_pred = pd.DataFrame(dis_metrics_pred)
    dis_metrics_prey = pd.DataFrame(dis_metrics_prey)

    dis_metrics_pred.to_csv(os.path.join(save_dir, "dis_metrics_pred.csv"), index=False)
    dis_metrics_prey.to_csv(os.path.join(save_dir, "dis_metrics_prey.csv"), index=False)

    es_metrics_pred = pd.DataFrame(es_metrics_pred)
    es_metrics_prey = pd.DataFrame(es_metrics_prey)

    es_metrics_pred.to_csv(os.path.join(save_dir, "es_metrics_pred.csv"), index=False)
    es_metrics_prey.to_csv(os.path.join(save_dir, "es_metrics_prey.csv"), index=False)

    expert_buffer.save(save_dir, type="expert")
    generative_buffer.save(save_dir, type="generative")

    print("Models successfully saved!")
    print("Training done!")


def save_checkpoint(path, epoch,
                    pred_policy, prey_policy,
                    pred_discriminator, prey_discriminator,
                    optim_dis_pred, optim_dis_prey,
                    expert_buffer, generative_buffer,
                    dis_metrics_pred, dis_metrics_prey,
                    es_metrics_pred, es_metrics_prey):

    ckpt_path = os.path.join(path, "checkpoints")
    ckpt_save = os.path.join(ckpt_path, f"ckpt_epoch{epoch+1:04d}")
    os.makedirs(ckpt_save, exist_ok=True)

    torch.save(pred_policy, os.path.join(ckpt_save, "pred_policy.pt"))
    torch.save(prey_policy, os.path.join(ckpt_save, "prey_policy.pt"))

    torch.save(pred_discriminator, os.path.join(ckpt_save, "pred_discriminator.pt"))
    torch.save(prey_discriminator, os.path.join(ckpt_save, "prey_discriminator.pt"))

    torch.save(optim_dis_pred, os.path.join(ckpt_save, "optim_pred.pt"))
    torch.save(optim_dis_prey, os.path.join(ckpt_save, "optim_prey.pt"))

    expert_buffer.save(ckpt_save, type="expert")
    generative_buffer.save(ckpt_save, type="generative")

    torch.save(dis_metrics_pred, os.path.join(ckpt_save, "dis_metrics_pred.pt"))
    torch.save(dis_metrics_prey, os.path.join(ckpt_save, "dis_metrics_prey.pt"))

    torch.save(es_metrics_pred, os.path.join(ckpt_save, "es_metrics_pred.pt"))
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

    def __call__(self, loss, generation, role="predator"):
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



def pretrain_policy(policy, expert_buffer, role, pred_bs=32, prey_bs=256, epochs=10, lr=1e-3, device='cpu', save_dir=None):

    if role == 'predator':
        batch, _ = expert_buffer.sample(pred_bs, prey_bs)
    else:
        _, batch = expert_buffer.sample(pred_bs, prey_bs)

    states = batch[..., :4]
    actions = batch[:, 0, 4].to(device)

    ds = TensorDataset(states, actions)
    bs = pred_bs if role=='predator' else prey_bs
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    pretrain_log = []

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)    # (bs,neigh,4)
            batch_actions = batch_actions.to(device)

            if role == 'predator':
                actions_pred, mu_pred, sigma_pred, weights_pred = policy.forward(batch_states)
                loss = F.mse_loss(actions_pred, batch_actions)
            else:
                actions_prey, mu_prey, sigma_prey, weights_prey, pred_gain = policy.forward(batch_states, weights_pred)
                loss = F.mse_loss(actions_prey, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)
            pretrain_log.append({"epoch": epoch, "loss": loss.item()})

    pretrain_dir = os.path.join(save_dir, "pretraining")
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(policy, os.path.join(pretrain_dir, f"pretrained_policy_{role}.pt"))
    torch.save(pretrain_log, os.path.join(pretrain_dir, f"pretrain_log_{role}.pt"))

    return policy



def pretrain_policy_with_validation(policy, pred_policy=None, expert_buffer=None, role=None, val_ratio=0.2, pred_bs=256, prey_bs=512, epochs=10, lr=1e-3, device='cpu', early_stopping=True, patience=20):
    if role == 'predator':
        batch, _ = expert_buffer.sample(pred_bs, prey_bs)
    else:
        _, batch = expert_buffer.sample(pred_bs, prey_bs)

    states  = batch[..., :4]
    actions = batch[:, 0, 4].squeeze()

    dataset = TensorDataset(states, actions)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    bs = pred_bs if role=='predator' else prey_bs
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # Early-Stopping-Tracking
    patience_counter = 0
    best_val_loss = float('inf')
    best_state = copy.deepcopy(policy.state_dict())

    # Logs
    if role == 'predator':
        logs = {"role": "predator", "mu_pred": [], "sigma_pred": [], "weights_pred": []}
    else:
        logs = {"role": "prey", "mu_prey": [], "sigma_prey": [], "weights_prey": [], "pred_gain": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        policy.train()
        total_train_loss = 0.0

        for batch_states, batch_actions in train_loader:
            batch_states  = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            if role == 'predator':
                actions_pred, mu_pred, sigma_pred, weights_pred = policy.forward(batch_states)
                loss = F.mse_loss(actions_pred, batch_actions)
            else:
                actions_pred, mu_pred, sigma_pred, weights_pred = pred_policy.forward(batch_states)
                actions_prey, mu_prey, sigma_prey, weights_prey, pred_gain = policy.forward(batch_states, weights_pred)
                print(pred_gain)
                loss = F.mse_loss(actions_prey, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_states.size(0)

        avg_train_loss = total_train_loss / train_size
        train_losses.append(avg_train_loss)

        if role == 'predator':
            logs["mu_pred"].append(mu_pred.detach().cpu().numpy())
            logs["sigma_pred"].append(sigma_pred.detach().cpu().numpy())
            logs["weights_pred"].append(weights_pred.detach().cpu().numpy())
        else:
            logs["mu_prey"].append(mu_prey.detach().cpu().numpy())
            logs["sigma_prey"].append(sigma_prey.detach().cpu().numpy())
            logs["weights_prey"].append(weights_prey.detach().cpu().numpy())
            logs["pred_gain"].append(pred_gain.detach().cpu().numpy())

        # ---- Validation ----
        policy.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states  = batch_states.to(device)
                batch_actions = batch_actions.to(device)

                if role == 'predator':
                    actions_pred, mu_pred, sigma_pred, weights_pred = policy.forward(batch_states)
                    loss = F.mse_loss(actions_pred, batch_actions)
                else:
                    actions_pred, mu_pred, sigma_pred, weights_pred = pred_policy.forward(batch_states)
                    actions_prey, mu_prey, sigma_prey, weights_prey, pred_gain_val = policy.forward(batch_states, weights_pred)
                    loss = F.mse_loss(actions_prey, batch_actions)

                total_val_loss += loss.item() * batch_states.size(0)

        avg_val_loss = total_val_loss / val_size
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(policy.state_dict())

        if role == 'predator':
            print(f"[{role.upper()}] Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"[{role.upper()}] Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Pred Gain: {pred_gain[0]}")
            
        if early_stopping:
            if avg_val_loss > avg_train_loss:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                print(f"[{role.upper()}] Early stopping triggered (Patience={patience}).")
                break

    policy.load_state_dict(best_state)

    # Plot
    epochs_run = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs_run + 1), train_losses, label='Train Loss', color='#005555', linewidth=2)
    plt.plot(range(1, epochs_run + 1), val_losses, label='Val Loss', color='#A7A7A8', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f"{role.capitalize()} Loss Curves")
    plt.legend()
    plt.show()

    return policy, logs