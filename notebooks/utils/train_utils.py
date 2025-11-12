import os
import time
import torch
import pandas as pd
from torch import autograd
from utils.env_utils import *
from collections import deque
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
    

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_metric = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, avg_es, role="predator"):
        if self.best_metric is None:
            self.best_metric = avg_es
            return False

        if avg_es > self.best_metric:
            self.best_metric = avg_es
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f"[{role.upper()}] Early stopping triggered after {self.patience} epochs.")
                self.early_stop = True

        return self.early_stop


class EarlyStoppingWindow:
    def __init__(self, patience=10, window_size=10, min_slope=0.0):
        from collections import deque
        self.window = window_size
        self.patience = patience
        self.min_slope = min_slope
        self.queue = deque(maxlen=window_size)
        self.bad_windows = 0
        self.early_stop = False

    def _slope(self, values):
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0

    def __call__(self, metric, role="predator"):
        self.queue.append(metric)
        if len(self.queue) < self.window:
            return False

        slope = self._slope(self.queue)

        if slope <= self.min_slope:
            self.bad_windows += 1
        else:
            self.bad_windows = 0

        if self.bad_windows >= self.patience:
            print(f"[{role.upper()}] Early stopping triggered after {self.patience} epochs.")
            self.early_stop = True

        return self.early_stop
    

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
                est_action = policy.forward_pred(batch_states)
            else:
                est_action = policy.forward_prey(batch_states)

            loss = F.mse_loss(est_action, batch_actions)
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