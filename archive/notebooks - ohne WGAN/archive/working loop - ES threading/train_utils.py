import torch
from torch import autograd
import datetime
import os
import time
import pandas as pd

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
        

def compute_discriminator_loss(expert_scores, policy_scores, lambda_gp, gp):
    discriminator_loss = ((- torch.log(expert_scores).mean()) + (- torch.log(1 - policy_scores).mean()) + lambda_gp * gp)
    return discriminator_loss


def save_models(path,
                pred_policy, prey_policy,
                pred_discriminator, prey_discriminator,
                optim_dis_pred, optim_dis_prey,
                expert_buffer, generative_buffer,
                losses_pred_discriminator, losses_prey_discriminator,
                es_metrics_pred, es_metrics_prey):

    save_dir = os.path.join(path, "final_output")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(pred_policy, os.path.join(save_dir, "pred_policy.pt"))
    torch.save(prey_policy, os.path.join(save_dir, "prey_policy.pt"))

    torch.save(pred_discriminator, os.path.join(save_dir, "pred_dis.pt"))
    torch.save(prey_discriminator, os.path.join(save_dir, "prey_dis.pt"))

    torch.save(optim_dis_pred, os.path.join(save_dir, "optim_pred.pt"))
    torch.save(optim_dis_prey, os.path.join(save_dir, "optim_prey.pt"))

    losses = pd.DataFrame({'pred_discriminator': losses_pred_discriminator,
                       'prey_discriminator': losses_prey_discriminator})
    
    losses.to_csv('discriminator_losses.csv', index=False)

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
                    losses_pred_discriminator, losses_prey_discriminator,
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

    torch.save(losses_pred_discriminator, os.path.join(ckpt_save, "loss_pred.pt"))
    torch.save(losses_prey_discriminator, os.path.join(ckpt_save, "loss_prey.pt"))

    torch.save(es_metrics_pred, os.path.join(ckpt_save, "es_metrics_pred.pt"))
    torch.save(es_metrics_prey, os.path.join(ckpt_save, "es_metrics_prey.pt"))

    print("Checkpoint successfully saved! \n ")

def remaining_time(start_time, generations, i):
    epoch_time = (time.time() - start_time) / (i + 1)
    remaining = (generations - i - 1) * epoch_time
    finish_ts = time.time() + remaining
    finish_dt = datetime.datetime.fromtimestamp(finish_ts)
    finish_time = finish_dt.replace(microsecond=0)
    return finish_time
