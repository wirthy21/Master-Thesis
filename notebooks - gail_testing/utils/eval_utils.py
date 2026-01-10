import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from geomloss import SamplesLoss
from utils.encoder_utils import *
from scipy.spatial import cKDTree

def get_expert_values(exp_pred_tensor=None, exp_prey_tensor=None, 
                      prey_mmd_loss=None, pred_mmd_loss=None, 
                      prey_encoder=None, pred_encoder=None, 
                      sinkhorn_loss=None):

    prey_mmd_list = []
    pred_mmd_list = []

    prey_sinkhorn_list = []
    pred_sinkhorn_list = []

    for i in tqdm(range(500)):
        expert_prey_batch1 = sample_data(exp_prey_tensor, batch_size=10, window_len=10)
        expert_prey_batch2 = sample_data(exp_prey_tensor, batch_size=10, window_len=10)

        prey_mmd = prey_mmd_loss(expert_prey_batch1, expert_prey_batch2).item()
        prey_mmd_list.append(prey_mmd)

        _, trans_exp_prey = prey_encoder(expert_prey_batch1[...,:5])
        _, trans_gen_prey = prey_encoder(expert_prey_batch2[...,:5])
        batch, frames, agents, dim = trans_exp_prey.shape
        prey_x = trans_exp_prey.reshape(batch * frames, agents, dim)
        prey_y = trans_gen_prey.reshape(batch * frames, agents, dim)
        sinkhorn_prey = sinkhorn_loss(prey_x, prey_y)
        prey_sinkhorn_list.append(sinkhorn_prey.mean().item())

        if exp_pred_tensor is not None:
            expert_pred_batch1 = sample_data(exp_pred_tensor, batch_size=10, window_len=10)
            expert_pred_batch2 = sample_data(exp_pred_tensor, batch_size=10, window_len=10)

            pred_mmd = pred_mmd_loss(expert_pred_batch1, expert_pred_batch2).item()
            pred_mmd_list.append(pred_mmd)

            _, trans_exp_pred = pred_encoder(expert_pred_batch1[...,:4])
            _, trans_gen_pred = pred_encoder(expert_pred_batch2[...,:4])
            batch, frames, agents, dim = trans_exp_pred.shape
            pred_x = trans_exp_pred.reshape(batch * frames, agents, dim)
            pred_y = trans_gen_pred.reshape(batch * frames, agents, dim)
            sinkhorn_pred = sinkhorn_loss(pred_x, pred_y)
            pred_sinkhorn_list.append(sinkhorn_pred.mean().item())

    print(f"\nExpert Prey MMD: {np.mean(prey_mmd_list)} ± {np.std(prey_mmd_list)}")
    print(f"Expert Prey Sinkhorn: {np.mean(prey_sinkhorn_list)} ± {np.std(prey_sinkhorn_list)}")
    
    if exp_pred_tensor is not None:
        print(f"\nExpert Pred MMD: {np.mean(pred_mmd_list)} ± {np.std(pred_mmd_list)}")
        print(f"Expert Pred Sinkhorn: {np.mean(pred_sinkhorn_list)} ± {np.std(pred_sinkhorn_list)}")

    mmd_means = (np.mean(prey_mmd_list), np.mean(pred_mmd_list) if exp_pred_tensor is not None else None)
    mmd_stds = (np.std(prey_mmd_list), np.std(pred_mmd_list) if exp_pred_tensor is not None else None)

    sinkhorn_means = (np.mean(prey_sinkhorn_list), np.mean(pred_sinkhorn_list) if exp_pred_tensor is not None else None)
    sinkhorn_stds = (np.std(prey_sinkhorn_list), np.std(pred_sinkhorn_list) if exp_pred_tensor is not None else None)
    
    return mmd_means, mmd_stds, sinkhorn_means, sinkhorn_stds



def plot_train_metrics(disc_metrics, dis_balance_factor, role="prey", save_dir=None):
    disc_dicts = [x[0] if role == "prey" else x[1] for x in disc_metrics]
    disc_df = pd.DataFrame(disc_dicts)
    disc_df["gen"] = (disc_df.index // dis_balance_factor) + 1
    disc_mean = disc_df.groupby("gen").mean(numeric_only=True).sort_index()

    gens = np.arange(len(disc_mean))

    score_diffs = (disc_mean["expert_score_mean"] - disc_mean["policy_score_mean"]).abs().to_numpy()
    abs_scores = disc_mean[["expert_score_mean", "policy_score_mean"]].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(gens, abs_scores[:, 0], label="Expert Score")
    axes[0].plot(gens, abs_scores[:, 1], label="Policy Score")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{role.upper()} Absolute Scores over Generations")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(gens, score_diffs, color="orange")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Score Difference")
    axes[1].set_title(f"{role.upper()} Score Difference over Generations")
    axes[1].grid(True)

    plt.tight_layout()

    plot_path = save_dir / f"disc_{role}_metrics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_eval_metrics(metrics_list, 
                      mmd_means=None, sinkhorn_means=None, 
                      mmd_stds=None, sinkhorn_stds=None,
                      role="prey", save_dir=None,
                      eval_steps=5, max_steps=1500):
    
    xs = np.arange(len(metrics_list)) * eval_steps

    if role == "prey":
        mmd_mean = np.array([m["mmd_prey_mean"] for m in metrics_list], dtype=float)
        mmd_std  = np.array([m["mmd_prey_std"]  for m in metrics_list], dtype=float)
        sk_mean  = np.array([m["sinkhorn_prey_mean"] for m in metrics_list], dtype=float)
        sk_std   = np.array([m["sinkhorn_prey_std"]  for m in metrics_list], dtype=float)
        exp_mmd_mean = mmd_means[0]
        exp_mmd_std = mmd_stds[0]
        exp_sinkhorn_mean = sinkhorn_means[0]
        exp_sinkhorn_std = sinkhorn_stds[0]
    else:
        mmd_mean = np.array([m["mmd_pred_mean"] for m in metrics_list], dtype=float)
        mmd_std  = np.array([m["mmd_pred_std"]  for m in metrics_list], dtype=float)
        sk_mean  = np.array([m["sinkhorn_pred_mean"] for m in metrics_list], dtype=float)
        sk_std   = np.array([m["sinkhorn_pred_std"]  for m in metrics_list], dtype=float)
        exp_mmd_mean = mmd_means[1]
        exp_mmd_std = mmd_stds[1]
        exp_sinkhorn_mean = sinkhorn_means[1]
        exp_sinkhorn_std = sinkhorn_stds[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(xs, mmd_mean, label=f"{role.upper()} MMD")
    axes[0].fill_between(xs, mmd_mean - mmd_std, mmd_mean + mmd_std, alpha=0.2)
    axes[0].axhline(exp_mmd_mean, color="red", linewidth=2, label="Expert MMD Mean ± Std")
    axes[0].fill_between(xs, exp_mmd_mean - exp_mmd_std, exp_mmd_mean + exp_mmd_std, alpha=0.2)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("MMD")
    axes[0].set_title(f"{role.upper()} MMD (Mean ± Std)")
    axes[0].set_xlim(0, max_steps)
    axes[0].grid(True)

    axes[1].plot(xs, sk_mean, label=f"{role.upper()} Sinkhorn")
    axes[1].fill_between(xs, sk_mean - sk_std, sk_mean + sk_std, alpha=0.2)
    axes[1].axhline(exp_sinkhorn_mean, color="red", linewidth=2, label="Expert Sinkhorn Mean ± Std")
    axes[1].fill_between(xs, exp_sinkhorn_mean - exp_sinkhorn_std, exp_sinkhorn_mean + exp_sinkhorn_std, alpha=0.2)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Sinkhorn")
    axes[1].set_title(f"{role.upper()} Sinkhorn (Mean ± Std)")
    axes[1].set_xlim(0, max_steps)
    axes[1].grid(True)

    plt.tight_layout()

    filename = f"eval_{role}_metrics.png"
    plot_path = save_dir / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_es_metrics(policy_metrics, role="prey", save_dir=None):
    gens = np.arange(len(policy_metrics))
    pms = [pm[role] for pm in policy_metrics]

    pin_delta_norm = np.array([pm[0].get("delta_norm", np.nan) for pm in pms], dtype=float)
    an_delta_norm  = np.array([pm[1].get("delta_norm", np.nan) for pm in pms], dtype=float)

    pin_diff_std = np.array([pm[0].get("diff_std", np.nan) for pm in pms], dtype=float)
    an_diff_std  = np.array([pm[1].get("diff_std", np.nan) for pm in pms], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(gens, pin_delta_norm, label="PIN delta_norm")
    axes[0].plot(gens, an_delta_norm, label="AN delta_norm")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("delta_norm")
    axes[0].set_title(f"{role.upper()} ES step size per generation")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(gens, pin_diff_std, label="PIN diff_std")
    axes[1].plot(gens, an_diff_std, label="AN diff_std")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("diff_std")
    axes[1].set_title(f"{role.upper()} Reward-diff std per generation")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plot_path = save_dir / f"es_{role}_policy_metrics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()




def compute_polarization(vx, vy):
    stacked_vs = np.stack([vx, vy], axis=1)
    norms = np.linalg.norm(stacked_vs, axis=1, keepdims=True)
    norm_vs = stacked_vs / norms
    prey_vs = norm_vs[1:]
    mean_vs = prey_vs.mean(axis=0)
    polarization_score = np.linalg.norm(mean_vs)
    return polarization_score


def degree_of_sparsity(xs, ys):
    points = np.column_stack((xs, ys))
    tree = cKDTree(points) #faster in calculation
    dists, _ = tree.query(points, k=2)
    nn_dists = dists[:, 1]
    return nn_dists.mean()


def compute_angular_momentum(x, y, vx, vy):
    positions = np.stack([x, y], axis=1)
    velocities = np.stack([vx, vy], axis=1)
    prey_pos = positions[1:]
    prey_vel = velocities[1:]

    center = prey_pos.mean(axis=0)
    rel_pos = prey_pos - center
    norms = np.linalg.norm(prey_vel, axis=1, keepdims=True)
    v_hat = prey_vel / (norms + 1e-8)

    Lz = rel_pos[:, 0] * v_hat[:, 1] - rel_pos[:, 1] * v_hat[:, 0] # cross product z-component
    return np.abs(Lz.mean())


def distance_to_predator(xs, ys):
    predator_pos = np.array([xs[0], ys[0]])
    prey_pos = np.stack([xs[1:], ys[1:]], axis=1)

    center = prey_pos.mean(axis=0)
    return float(np.linalg.norm(center - predator_pos))


def escape_alignment(xs, ys, vxs, vys):
    pred_pos = np.array([xs[0], ys[0]])

    prey_pos = np.stack([xs[1:], ys[1:]], axis=1)
    prey_vel = np.stack([vxs[1:], vys[1:]], axis=1)

    escape_dir = prey_pos - pred_pos
    escape_dir_norm = np.linalg.norm(escape_dir, axis=1, keepdims=True)
    escape_dir = escape_dir / (escape_dir_norm + 1e-8)

    vel_norm = np.linalg.norm(prey_vel, axis=1, keepdims=True)
    prey_vel_normed = prey_vel / (vel_norm + 1e-8)

    alignment = np.mean(np.sum(prey_vel_normed * escape_dir, axis=1))
    return float(alignment)
