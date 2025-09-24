import os
import ast
import torch
import keyboard
import numpy as np
import pandas as pd
from utils.env_utils import *
from utils.train_utils import *
import matplotlib.pyplot as plt
from marl_aquarium import aquarium_v0

def get_eval_features(global_state):
    sorted_gs = dict(sorted(global_state.items()))
    items = list(sorted_gs.items())
    agents, raw = zip(*items)

    n = len(raw)

    xs = np.fromiter((r[1] for r in raw), dtype=np.float32, count=n)
    ys = np.fromiter((r[2] for r in raw), dtype=np.float32, count=n)
    directions = np.fromiter((r[3] for r in raw), dtype=np.float32, count=n)
    speeds = np.fromiter((r[4] for r in raw), dtype=np.float32, count=n)
    
    angle = directions * 360 - 180  # convert to [-180,180]
    thetas = np.deg2rad(angle)  # convert to [-pi,pi]
    thetas_norm = thetas / np.pi # convert to [-1,1]

    thetas = np.deg2rad(directions) # radians
    cos_t = np.cos(thetas)                        
    sin_t = np.sin(thetas)
    vxs = cos_t * speeds                       
    vys = sin_t * speeds

    # pairwise distances
    dx = xs[None, :] - xs[:, None] # range [-1,1]
    dy = ys[None, :] - ys[:, None] # range [-1,1]

    # relative velocities
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :] # range [-1,1]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :] # range [-1,1]
    
    n = xs.shape[0]
    thetas_mat = np.tile(thetas_norm[:, None], (1, n))
    features = np.stack([dx, dy, rel_vx, rel_vy, thetas_mat], axis=-1)

    mask = ~np.eye(n, dtype=bool) # shape (N, N)
    neigh = features[mask].reshape(n, n-1, 5)

    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:]) 
    
    return pred_tensor, prey_tensor, xs, ys, dx, dy, vxs, vys


def plot_pretraining(logs_pretrain_pred, logs_pretrain_prey):
    # Listen von Dicts in x/y umwandeln
    ep_pred  = [d["epoch"] for d in logs_pretrain_pred]
    loss_pred = [d["loss"]  for d in logs_pretrain_pred]

    ep_prey  = [d["epoch"] for d in logs_pretrain_prey]
    loss_prey = [d["loss"]  for d in logs_pretrain_prey]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Predator
    axes[0].plot(ep_pred, loss_pred, label="Loss")
    axes[0].set_title("Pretraining Predator Policy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Prey
    axes[1].plot(ep_prey, loss_prey, label="Loss")
    axes[1].set_title("Pretraining Prey Policy Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def compute_polarization(vx, vy):
    v = np.stack([vx, vy], axis=1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / (norms + 1e-8)

    u_pred = u[0:1]
    pred_score = np.linalg.norm(u_pred.mean(axis=0))

    u_prey = u[1:]
    prey_score = np.linalg.norm(u_prey.mean(axis=0))

    return (pred_score, prey_score)


def compute_angular_momentum(x, y, vx, vy):
    v = np.stack([vx, vy], axis=1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / (norms + 1e-8)
    r = np.stack([x, y], axis=1)
    center = r.mean(axis=0)
    r_rel = r - center
    cross_z = r_rel[:,0] * u[:,1] - r_rel[:,1] * u[:,0]
    pred_am = np.abs(cross_z[0])
    prey_am = np.abs(cross_z[1:].mean())
    return (pred_am, prey_am)


def mean_pairwise_distance(dx, dy):
    dx = np.array(dx)
    dy = np.array(dy)
    
    pred_dx = dx[0, 1:]
    pred_dy = dy[0, 1:]
    pred_dist = np.sqrt(pred_dx**2 + pred_dy**2).mean()
    
    prey_dx = dx[1:, 1:]
    prey_dy = dy[1:, 1:]
    prey_dist = np.sqrt(prey_dx**2 + pred_dy**2).mean()
    
    return pred_dist, prey_dist



def run_policies(env, pred_policy, prey_policy): 
    print("Press 'q' to end simulation.")

    metrics = []

    while True:
        if keyboard.is_pressed('q'):
            break

        global_state = env.state().item()
        pred_tensor, prey_tensor, xs, ys, dx, dy, vxs, vys = get_eval_features(global_state)

        pred_states = pred_tensor[..., :4]
        con_pred = pred_policy.forward_pred(pred_states)
        dis_pred = continuous_to_discrete(con_pred, 360, role='predator')

        prey_states = prey_tensor[..., :4]
        con_prey = prey_policy.forward_prey(prey_states)
        dis_prey = continuous_to_discrete(con_prey, 360, role='prey')

        action_dict = {'predator_0': dis_pred}
        for i, agent_name in enumerate(sorted([agent for agent in env.agents if agent.startswith("prey")])):
            action_dict[agent_name] = dis_prey[i]

        env.step(action_dict)

        metrics.append({"polarization": compute_polarization(vxs, vys),
                        "angular_momentum": compute_angular_momentum(xs, ys, vxs, vys),
                        "mean_pairwise_distance": mean_pairwise_distance(dx, dy),
                        "xs": xs,
                        "ys": ys,
                        "dx": dx,
                        "dy": dy,
                        "vxs": vxs,
                        "vys": vys})
        
        env.render()

    try:
        env.close()
    except:
        pass

    return metrics


def plot_metrics(metrics):
    pred_p = [m["polarization"][0] for m in metrics]
    prey_p = [m["polarization"][1] for m in metrics]
    pred_am = [m["angular_momentum"][0] for m in metrics]
    prey_am = [m["angular_momentum"][1] for m in metrics]
    pred_dis = [m["mean_pairwise_distance"][0] for m in metrics]
    prey_dis = [m["mean_pairwise_distance"][1] for m in metrics]
    t = np.arange(len(metrics))

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes[0, 0].plot(t, pred_p)
    axes[0, 0].set_title("Pred Polarization")
    axes[0, 1].plot(t, pred_am)
    axes[0, 1].set_title("Pred Angular Momentum")
    axes[0, 2].plot(t, pred_dis)
    axes[0, 2].set_title("Pred Mean Pairwise Distance")
    axes[1, 0].plot(t, prey_p)
    axes[1, 0].set_title("Prey Polarization")
    axes[1, 1].plot(t, prey_am)
    axes[1, 1].set_title("Prey Angular Momentum")
    axes[1, 2].plot(t, prey_dis)
    axes[1, 2].set_title("Prey Mean Pairwise Distance")


    fig.tight_layout()
    plt.show()


def evaluate_discriminator(dis_metrics_pred, dis_metrics_prey):

    for df in (dis_metrics_pred, dis_metrics_prey):
        df.rename(columns={'0': 'discriminator_loss', '1': 'gradient_penalty', '2': 'exp_scores', '3': 'gen_scores'}, inplace=True)

    metrics = {'Predator': dis_metrics_pred, 'Prey': dis_metrics_prey}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    for col, (role_name, df) in enumerate(metrics.items()):
        gens = df.index

        ax = axes[0, col]
        ax.plot(gens, df['discriminator_loss'], label='Loss')
        ax.set_title('Discriminator Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Generations')
        ax.legend()

        ax.annotate(role_name, xy=(0.5, 1.25), xycoords='axes fraction', ha='center', fontsize='x-large', fontweight='bold')

        ax = axes[1, col]
        ax.plot(gens, df['gradient_penalty'], label='Penalty')
        ax.set_title('Gradient Penalty')
        ax.set_ylabel('Penalty')
        ax.set_xlabel('Generations')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_scores(dis_metrics_pred, dis_metrics_prey):
    for df in (dis_metrics_pred, dis_metrics_prey):
        df.rename(columns={
            '0': 'discriminator_loss',
            '1': 'gradient_penalty',
            '2': 'exp_scores',
            '3': 'gen_scores'
        }, inplace=True)

    metrics = {'Predator': dis_metrics_pred, 'Prey': dis_metrics_prey}

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for idx, (role_name, df) in enumerate(metrics.items()):
        gens = df.index
        ax = axes[idx]

        ax.plot(gens, df['exp_scores'], color='red', label='Expert Score')
        ax.plot(gens, df['gen_scores'], color='blue', label='Policy Score')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)

        ax.set_title(f'{role_name} Scores')
        ax.set_ylabel('Score')
        ax.legend(loc='upper left')

    axes[-1].set_xlabel('Generations')
    plt.tight_layout()
    plt.show()


def evaluate_es(es_metrics_pred, es_metrics_prey):
    es_dict = {'Predator': es_metrics_pred.rename(columns={'0': 'pairwise_update', '1': 'attention_update'}),
               'Prey':     es_metrics_prey.rename(columns={'0': 'pairwise_update', '1': 'attention_update'})}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col', sharey='row')

    for col, (role_name, es) in enumerate(es_dict.items()):
        pair_df = pd.DataFrame([ast.literal_eval(s) for s in es['pairwise_update']])
        att_df  = pd.DataFrame([ast.literal_eval(s) for s in es['attention_update']])
        gens = pair_df['generation']

        # Pairwise Update
        ax = axes[0, col]
        ax.plot(gens, pair_df['avg_reward_diff'], label='Reward')
        ax.fill_between(
            gens,
            pair_df['avg_reward_diff'] - pair_df['diff_std'],
            pair_df['avg_reward_diff'] + pair_df['diff_std'],
            alpha=0.2,
            label='Std'
        )
        coeffs_p = np.polyfit(gens, pair_df['avg_reward_diff'], 1)
        ax.plot(gens, np.poly1d(coeffs_p)(gens), linestyle='--', label='Regression')
        slope_p = coeffs_p[0]
        ax.set_title(f'Pairwise-Interaction Network\nSlope: {slope_p:.4f}')
        if col == 0:
            ax.set_ylabel('Avg Reward Diff')
        ax.set_xlabel('Generations')
        ax.legend(loc='upper left')
        ax.annotate(role_name, xy=(0.5, 1.25), xycoords='axes fraction',
                    ha='center', fontsize='x-large', fontweight='bold')

        # Attention Update
        ax = axes[1, col]
        ax.plot(gens, att_df['avg_reward_diff'], label='Reward')
        ax.fill_between(
            gens,
            att_df['avg_reward_diff'] - att_df['diff_std'],
            att_df['avg_reward_diff'] + att_df['diff_std'],
            alpha=0.2,
            label='Std'
        )
        coeffs_a = np.polyfit(gens, att_df['avg_reward_diff'], 1)
        ax.plot(gens, np.poly1d(coeffs_a)(gens), linestyle='--', label='Regression')
        slope_a = coeffs_a[0]
        ax.set_title(f'Attention Network\nSlope: {slope_a:.4f}')
        if col == 0:
            ax.set_ylabel('Avg Reward Diff')
        ax.set_xlabel('Generations')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_grad_norm(es_metrics_pred, es_metrics_prey):
    es_dict = {
        'Predator': es_metrics_pred.rename(columns={'0': 'pairwise_update', '1': 'attention_update'}),
        'Prey':     es_metrics_prey.rename(columns={'0': 'pairwise_update', '1': 'attention_update'})
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col', sharey='row')

    for col, (role, es) in enumerate(es_dict.items()):
        # Strings parsen
        pair_df = pd.DataFrame([ast.literal_eval(s) for s in es['pairwise_update']])
        att_df  = pd.DataFrame([ast.literal_eval(s) for s in es['attention_update']])
        gens = pair_df['generation']

        # Pairwise grad_norm
        ax = axes[0, col]
        ax.plot(gens, pair_df['grad_norm'], label='Grad Norm')
        ax.set_title('Pairwise-Interaction Network')
        if col == 0:
            ax.set_ylabel('Grad Norm')
        ax.legend(loc='upper right')
        ax.annotate(role, xy=(0.5, 1.15), xycoords='axes fraction',
                    ha='center', fontsize='x-large', fontweight='bold')

        # Attention grad_norm
        ax = axes[1, col]
        ax.plot(gens, att_df['grad_norm'], label='Grad Norm')
        ax.set_title('Attention Network')
        if col == 0:
            ax.set_ylabel('Grad Norm')
        ax.legend(loc='upper right')

    for ax in axes.flat:
        ax.set_xlabel('Generations')

    plt.tight_layout()
    plt.show()


def plot_trajectory(metrics, role="predator"):

    xs_pred = np.array([m["xs"][0] for m in metrics])
    ys_pred = np.array([m["ys"][0] for m in metrics])

    xs_prey = np.stack([m["xs"][1:] for m in metrics])
    ys_prey = np.stack([m["ys"][1:] for m in metrics])

    plt.figure(figsize=(6,6))

    if role in ('predator', 'both'):
        plt.plot(xs_pred, ys_pred, color='red', label='Predator')

    if role in ('prey', 'both'):
        T, N = xs_prey.shape
        for i in range(N):
            plt.plot(xs_prey[:, i], ys_prey[:, i], color='gray', alpha=0.6)

    plt.title(f'{role.capitalize()} Trajectory Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    plt.show()