import os
import ast
import torch
import keyboard
import numpy as np
import pandas as pd
from utils.env_utils import *
from utils.train_utils import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
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


def compute_expert_metrics2(expert_data):
    polarizations = []
    angular_momenta = []
    sparsities = []
    distances_to_predator = []
    escape_alignments = []

    for video in expert_data.keys():
        xs  = expert_data[video]["xs"]
        ys  = expert_data[video]["ys"]
        vxs = expert_data[video]["vxs"]
        vys = expert_data[video]["vys"]

        polarizations.append(compute_polarization(vxs, vys))
        angular_momenta.append(compute_angular_momentum(xs, ys, vxs, vys))
        sparsities.append(degree_of_sparsity(xs, ys))
        distances_to_predator.append(distance_to_predator(xs, ys))
        escape_alignments.append(escape_alignment(xs, ys, vxs, vys))

    return {"polarization": polarizations, 
            "angular_momentum": angular_momenta,
            "sparsity": sparsities,
            "distance_to_predator": distances_to_predator,
            "escape_alignment": escape_alignments}


def compute_expert_metrics(expert_data, num_agents):
    polarizations = []
    angular_momenta = []
    sparsities = []
    distances_to_predator = []
    escape_alignments = []

    for video in expert_data.keys():
        xs  = np.asarray(expert_data[video]["xs"])  # shape (T*N,)
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        # --- in (T, N) umformen ---
        T = len(xs) // num_agents
        xs  = xs.reshape(T, num_agents)
        ys  = ys.reshape(T, num_agents)
        vxs = vxs.reshape(T, num_agents)
        vys = vys.reshape(T, num_agents)

        # --- pro Frame Metriken berechnen ---
        for t in range(T):
            x_t  = xs[t]    # (N,)
            y_t  = ys[t]
            vx_t = vxs[t]
            vy_t = vys[t]

            polarizations.append(compute_polarization(vx_t, vy_t))
            angular_momenta.append(compute_angular_momentum(x_t, y_t, vx_t, vy_t))
            sparsities.append(degree_of_sparsity(x_t, y_t))
            distances_to_predator.append(distance_to_predator(x_t, y_t))
            escape_alignments.append(escape_alignment(x_t, y_t, vx_t, vy_t))

    return {
        "polarization": polarizations,
        "angular_momentum": angular_momenta,
        "sparsity": sparsities,
        "distance_to_predator": distances_to_predator,
        "escape_alignment": escape_alignments,
    }


def run_policies(env, pred_policy, prey_policy, render=True):
    if render:
        print("Press 'q' to end simulation.")

    metrics = []

    while True:
        if render and keyboard.is_pressed('q'):
            break

        global_state = env.state().item()
        pred_tensor, prey_tensor, xs, ys, dx, dy, vxs, vys = get_eval_features(global_state)

        pred_states = pred_tensor[..., :4]
        action_pred, mu_pred, sigma_pred, weights_pred = pred_policy.forward(pred_states)
        dis_pred = continuous_to_discrete(action_pred, 360, role='predator')

        prey_states = prey_tensor[..., :4]
        action_prey, mu_prey, sigma_prey, weights_prey, pred_gain = prey_policy.forward(prey_states)
        dis_prey = continuous_to_discrete(action_prey, 360, role='prey')

        # Action dictionary
        action_dict = {'predator_0': dis_pred}
        for i, agent_name in enumerate(sorted([agent for agent in env.agents if agent.startswith("prey")])):
            action_dict[agent_name] = dis_prey[i]

        env.step(action_dict)

        # Log metrics
        metrics.append({
            "polarization": compute_polarization(vxs, vys),
            "angular_momentum": compute_angular_momentum(xs, ys, vxs, vys),
            "degree_of_sparsity": degree_of_sparsity(xs, ys),
            "distance_to_predator": distance_to_predator(xs, ys),
            "escape_alignment": escape_alignment(xs, ys, vxs, vys),
            "actions": (dis_pred, dis_prey),
            "mu": (mu_pred, mu_prey),
            "sigma": (sigma_pred, sigma_prey),
            "weights": (weights_pred, weights_prey),
            "pred_gain": pred_gain,
            "xs": xs,
            "ys": ys,
            "dx": dx,
            "dy": dy,
            "vxs": vxs,
            "vys": vys
        })

        # Render only if user wants it
        if render:
            env.render()

    # Try closing the environment
    try:
        env.close()
    except:
        pass

    return metrics


def run_policies_in_steps(env, pred_policy, prey_policy, steps=200, render=True):
    if render:
        print("Press 'q' to end simulation.")

    metrics = []

    for frame in range(steps):
        if render and keyboard.is_pressed('q'):
            break

        global_state = env.state().item()
        pred_tensor, prey_tensor, xs, ys, dx, dy, vxs, vys = get_eval_features(global_state)

        # Predator
        pred_states = pred_tensor[..., :4]
        action_pred, mu_pred, sigma_pred, weights_pred = pred_policy.forward(pred_states)
        dis_pred = continuous_to_discrete(action_pred, 360, role='predator')

        prey_states = prey_tensor[..., :4]
        action_prey, mu_prey, sigma_prey, weights_prey, pred_gain = prey_policy.forward(prey_states)
        dis_prey = continuous_to_discrete(action_prey, 360, role='prey')

        # Action dictionary
        action_dict = {'predator_0': dis_pred}
        for i, agent_name in enumerate(sorted([agent for agent in env.agents if agent.startswith("prey")])):
            action_dict[agent_name] = dis_prey[i]

        env.step(action_dict)

        # Log metrics
        metrics.append({
            "polarization": compute_polarization(vxs, vys),
            "angular_momentum": compute_angular_momentum(xs, ys, vxs, vys),
            "degree_of_sparsity": degree_of_sparsity(xs, ys),
            "distance_to_predator": distance_to_predator(xs, ys),
            "escape_alignment": escape_alignment(xs, ys, vxs, vys),
            "actions": (dis_pred, dis_prey),
            "mu": (mu_pred, mu_prey),
            "sigma": (sigma_pred, sigma_prey),
            "weights": (weights_pred, weights_prey),
            "pred_gain": pred_gain,
            "xs": xs,
            "ys": ys,
            "dx": dx,
            "dy": dy,
            "vxs": vxs,
            "vys": vys
        })

        # Render only if user wants it
        if render:
            env.render()

    # Try closing the environment
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
        df.rename(columns={'0': 'wasserstein_loss', '1': 'gradient_penalty', '2': 'exp_scores', '3': 'gen_scores'}, inplace=True)

    metrics = {'Predator': dis_metrics_pred, 'Prey': dis_metrics_prey}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    for col, (role_name, df) in enumerate(metrics.items()):
        gens = df.index

        ax = axes[0, col]
        ax.plot(gens, df['wasserstein_loss'], label='Loss')
        ax.set_title('Wasserstein Loss')
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
            '0': 'wasserstein_loss',
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


