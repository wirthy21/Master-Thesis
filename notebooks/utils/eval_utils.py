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
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


#############################
##### POLICY EVALUATION #####
#############################

# height, width = 800, 800
# v_predator = 5.0
# v_prey = 4.0
def get_eval_features(global_state, max_speed=15.0):
    sorted_gs = dict(sorted(global_state.items()))
    items = list(sorted_gs.items())
    agents, raw = zip(*items)

    n = len(raw)

    # Predator: [0, scaled_position_x, scaled_position_y, scaled_direction, scaled_speed]
    # Prey:     [1, scaled_position_x, scaled_position_y, scaled_direction, scaled_speed]
    xs = np.fromiter((r[1] for r in raw), dtype=np.float32, count=n) # [0, 1]
    ys = np.fromiter((r[2] for r in raw), dtype=np.float32, count=n) # [0, 1]
    directions = np.fromiter((r[3] for r in raw), dtype=np.float32, count=n) # [0, 1]
    speeds_norm = np.fromiter((r[4] for r in raw), dtype=np.float32, count=n) # [0, 1]
    speeds = speeds_norm * max_speed  # [0, max_speed] to align with expert
    
    thetas = (directions - 0.5) * (2 * np.pi)     # [-pi, pi]
    thetas_norm = thetas / np.pi # convert to [-1,1]

    cos_t = np.cos(thetas)                        
    sin_t = np.sin(thetas)
    vxs = cos_t * speeds                       
    vys = sin_t * speeds

    # pairwise distances
    dx = xs[None, :] - xs[:, None]  # [-1, 1]
    dy = ys[None, :] - ys[:, None]  # [-1, 1]

    # relative velocities
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :] # range [-1,1]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :] # range [-1,1]

    max_speed_env = 25.0
    rel_vx = np.clip(rel_vx, -max_speed_env, max_speed_env) / max_speed_env # range [-1,1]
    rel_vy = np.clip(rel_vy, -max_speed_env, max_speed_env) / max_speed_env # range [-1,1]
    
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


def compute_expert_data_ranges(expert_data):
    all_x = []
    all_y = []
    all_vx = []
    all_vy = []

    for video in expert_data.keys():
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        all_x.append(xs)
        all_y.append(ys)
        all_vx.append(vxs)
        all_vy.append(vys)

    all_x  = np.concatenate(all_x)
    all_y  = np.concatenate(all_y)
    all_vx = np.concatenate(all_vx)
    all_vy = np.concatenate(all_vy)

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    vx_min, vx_max = float(all_vx.min()), float(all_vx.max())
    vy_min, vy_max = float(all_vy.min()), float(all_vy.max())

    return (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max)


def compute_expert_metrics(expert_data, num_agents):
    polarizations = []
    angular_momenta = []
    sparsities = []
    distances_to_predator = []
    escape_alignments = []

    # x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max
    x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max = compute_expert_data_ranges(expert_data)

    for video in expert_data.keys():
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        vxs = 2* (vxs - vx_min) / (vx_max - vx_min) -1
        vys = 2* (vys - vy_min) / (vy_max - vy_min) -1

        T = len(xs) // num_agents
        xs  = xs.reshape(T, num_agents)
        ys  = ys.reshape(T, num_agents)
        vxs = vxs.reshape(T, num_agents)
        vys = vys.reshape(T, num_agents)

        for t in range(T):
            x_t  = xs[t]
            y_t  = ys[t]
            vx_t = vxs[t]
            vy_t = vys[t]

            polarizations.append(compute_polarization(vx_t, vy_t))
            angular_momenta.append(compute_angular_momentum(x_t, y_t, vx_t, vy_t))
            sparsities.append(degree_of_sparsity(x_t, y_t))
            distances_to_predator.append(distance_to_predator(x_t, y_t))
            escape_alignments.append(escape_alignment(x_t, y_t, vx_t, vy_t))

    return {"polarization": polarizations,
            "angular_momentum": angular_momenta,
            "sparsity": sparsities,
            "distance_to_predator": distances_to_predator,
            "escape_alignment": escape_alignments}


def run_policies(env, pred_policy, prey_policy, render=True):
    if render:
        print("Press 'q' to end simulation.")

    metrics = []

    while True:
        if render and keyboard.is_pressed('q'):
            break

        global_state = env.state().item()
        pred_tensor, prey_tensor, xs, ys, directions, dx, dy, vxs, vys = get_eval_features(global_state)

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

        # Indexing - needed for mapping weights to neighbors
        n_agents = len(xs)
        all_idx = np.arange(n_agents, dtype=int)
        pred_idx = 0 # after sorting in get_eval pred first
        prey_indices = all_idx[1:]      
        pred_neighbor_idx = all_idx[all_idx != pred_idx] # neighbor of pred exept itself
        prey_neighbor_idx = np.stack([all_idx[all_idx != i] for i in prey_indices], axis=0) # neighbor of preys except themselves

        # Predator
        pred_states = pred_tensor[..., :4]
        pred_heading = pred_tensor[0, 0, 4].item()
        action_pred, mu_pred, sigma_pred, weights_pred = pred_policy.forward(pred_states)
        dis_pred = continuous_to_discrete(action_pred, 360, role='predator')

        prey_states = prey_tensor[..., :4]
        prey_headings = prey_tensor[:, 0, 4]
        prey_headings_list = prey_headings.tolist()
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
            "headings": [pred_heading] + prey_headings_list,
            "mu": (mu_pred, mu_prey),
            "sigma": (sigma_pred, sigma_prey),
            "weights": (weights_pred, weights_prey),
            "weights_idx": {"pred": {"self": pred_idx, "neighbors": pred_neighbor_idx}, "prey": {"self": prey_indices, "neighbors": prey_neighbor_idx}},
            "pred_gain": pred_gain,
            "dx": dx,
            "dy": dy,
            "vxs": vxs,
            "vys": vys,
            "global_state": global_state
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


def plot_swarm_metrics(gail_metrics, bc_metrics, couzin_metrics, expert_metrics):
    df_gail = pd.DataFrame(gail_metrics)
    df_bc = pd.DataFrame(bc_metrics)
    steps = np.arange(len(df_gail))

    fig, axes = plt.subplots(1, 3, figsize=(24, 5))

    ### POLARIZATION ###
    gail_polarization = list(df_gail["polarization"])
    bc_polarization = list(df_bc["polarization"])
    couzin_polarization = [m["polarization"] for m in couzin_metrics]
    expert_polarization = np.mean(expert_metrics["polarization"])

    axes[0].plot(steps, gail_polarization, label="GAIL", color="#29485d", linewidth=1)
    axes[0].plot(steps, bc_polarization, label="BC", color="#00b1ea", linewidth=1)
    axes[0].plot(steps, couzin_polarization, label="Couzin", color="#c6d325", linewidth=1)
    axes[0].axhline(expert_polarization, label="Expert", color="#ef7c00", linewidth=1)

    axes[0].set_xlabel("Steps", fontsize=14)
    axes[0].set_ylabel("Polarization", fontsize=14)
    axes[0].set_title("Polarization Over Time", fontsize=18)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    ### ANGULAR MOMENTUM ###
    gail_am = list(df_gail["angular_momentum"])
    bc_am = list(df_bc["angular_momentum"])
    couzin_am = [m["angular_momentum"] for m in couzin_metrics]
    expert_am = np.mean(expert_metrics["angular_momentum"])
    max_am = max(max(gail_am), max(bc_am), max(couzin_am), expert_am)

    axes[1].plot(steps, gail_am, label="GAIL", color="#29485d", linewidth=1)
    axes[1].plot(steps, bc_am, label="BC", color="#00b1ea", linewidth=1)
    axes[1].plot(steps, couzin_am, label="Couzin", color="#c6d325", linewidth=1)
    axes[1].axhline(expert_am, label="Expert", color="#ef7c00", linewidth=1)

    axes[1].set_xlabel("Steps", fontsize=14)
    axes[1].set_ylabel("Angular Momentum", fontsize=14)
    axes[1].set_title("Angular Momentum Over Time", fontsize=18)
    axes[1].set_ylim(0, max_am)
    axes[1].legend()

    ### DEGREE OF SPARSITY ###
    gail_dos = list(df_gail["degree_of_sparsity"])
    bc_dos = list(df_bc["degree_of_sparsity"])
    couzin_dos = [m["degree_of_sparsity"] for m in couzin_metrics]
    expert_dos = np.mean(expert_metrics["sparsity"])
    max_dos = max(max(gail_dos), max(bc_dos), max(couzin_dos))

    axes[2].plot(steps, gail_dos, label="GAIL", color="#29485d", linewidth=1) #005555
    axes[2].plot(steps, bc_dos, label="BC", color="#00b1ea", linewidth=1) #777777
    axes[2].plot(steps, couzin_dos, label="Couzin", color="#c6d325", linewidth=1) #a7a7a8
    axes[2].axhline(expert_dos, label="Expert", color="#ef7c00", linewidth=1) #000000

    axes[2].set_xlabel("Steps", fontsize=14)
    axes[2].set_ylabel("Degree of Sparsity", fontsize=14)
    axes[2].set_title("Degree of Sparsity Over Time", fontsize=18)
    axes[2].set_ylim(0, max_dos)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    

def plot_pred_prey_metrics(gail_metrics, bc_metrics, couzin_metrics, expert_metrics):
    df_gail = pd.DataFrame(gail_metrics)
    df_bc = pd.DataFrame(bc_metrics)
    steps = np.arange(len(df_gail))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- DISTANCE TO PREDATOR ---
    gail_dtp = df_gail["distance_to_predator"].tolist()
    bc_dtp = df_bc["distance_to_predator"].tolist()
    couzin_dtp = [m["distance_to_predator"] for m in couzin_metrics]
    expert_dtp = np.mean(expert_metrics["distance_to_predator"])

    axes[0].plot(steps, gail_dtp, label="GAIL", color="#29485d", linewidth=1)
    axes[0].plot(steps, bc_dtp, label="BC", color="#00b1ea", linewidth=1)
    axes[0].plot(steps, couzin_dtp, label="Couzin", color="#c6d325", linewidth=1)
    axes[0].axhline(expert_dtp, label="Expert", color="#ef7c00", linewidth=1)

    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Distance to Predator")
    axes[0].set_title("Distance to Predator Over Time")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- ESCAPE ALIGNMENT ---
    gail_ea = df_gail["escape_alignment"].tolist()
    bc_ea = df_bc["escape_alignment"].tolist()
    couzin_ea = [m["escape_alignment"] for m in couzin_metrics]
    expert_ea = np.mean(expert_metrics["escape_alignment"])

    axes[1].plot(steps, gail_ea, label="GAIL", color="#29485d", linewidth=1)
    axes[1].plot(steps, bc_ea, label="BC", color="#00b1ea", linewidth=1)
    axes[1].plot(steps, couzin_ea, label="Couzin", color="#c6d325", linewidth=1)
    axes[1].axhline(expert_ea, label="Expert", color="#ef7c00", linewidth=1)

    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Escape Alignment")
    axes[1].set_title("Escape Alignment Over Time")
    axes[1].set_ylim(-1, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


##########################
##### ATTENTION MAPS #####
##########################


def compute_pin_an_maps(pin, an, role, v=1.0,
                        x_range=(-150, 150),  y_range=(-150, 150),
                        n_points=80, n_orient=72, device="cpu"):

    pin.to(device).eval()
    an.to(device).eval()

    xs = np.linspace(*x_range, n_points)
    ys = np.linspace(*y_range, n_points)

    thetas = torch.linspace(0.0, 2 * np.pi, n_orient + 1, device=device)[:-1]
    vx = v * torch.cos(thetas)
    vy = v * torch.sin(thetas)

    action_map = np.zeros((n_points, n_points), dtype=np.float32)
    attn_map = np.zeros((n_points, n_points), dtype=np.float32)

    with torch.no_grad():
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                rel_x = torch.full((n_orient, 1), float(x), device=device)
                rel_y = torch.full((n_orient, 1), float(y), device=device)
                inputs = torch.cat([rel_x, rel_y, vx.unsqueeze(1), vy.unsqueeze(1)], dim=1)

                # PIN: mu, sigma
                mu, sigma = pin(inputs)
                turn = torch.tanh(mu.squeeze())
                action_map[iy, ix] = turn.mean().item()

                # AN: attention logits/weights
                w_logits = an(inputs)
                if isinstance(w_logits, tuple):
                    w_logits = w_logits[0]
                w_logits = w_logits.squeeze()
                attn_map[iy, ix] = w_logits.mean().item()

    return xs, ys, action_map, attn_map



def plot_policy_maps(xs, ys, action_map, attn_map, role="predator", img_path=None):

    cmap_pin_color = "inferno"
    cmap_an_color = "RdBu"
    x, y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ----- Pairwise-Interaction Map -----
    vmax_act = np.nanmax(np.abs(action_map))
    norm_act = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_act, vmax=vmax_act)

    im0 = axes[0].contourf(x, y, action_map, levels=30, cmap=cmap_pin_color, norm=norm_act)
    axes[0].set_title(f"[{role.upper()}] Pairwise-Interaction Map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], label="action")

    # ----- Attention Map -----
    vmax_att = np.nanmax(np.abs(attn_map))
    norm_att = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_att, vmax=vmax_att)

    im1 = axes[1].contourf(x, y, attn_map, levels=30, cmap=cmap_an_color, norm=norm_att)
    axes[1].set_title(f"[{role.upper()}] Attention Map")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], label="attention (logit)")

    x_range = (xs.min(), xs.max())
    y_range = (ys.min(), ys.max())
    for ax in axes:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    if img_path is not None and os.path.exists(img_path):
        icon = mpimg.imread(img_path)
        imgbox = OffsetImage(icon, zoom=0.45)
        center = (0, 0)
        for ax in axes:
            ab = AnnotationBbox(imgbox, center, frameon=False, xycoords='data')
            ax.add_artist(ab)

    plt.tight_layout()
    plt.show()


####################################
##### Discriminator Evaluation #####
####################################


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


