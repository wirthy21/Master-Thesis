import os
import ast
import torch
import keyboard
import numpy as np
import pandas as pd
from utils.train_utils import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import custom_marl_aquarium
import matplotlib.image as mpimg
import matplotlib.colors as colors
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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




def plot_swarm_metrics(gail_metrics, bc_metrics, couzin_metrics, random_metrics, expert_metrics):
    steps = np.arange(len(gail_metrics))

    fig, axes = plt.subplots(1, 3, figsize=(24, 5))

    ### POLARIZATION ###
    gail_polarization = [m["polarization"] for m in gail_metrics]
    bc_polarization = [m["polarization"] for m in bc_metrics]
    couzin_polarization = [m["polarization"] for m in couzin_metrics]
    random_polarization = [m["polarization"] for m in random_metrics]
    expert_polarization = np.mean(expert_metrics["polarization"])

    axes[0].plot(steps, gail_polarization, label="GAIL", color="#8B0000", linewidth=1)
    axes[0].plot(steps, bc_polarization, label="BC", color="#F08080", linewidth=1)
    axes[0].plot(steps, couzin_polarization, label="Couzin", color="#003366", linewidth=1)
    axes[0].plot(steps, random_polarization, label="Random", color="#7EC8E3", linewidth=1)
    axes[0].axhline(expert_polarization, label="Expert", color="#000000", linewidth=1)

    axes[0].set_xlabel("Steps", fontsize=14)
    axes[0].set_ylabel("Polarization", fontsize=14)
    axes[0].set_title("Polarization Over Time", fontsize=18)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    ### ANGULAR MOMENTUM ###
    gail_am = [m["angular_momentum"] for m in gail_metrics]
    bc_am = [m["angular_momentum"] for m in bc_metrics]
    couzin_am = [m["angular_momentum"] for m in couzin_metrics]
    random_am = [m["angular_momentum"] for m in random_metrics]
    expert_am = np.mean(expert_metrics["angular_momentum"])
    max_am = max(max(gail_am), max(bc_am), max(couzin_am), max(random_am), expert_am)

    axes[1].plot(steps, gail_am, label="GAIL", color="#8B0000", linewidth=1)
    axes[1].plot(steps, bc_am, label="BC", color="#F08080", linewidth=1)
    axes[1].plot(steps, couzin_am, label="Couzin", color="#003366", linewidth=1)
    axes[1].plot(steps, random_am, label="Random", color="#7EC8E3", linewidth=1)
    axes[1].axhline(expert_am, label="Expert", color="#000000", linewidth=1)

    axes[1].set_xlabel("Steps", fontsize=14)
    axes[1].set_ylabel("Angular Momentum", fontsize=14)
    axes[1].set_title("Angular Momentum Over Time", fontsize=18)
    axes[1].set_ylim(0, max_am)
    axes[1].legend()

    ### DEGREE OF SPARSITY ###
    gail_dos = [m["degree_of_sparsity"] for m in gail_metrics]
    bc_dos = [m["degree_of_sparsity"] for m in bc_metrics]
    couzin_dos = [m["degree_of_sparsity"] for m in couzin_metrics]
    random_dos = [m["degree_of_sparsity"] for m in random_metrics]
    expert_dos = np.mean(expert_metrics["sparsity"])
    max_dos = max(max(gail_dos), max(bc_dos), max(couzin_dos), max(random_dos))

    axes[2].plot(steps, gail_dos, label="GAIL", color="#8B0000", linewidth=1) #005555
    axes[2].plot(steps, bc_dos, label="BC", color="#F08080", linewidth=1) #777777
    axes[2].plot(steps, couzin_dos, label="Couzin", color="#003366", linewidth=1) #a7a7a8
    axes[2].plot(steps, random_dos, label="Random", color="#7EC8E3", linewidth=1) #00b1ea
    axes[2].axhline(expert_dos, label="Expert", color="#000000", linewidth=1) #000000

    axes[2].set_xlabel("Steps", fontsize=14)
    axes[2].set_ylabel("Degree of Sparsity", fontsize=14)
    axes[2].set_title("Degree of Sparsity Over Time", fontsize=18)
    axes[2].set_ylim(0, max_dos)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def minmax_norm(dtp):
    dtp = np.asarray(dtp, dtype=np.float32)
    min, max = np.min(dtp), np.max(dtp)
    return (dtp - min) / (max - min + 1e-8)
    

def plot_pred_prey_metrics(gail_metrics, bc_metrics, couzin_metrics, random_metrics, expert_metrics):
    steps = np.arange(len(gail_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- DISTANCE TO PREDATOR ---
    gail_dtp = [m["distance_to_predator"] for m in gail_metrics]
    bc_dtp = [m["distance_to_predator"] for m in bc_metrics]
    random_dtp = [m["distance_to_predator"] for m in random_metrics]
    couzin_dtp = [m["distance_to_predator"] for m in couzin_metrics]

    gail_dtp = minmax_norm(gail_dtp)
    bc_dtp = minmax_norm(bc_dtp)
    random_dtp = minmax_norm(random_dtp)
    couzin_dtp = minmax_norm(couzin_dtp)

    expert_dtp = np.mean(expert_metrics["distance_to_predator"])

    axes[0].plot(steps, gail_dtp, label="GAIL", color="#8B0000", linewidth=1)
    axes[0].plot(steps, bc_dtp, label="BC", color="#F08080", linewidth=1)
    axes[0].plot(steps, couzin_dtp, label="Couzin", color="#003366", linewidth=1)
    axes[0].plot(steps, random_dtp, label="Random", color="#7EC8E3", linewidth=1)
    axes[0].axhline(expert_dtp, label="Expert", color="#000000", linewidth=1)

    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Distance to Predator")
    axes[0].set_title("Distance to Predator Over Time")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- ESCAPE ALIGNMENT ---
    gail_ea = [m["escape_alignment"] for m in gail_metrics]
    bc_ea = [m["escape_alignment"] for m in bc_metrics]
    couzin_ea = [m["escape_alignment"] for m in couzin_metrics]
    random_ea = [m["escape_alignment"] for m in random_metrics]
    expert_ea = np.mean(expert_metrics["escape_alignment"])

    axes[1].plot(steps, gail_ea, label="GAIL", color="#8B0000", linewidth=1)
    axes[1].plot(steps, bc_ea, label="BC", color="#F08080", linewidth=1)
    axes[1].plot(steps, couzin_ea, label="Couzin", color="#003366", linewidth=1)
    axes[1].plot(steps, random_ea, label="Random", color="#7EC8E3", linewidth=1)
    axes[1].axhline(expert_ea, label="Expert", color="#000000", linewidth=1)

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


def compute_pin_an_maps(pin, an, grid_size=100, n_orient=72, role="prey"):

    pin.to("cpu").eval()
    an.to("cpu").eval()

    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)

    thetas = torch.linspace(-np.pi, np.pi, n_orient+1, device="cpu")[:-1]
    rel_vx = 1 * torch.cos(thetas)
    rel_vy = 1 * torch.sin(thetas)

    action_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            dx = torch.full((n_orient, 1), float(x), device="cpu")
            dy = torch.full((n_orient, 1), float(y), device="cpu")
            inputs = torch.cat([dx, dy, rel_vx.unsqueeze(1), rel_vy.unsqueeze(1)], dim=1)

            if role == "prey":
                flag = torch.zeros((n_orient, 1), device="cpu")
                inputs = torch.cat([flag, dx, dy, rel_vx.unsqueeze(1), rel_vy.unsqueeze(1)], dim=1)

            # PIN: mu, sigma
            mu, sigma = pin(inputs)
            turn = torch.tanh(mu.squeeze())
            action_map[iy, ix] = turn.mean().item()

            # AN: attention logits/weights
            w_logits = an(inputs)
            w_logits = w_logits[0].squeeze()
            attn_map[iy, ix] = w_logits.mean().item()

    return xs, ys, action_map, attn_map



def plot_policy_maps(xs, ys, action_map, attn_map, role="predator", img_path=None):

    cmap_pin_color = "inferno"
    cmap_an_color = "RdBu"
    x, y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ----- Pairwise-Interaction Map -----
    scaled_action_map = action_map * 180
    vmax_act = np.nanmax(np.abs(scaled_action_map))
    norm_act = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_act, vmax=vmax_act)

    im0 = axes[0].contourf(x, y, scaled_action_map, levels=30, cmap=cmap_pin_color, norm=norm_act)
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


def plot_trajectory(metrics, role="Pred & Prey"):

    xs_pred = np.array([m["xs"][0] for m in metrics])
    ys_pred = np.array([m["ys"][0] for m in metrics])

    xs_prey = np.stack([m["xs"][1:] for m in metrics])
    ys_prey = np.stack([m["ys"][1:] for m in metrics])

    plt.figure(figsize=(6,6))

    if role in ('predator', 'Pred & Prey'):
        plt.plot(xs_pred, ys_pred, color='red', label='Predator')

    if role in ('prey', 'Pred & Prey'):
        _, n_agents = xs_prey.shape
        for i in range(n_agents):
            plt.plot(xs_prey[:, i], ys_prey[:, i], color='gray', alpha=0.6)

    plt.title(f'{role} - Trajectory Plot')
    plt.tight_layout()
    plt.show()



##########################
#####  SOCIAL ROLES  #####
##########################


def compute_incoming_weights(frame_metrics):
    weights_pred, weights_prey = frame_metrics["weights"]
    idx_info = frame_metrics["weights_idx"]

    pred_self = idx_info["pred"]["self"]
    pred_neighbors = np.array(idx_info["pred"]["neighbors"])
    prey_self = np.array(idx_info["prey"]["self"])
    prey_neighbors = np.array(idx_info["prey"]["neighbors"])

    incoming = np.zeros(33, dtype=float)

    for j, tgt in enumerate(pred_neighbors):
        incoming[tgt] += float(weights_pred[0, j])

    for r, focal in enumerate(prey_self):
        for c, tgt in enumerate(prey_neighbors[r]):
            incoming[tgt] += float(weights_prey[r, c])

    prey_incoming = incoming[1:]
    incoming_min = prey_incoming.min()
    incoming_max = prey_incoming.max()
    incoming_scaled = 0.2 + (prey_incoming - incoming_min) / (incoming_max - incoming_min) * (1.0 - 0.2) # Scale to alpha range for transparency
    incoming = np.insert(incoming_scaled, 0, incoming[0])

    incoming_dict = {f"predator_{pred_self}": 0.999} #{f"predator_{pred_self}": incoming[pred_self]}
    for i in prey_self:
        incoming_dict[f"prey_{i}"] = incoming[i]

    return incoming_dict



def draw_attention_graph(metrics, frame_idx=1, pred_img_path=None, prey_img_path=None):
    
    states = metrics[frame_idx]["global_state"]
    sorted_gs = dict(sorted(states.items())) # predator first
    arrays = list(sorted_gs.values())

    xs = np.array([a[1] for a in arrays], dtype=np.float32)  # [0, 1]
    ys = np.array([a[2] for a in arrays], dtype=np.float32)  # [0, 1]
    directions = np.array([a[3] for a in arrays], dtype=np.float32)  # [0, 1]

    # center the graph
    x_center = xs.mean()
    y_center = ys.mean()
    xs_centered = xs - x_center
    ys_centered = ys - y_center

    max_range = max(max(xs_centered), max(ys_centered), abs(min(xs_centered)), abs(min(ys_centered))) + 0.05 # automated scaling + margin

    n_agents = len(xs_centered)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            ax.plot([xs_centered[i], xs_centered[j]],
                    [ys_centered[i], ys_centered[j]],
                    color="gray", linewidth=1, alpha=0.2)

    # Nodes as Images
    pred_image = Image.open(pred_img_path).convert("RGBA")
    prey_image = Image.open(prey_img_path).convert("RGBA")

    alphas = compute_incoming_weights(metrics[frame_idx])  # to get incoming weights for transparency

    for i in range(n_agents):
        angle_rad = directions[i] * 2 * np.pi # Heading [0,1] to (0–360°)
        angle_deg = np.degrees(angle_rad)

        base_img = pred_image if i == 0 else prey_image

        alpha = alphas[f"predator_0"] if i == 0 else alphas[f"prey_{i}"]

        rotated_img = base_img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
        rotated_img = np.asarray(rotated_img).copy()

        # Mark leader prey red
        if alpha == 1.0:
            rotated_img = rotated_img.astype(float)
            rotated_img[..., 0] *= 1.8
            rotated_img[..., 1] *= 0.6
            rotated_img[..., 2] *= 0.6
            rotated_img = np.clip(rotated_img, 0, 255).astype(np.uint8)

        rotated_img[:, :, 3] = (rotated_img[:, :, 3].astype(float) * alpha).astype(np.uint8)

        imgbox = OffsetImage(rotated_img, zoom=0.45)
        imgbox.rotation = angle_deg
        ann_box = AnnotationBbox(imgbox, (xs_centered[i], ys_centered[i]), frameon=False, xycoords='data', zorder=2)
        ax.add_artist(ann_box)

    ax.set_title(f"Attention Graph (Frame {frame_idx})")
    ax.set_facecolor("#ACCEE7") # blue from marl_aquarium
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.show()



def draw_predator_attention_graph(metrics, frame_idx=1, pred_img_path=None, prey_img_path=None):
    
    states = metrics[frame_idx]["global_state"]
    sorted_gs = dict(sorted(states.items())) # predator first
    arrays = list(sorted_gs.values())

    xs = np.array([a[1] for a in arrays], dtype=np.float32)  # [0, 1]
    ys = np.array([a[2] for a in arrays], dtype=np.float32)  # [0, 1]
    directions = np.array([a[3] for a in arrays], dtype=np.float32)  # [0, 1]

    # center the graph
    x_center = xs.mean()
    y_center = ys.mean()
    xs_centered = xs - x_center
    ys_centered = ys - y_center

    max_range = max(max(xs_centered), max(ys_centered), abs(min(xs_centered)), abs(min(ys_centered))) + 0.05 # automated scaling + margin

    n_agents = len(xs_centered)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges
    for i in range(1): # only predator's edges
        for j in range(i+1, n_agents):
            ax.plot([xs_centered[i], xs_centered[j]],
                    [ys_centered[i], ys_centered[j]],
                    color="gray", linewidth=1, alpha=0.2)

    # Nodes as Images
    pred_image = Image.open(pred_img_path).convert("RGBA")
    prey_image = Image.open(prey_img_path).convert("RGBA")

    pred_weights = metrics[2]["weights"][0][0]

    weight_min = pred_weights.min()
    weight_max = pred_weights.max()
    weights_scaled = 0.2 + (pred_weights - weight_min) / (weight_max - weight_min) * (1.0 - 0.2) # Scale to alpha range for transparency

    weights_arr = weights_scaled.detach().cpu().numpy()

    for i in range(n_agents):
        angle_rad = directions[i] * 2 * np.pi # Heading [0,1] to (0–360°)
        angle_deg = np.degrees(angle_rad)

        base_img = pred_image if i == 0 else prey_image

        alpha = 0.999 if i == 0 else weights_arr[i-1]

        rotated_img = base_img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
        rotated_img = np.asarray(rotated_img).copy()

        if alpha == 1.0:
                rotated_img = rotated_img.astype(float)
                rotated_img[..., 0] *= 0.6
                rotated_img[..., 1] *= 0.6
                rotated_img[..., 2] *= 1.6
                rotated_img = np.clip(rotated_img, 0, 255).astype(np.uint8)

        rotated_img[:, :, 3] = (rotated_img[:, :, 3].astype(float) * alpha).astype(np.uint8)

        imgbox = OffsetImage(rotated_img, zoom=0.45)
        imgbox.rotation = angle_deg
        ann_box = AnnotationBbox(imgbox, (xs_centered[i], ys_centered[i]), frameon=False, xycoords='data', zorder=2)
        ax.add_artist(ann_box)

    ax.set_title(f"Predator Attention Graph (Frame {frame_idx})")
    ax.set_facecolor("#ACCEE7") # blue from marl_aquarium
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.show()