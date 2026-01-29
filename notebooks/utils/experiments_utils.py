import os
import cv2
import torch
import datetime
import numpy as np
import pandas as pd
from utils.sim_utils import *
from utils.eval_utils import *
from utils.train_utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


"""
References:
Some of the analysis derived from: Wu et al. (2025) - Adversarial imitation learning with deep attention network for swarm systems (https://doi.org/10.1007/s40747-024-01662-2)
Heat maps: https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolor_demo.html
Heat maps: https://matplotlib.org/stable/users/explain/colors/colormapnorms.html

Trajectory Plot (idea): https://www.researchgate.net/figure/Monte-Carlo-simulation-of-electron-trajectories-in-solid-TiC-red-lines-backscattered_fig3_334222567
"""


def compute_expert_data_ranges(expert_data):
    """
    Outputs the value ranges for x, y, vx, vy from expert data
    Needed for normalization and undistorted data plotting

    Input: expert data
    Output: value ranges for x, y, vx, vy
    """

    all_x = []
    all_y = []
    all_vx = []
    all_vy = []

    for video in expert_data.keys():
        # get data
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        all_x.append(xs)
        all_y.append(ys)
        all_vx.append(vxs)
        all_vy.append(vys)

    # concatenate all data
    all_x  = np.concatenate(all_x)
    all_y  = np.concatenate(all_y)
    all_vx = np.concatenate(all_vx)
    all_vy = np.concatenate(all_vy)

    # compute min/max
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    vx_min, vx_max = float(all_vx.min()), float(all_vx.max())
    vy_min, vy_max = float(all_vy.min()), float(all_vy.max())

    return (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max)


def compute_expert_metrics(expert_data, num_agents):
    """
    Calculates swarm metrics from expert data

    Input: expert data, number of agents
    Output: polarization, angular momentum, degree of sparsity, distance to predator, escape alignment
    """

    polarizations = []
    angular_momenta = []
    sparsities = []
    distances_to_predator = []
    distance_nearest_prey = []
    escape_alignments = []

    # compute data ranges for normalization
    x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max = compute_expert_data_ranges(expert_data)

    for video in expert_data.keys():
        # get data
        xs  = np.asarray(expert_data[video]["xs"])
        ys  = np.asarray(expert_data[video]["ys"])
        vxs = np.asarray(expert_data[video]["vxs"])
        vys = np.asarray(expert_data[video]["vys"])

        # normalize data to [-1, 1]
        vxs = 2 * (vxs - vx_min) / (vx_max - vx_min) -1
        vys = 2 * (vys - vy_min) / (vy_max - vy_min) -1

        # reshape data
        time = len(xs) // num_agents
        xs  = xs.reshape(time, num_agents)
        ys  = ys.reshape(time, num_agents)
        vxs = vxs.reshape(time, num_agents)
        vys = vys.reshape(time, num_agents)

        for t in range(time):
            # get positions and velocities
            x_t  = xs[t]
            y_t  = ys[t]
            vx_t = vxs[t]
            vy_t = vys[t]

            # compute metrics
            polarizations.append(compute_polarization(vx_t, vy_t))
            angular_momenta.append(compute_angular_momentum(x_t, y_t, vx_t, vy_t))
            sparsities.append(degree_of_sparsity(x_t, y_t))
            distances_to_predator.append(distance_to_predator(x_t, y_t))
            distance_nearest_prey.append(pred_distance_to_nearest_prey(x_t, y_t))
            escape_alignments.append(escape_alignment(x_t, y_t, vx_t, vy_t))

    return {"polarization": polarizations,
            "angular_momentum": angular_momenta,
            "sparsity": sparsities,
            "distance_to_predator": distances_to_predator,
            "distance_nearest_prey": distance_nearest_prey,
            "escape_alignment": escape_alignments}


def minmax_norm(data, min=None, max=None):
    """
    Applies min-max normalization (for visualization purposes)

    Input: data
    Output: normalized data range [0, 1]
    """
    data = np.asarray(data, dtype=np.float32)
    if min is None or max is None:
        min, max = data.min(), data.max()
    return (data - min) / (max - min + 1e-8)


def plot_swarm_metrics(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None):
    """
    Plots swarm metrics over time to compare different models
    Code allows dynamic inclusion/exclusion of different models

    Input: model metrics
    Output: plots of polarization, angular momentum, degree of sparsity over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))
    fig, axes = plt.subplots(1, 3, figsize=(24, 5))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    # get polarization data
    gail_polarization = [m.get("polarization") for m in gail_metrics if "polarization" in m]
    bc_polarization   = [m.get("polarization") for m in bc_metrics   if "polarization" in m]
    couzin_polarization = [m.get("polarization") for m in couzin_metrics if "polarization" in m]
    random_polarization = [m.get("polarization") for m in random_metrics if "polarization" in m]
    expert_polarization = np.mean(expert_metrics["polarization"]) if "polarization" in expert_metrics else None
    expert_polarization_std = np.std(expert_metrics["polarization"]) if "polarization" in expert_metrics else None

    # plot polarization
    axes[0].plot(steps, gail_polarization, label="GAIL", color="#8B0000", linewidth=1) if len(gail_polarization) > 0 else None
    axes[0].plot(steps, bc_polarization, label="BC", color="#F08080", linewidth=1) if len(bc_polarization) > 0 else None
    axes[0].plot(steps, couzin_polarization, label="Couzin", color="#003366", linewidth=1) if len(couzin_polarization) > 0 else None
    axes[0].plot(steps, random_polarization, label="Random", color="#7EC8E3", linewidth=1) if len(random_polarization) > 0 else None
    axes[0].axhline(expert_polarization, label="Expert", color="#000000", linewidth=1) if expert_polarization is not None else None
    axes[0].axhspan(expert_polarization - expert_polarization_std, expert_polarization + expert_polarization_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_polarization is not None and expert_polarization_std is not None) else None
    axes[0].set_xlabel("Steps", fontsize=14)
    axes[0].set_ylabel("Polarization", fontsize=14)
    axes[0].set_title("Polarization Over Time", fontsize=18)
    axes[0].set_xlim(0, steps[-1])
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # get angular momentum data
    gail_am = [m.get("angular_momentum") for m in gail_metrics if "angular_momentum" in m] if len(gail_metrics)>0 else []
    bc_am = [m.get("angular_momentum") for m in bc_metrics if "angular_momentum" in m] if len(bc_metrics)>0 else []
    couzin_am = [m.get("angular_momentum") for m in couzin_metrics if "angular_momentum" in m] if len(couzin_metrics)>0 else []
    random_am = [m.get("angular_momentum") for m in random_metrics if "angular_momentum" in m] if len(random_metrics)>0 else []
    expert_am = np.mean(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None
    expert_am_std = np.std(expert_metrics["angular_momentum"]) * 2160 if "angular_momentum" in expert_metrics else None

    # plot angular momentum
    axes[1].plot(steps, gail_am, label="GAIL", color="#8B0000", linewidth=1) if (gail_am is not None and len(gail_am) > 0) else None
    axes[1].plot(steps, bc_am, label="BC", color="#F08080", linewidth=1) if (bc_am is not None and len(bc_am) > 0) else None
    axes[1].plot(steps, couzin_am, label="Couzin", color="#003366", linewidth=1) if (couzin_am is not None and len(couzin_am) > 0) else None
    axes[1].plot(steps, random_am, label="Random", color="#7EC8E3", linewidth=1) if (random_am is not None and len(random_am) > 0) else None
    axes[1].axhline(expert_am, label="Expert", color="#000000", linewidth=1) if expert_am is not None else None
    axes[1].axhspan(expert_am - expert_am_std, expert_am + expert_am_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_am is not None and expert_am_std is not None) else None
    axes[1].set_xlabel("Steps", fontsize=14)
    axes[1].set_ylabel("Angular Momentum", fontsize=14)
    axes[1].set_title("Angular Momentum Over Time", fontsize=18)
    axes[1].set_xlim(0, steps[-1])
    axes[1].legend()

    # get degree of sparsity data
    gail_dos = [m.get("degree_of_sparsity") for m in gail_metrics if "degree_of_sparsity" in m] if len(gail_metrics)>0 else []
    bc_dos = [m.get("degree_of_sparsity") for m in bc_metrics if "degree_of_sparsity" in m] if len(bc_metrics)>0 else []
    couzin_dos = [m.get("degree_of_sparsity") for m in couzin_metrics if "degree_of_sparsity" in m] if len(couzin_metrics)>0 else []
    random_dos = [m.get("degree_of_sparsity") for m in random_metrics if "degree_of_sparsity" in m] if len(random_metrics)>0 else []
    expert_dos = np.mean(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None
    expert_dos_std = np.std(expert_metrics["sparsity"]) * 2160 if "sparsity" in expert_metrics else None

    # plot degree of sparsity
    axes[2].plot(steps, gail_dos, label="GAIL", color="#8B0000", linewidth=1) if len(gail_dos) > 0 else None
    axes[2].plot(steps, bc_dos, label="BC", color="#F08080", linewidth=1) if len(bc_dos) > 0 else None
    axes[2].plot(steps, couzin_dos, label="Couzin", color="#003366", linewidth=1) if len(couzin_dos) > 0 else None 
    axes[2].plot(steps, random_dos, label="Random", color="#7EC8E3", linewidth=1) if len(random_dos) > 0 else None 
    axes[2].axhline(expert_dos, label="Expert", color="#000000", linewidth=1) if expert_dos is not None else None 
    axes[2].axhspan(expert_dos - expert_dos_std, expert_dos + expert_dos_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dos is not None and expert_dos_std is not None) else None
    axes[2].set_xlabel("Steps", fontsize=14)
    axes[2].set_ylabel("Degree of Sparsity", fontsize=14)
    axes[2].set_title("Degree of Sparsity Over Time", fontsize=18)
    axes[2].set_xlim(0, steps[-1])
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return (expert_polarization, expert_polarization_std, expert_am, expert_am_std, expert_dos, expert_dos_std)


def plot_pred_prey_metrics(gail_metrics=None, bc_metrics=None, couzin_metrics=None, random_metrics=None, expert_metrics=None):
    """
    Plots predator-related metrics over time to compare different models
    Code allows dynamic inclusion/exclusion of different models

    Input: model metrics
    Output: plots of distance to predator and escape alignment over time
    """

    # time dimension
    steps = np.arange(len(gail_metrics[0]))

    # get metrics from simulation
    gail_metrics = gail_metrics[0] if gail_metrics is not None else []
    bc_metrics = bc_metrics[0] if bc_metrics is not None else []
    couzin_metrics = couzin_metrics if couzin_metrics is not None else []
    random_metrics = random_metrics[0] if random_metrics is not None else []
    expert_metrics = expert_metrics if expert_metrics is not None else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # get distance to predator data
    gail_dtp = [m.get("distance_to_predator") for m in gail_metrics if "distance_to_predator" in m] 
    bc_dtp = [m.get("distance_to_predator") for m in bc_metrics if "distance_to_predator" in m]
    random_dtp = [m.get("distance_to_predator") for m in random_metrics if "distance_to_predator" in m]
    couzin_dtp = [m.get("distance_to_predator") for m in couzin_metrics if "distance_to_predator" in m]
    expert_dtp = np.mean(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None
    expert_dtp_std = np.std(expert_metrics["distance_to_predator"]) * 2160 if "distance_to_predator" in expert_metrics else None

    # plot distance to predator
    axes[0].plot(steps, gail_dtp, label="GAIL", color="#8B0000", linewidth=1) if len(gail_dtp) > 0 else None
    axes[0].plot(steps, bc_dtp, label="BC", color="#F08080", linewidth=1) if len(bc_dtp) > 0 else None
    axes[0].plot(steps, couzin_dtp, label="Couzin", color="#003366", linewidth=1) if len(couzin_dtp) > 0 else None
    axes[0].plot(steps, random_dtp, label="Random", color="#7EC8E3", linewidth=1) if len(random_dtp) > 0 else None
    axes[0].axhline(expert_dtp, label="Expert", color="#000000", linewidth=1) if expert_dtp is not None else None
    axes[0].axhspan(expert_dtp - expert_dtp_std, expert_dtp + expert_dtp_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_dtp is not None and expert_dtp_std is not None) else None
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Distance to Predator")
    axes[0].set_title("Distance to Predator Over Time")
    axes[0].set_ylim(0, 2160)
    axes[0].legend()


    # get nearest prey distance data
    gail_pnd = [m.get("distance_nearest_prey") for m in gail_metrics if "distance_nearest_prey" in m]
    bc_pnd = [m.get("distance_nearest_prey") for m in bc_metrics   if "distance_nearest_prey" in m]
    couzin_pnd = [m.get("distance_nearest_prey") for m in couzin_metrics if "distance_nearest_prey" in m]
    random_pnd = [m.get("distance_nearest_prey") for m in random_metrics if "distance_nearest_prey" in m]
    expert_pnd = np.mean(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None
    expert_pnd_std = np.std(np.asarray(expert_metrics["distance_nearest_prey"], dtype=float) * 2160) if "distance_nearest_prey" in expert_metrics else None

    # plot nearest prey distance
    axes[1].plot(steps, gail_pnd, label="GAIL", color="#8B0000", linewidth=1) if len(gail_pnd) > 0 else None
    axes[1].plot(steps, bc_pnd, label="BC", color="#F08080", linewidth=1) if len(bc_pnd) > 0 else None
    axes[1].plot(steps, couzin_pnd, label="Couzin", color="#003366", linewidth=1) if len(couzin_pnd) > 0 else None
    axes[1].plot(steps, random_pnd, label="Random", color="#7EC8E3", linewidth=1) if len(random_pnd) > 0 else None
    axes[1].axhline(expert_pnd, label="Expert", color="#000000", linewidth=1) if expert_pnd is not None else None
    axes[1].axhspan(expert_pnd - expert_pnd_std, expert_pnd + expert_pnd_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_pnd is not None and expert_pnd_std is not None) else None
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Nearest Prey Distance")
    axes[1].set_title("Predator Distance to Nearest Prey")
    axes[1].set_xlim(0, steps[-1])
    axes[1].set_ylim(0, 2160)
    axes[1].legend()


    # get escape alignment data
    gail_ea = [m.get("escape_alignment") for m in gail_metrics if "escape_alignment" in m]
    bc_ea = [m.get("escape_alignment") for m in bc_metrics if "escape_alignment" in m]
    couzin_ea = [m.get("escape_alignment") for m in couzin_metrics if "escape_alignment" in m]
    random_ea = [m.get("escape_alignment") for m in random_metrics if "escape_alignment" in m]
    expert_ea = np.mean(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None
    expert_ea_std = np.std(expert_metrics["escape_alignment"]) if "escape_alignment" in expert_metrics else None

    # plot escape alignment
    axes[2].plot(steps, gail_ea, label="GAIL", color="#8B0000", linewidth=1) if len(gail_ea) > 0 else None
    axes[2].plot(steps, bc_ea, label="BC", color="#F08080", linewidth=1) if len(bc_ea) > 0 else None
    axes[2].plot(steps, couzin_ea, label="Couzin", color="#003366", linewidth=1) if len(couzin_ea) > 0 else None
    axes[2].plot(steps, random_ea, label="Random", color="#7EC8E3", linewidth=1) if len(random_ea) > 0 else None
    axes[2].axhline(expert_ea, label="Expert", color="#000000", linewidth=1) if expert_ea is not None else None
    axes[2].axhspan(expert_ea - expert_ea_std, expert_ea + expert_ea_std, color="#5E5A5A", alpha=0.12, linewidth=0) if (expert_ea is not None and expert_ea_std is not None) else None
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Escape Alignment")
    axes[2].set_title("Escape Alignment Over Time")
    axes[2].set_xlim(0, steps[-1])
    axes[2].set_ylim(-1, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return (expert_dtp, expert_dtp_std, expert_pnd, expert_pnd_std, expert_ea, expert_ea_std)



##########################
##### ATTENTION MAPS #####
##########################


def compute_pin_an_maps(pin, an, grid_size=100, n_orient=100, role="prey_pred"):
    """
    Computes policy and attention maps for PIN and AN models

    Input: PIN and AN, grid size, number of orientations, role
    Output: xs, ys, action map, attention map
    """

    pin.to("cpu").eval()
    an.to("cpu").eval()

    # relative positions grid
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)

    # sample relative velocity directions
    thetas = torch.linspace(-np.pi, np.pi, n_orient+1, device="cpu")[:-1]
    rel_vx = 1 * torch.cos(thetas)
    rel_vy = 1 * torch.sin(thetas)

    # initialize maps
    action_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    attn_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # build input tensor for this grid cell
            dx = torch.full((n_orient, 1), float(x), device="cpu")
            dy = torch.full((n_orient, 1), float(y), device="cpu")
            inputs = torch.cat([dx, dy, rel_vx.unsqueeze(1), rel_vy.unsqueeze(1)], dim=1)

            if role == "prey":
                # to handle flag feature for prey
                flag = torch.zeros((n_orient, 1), device="cpu")
                inputs = torch.cat([flag, dx, dy, rel_vx.unsqueeze(1), rel_vy.unsqueeze(1)], dim=1)

            if role == "prey_pred": # for prey-pred one-to-one relationship
                # to handle flag feature for prey
                flag = torch.ones((n_orient, 1), device="cpu")
                inputs = torch.cat([flag, dx, dy, rel_vx.unsqueeze(1), rel_vy.unsqueeze(1)], dim=1)

            # PIN forward, tanh for better visualization
            mu, sigma = pin(inputs)
            turn = torch.tanh(mu.squeeze())
            action_map[iy, ix] = turn.mean().item()

            # AN forward, attention logits/weights
            w_logits = an(inputs)
            w_logits = w_logits[0].squeeze()
            attn_map[iy, ix] = w_logits.mean().item()

            # scale attention weights to [0, 1]
            attention_map = minmax_norm(attn_map)

    return xs, ys, action_map, attention_map



def plot_policy_maps(xs, ys, action_map, attn_map, role="predator", img_path=None):
    """
    Plots policy and attention maps for PIN and AN models

    Input: xs, ys, action map, attention map, role, image path
    Output: plots of policy and attention maps
    """

    # color palettes
    cmap_pin_color = "inferno"
    cmap_an_color = "RdBu"

    # create meshgrid for plotting
    x, y = np.meshgrid(xs, ys)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # scale to degrees for better interpretability
    scaled_action_map = action_map * 180
    vmax_act = np.nanmax(np.abs(scaled_action_map)) # symmetric color scale
    norm_act = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_act, vmax=vmax_act) # centered at 0

    # plot PIN map
    im0 = axes[0].contourf(x, y, scaled_action_map, levels=30, cmap=cmap_pin_color, norm=norm_act)
    axes[0].set_title(f"[{role.upper()}] Pairwise-Interaction Map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], label="action")


    vmax_att = np.nanmax(np.abs(attn_map)) # symmetric color scale
    norm_att = colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_att, vmax=vmax_att) # centered at 0

    # plot AN map
    im1 = axes[1].contourf(x, y, attn_map, levels=30, cmap=cmap_an_color, norm=norm_att)
    axes[1].set_title(f"[{role.upper()}] Attention Map")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], label="attention")

    # set same axis ranges
    x_range = (xs.min(), xs.max())
    y_range = (ys.min(), ys.max())
    for ax in axes:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    # add agent icon in center (alignment of picture and action map is correct)
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
    """
    Plots the trajectories of predator and prey agents

    Input: metrics, role
    Output: plots of trajectories
    """

    metrics = metrics[0]

    # extract pred positions
    xs_pred = np.array([m["xs"][0] for m in metrics])
    ys_pred = np.array([m["ys"][0] for m in metrics])

    # extract prey positions
    xs_prey = np.stack([m["xs"][1:] for m in metrics])
    ys_prey = np.stack([m["ys"][1:] for m in metrics])

    plt.figure(figsize=(6,6))

    # plot trajectories of predator
    if role in ('predator', 'Pred & Prey'):
        plt.plot(xs_pred, ys_pred, color='red', label='Predator')

    # plot trajectories of prey
    if role in ('prey', 'Pred & Prey'):
        _, n_agents = xs_prey.shape
        for i in range(n_agents):
            plt.plot(xs_prey[:, i], ys_prey[:, i], color='gray', alpha=0.6)

    plt.title(f'{role} - Trajectory Plot')
    plt.tight_layout()
    plt.show()



###################################
#####  Trajectory Prediction  #####
###################################

def trajectory_offsets(pred_policy, prey_policy, init_pool, mc_samples=50, clip_idx=0, role="prey"):
    """
    Plots the trajectory prediction errors over time

    Input: policies, initial pool, Monte Carlo samples, clip index
    Output: plot of trajectory prediction errors
    """

    # reshape init pool to (n_clips, clip_len, n_agents, 3)
    clips = init_pool.view(24, 10, 33, 3)

    # select clip
    traj = int(clip_idx)
    clip = clips[traj]
    init_pos = clip[0].clone()

    metric_list = []

    # Monte Carlo sampling
    for mc in range(mc_samples):
        _, _, metrics = run_env_simulation(visualization='off',
                                           init_pool=init_pos, experiment=True,
                                           prey_policy=prey_policy,
                                           pred_policy=pred_policy,
                                           n_prey=32, n_pred=1,
                                           max_steps=10,
                                           pred_speed=10, prey_speed=10,
                                           area_width=2160, area_height=2160,
                                           max_turn=0.314,
                                           step_size=1.0)
        
        metric_list.append(metrics[0])

    # extract generated and expert trajectories
    if role == "prey":
        gen_xs = np.array([[step["xs"][1:] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)
        gen_ys = np.array([[step["ys"][1:] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)
        gen_thetas = np.array([[step["theta"][1:] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)
    else:
        gen_xs = np.array([[step["xs"][:1] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)
        gen_ys = np.array([[step["ys"][:1] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)
        gen_thetas = np.array([[step["theta"][:1] for step in rollout_metrics] for rollout_metrics in metric_list], dtype=np.float32)

    # extract expert trajectories
    if role == "prey":
        exp_xs = clip[:, 1:, 0].detach().cpu().numpy().astype(np.float32)
        exp_ys = clip[:, 1:, 1].detach().cpu().numpy().astype(np.float32)
        exp_thetas = clip[:, 1:, 2].detach().cpu().numpy().astype(np.float32)
    else:
        exp_xs = clip[:, :1, 0].detach().cpu().numpy().astype(np.float32)
        exp_ys = clip[:, :1, 1].detach().cpu().numpy().astype(np.float32)
        exp_thetas = clip[:, :1, 2].detach().cpu().numpy().astype(np.float32)

    # compute position error
    dx = gen_xs - exp_xs[None, :, :]
    dy = gen_ys - exp_ys[None, :, :]
    position_error = np.sqrt(dx**2 + dy**2)            

    # compute theta error, handling angle wrapping correctly
    gen_theta_deg = (np.rad2deg(gen_thetas) + 180) % 360 - 180
    exp_theta_deg = (np.rad2deg(exp_thetas) + 180) % 360 - 180
    theta_diff = (gen_theta_deg - exp_theta_deg[None, :, :] + 180) % 360 - 180
    theta_error = np.abs(theta_diff)                   

    # mean over agents
    position_error_agents = position_error.mean(axis=2) 
    theta_error_agents = theta_error.mean(axis=2)       

    # mean/std over MC samples, scale to 2160 pixels
    position_mean = position_error_agents.mean(axis=0) * 2160
    position_std  = position_error_agents.std(axis=0) * 2160
    theta_mean = theta_error_agents.mean(axis=0) 
    theta_std  = theta_error_agents.std(axis=0) 

    # print results per step
    for step in range(position_mean.shape[0]):
        print(f"Step {step} | Position Error: {position_mean[step]:.4f} ± {position_std[step]:.4f} | "
              f"Theta Error: {theta_mean[step]:.2f}° ± {theta_std[step]:.2f}°")

    # plot errors over time
    time = np.arange(position_mean.shape[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)

    # position error plot
    ax1.plot(time, position_mean)
    ax1.fill_between(time, position_mean - position_std, position_mean + position_std, alpha=0.12)
    ax1.axhline(0, color="red", linewidth=3)
    ax1.grid(True, linewidth=0.8, alpha=0.5)
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(0)
    ax1.set_xlabel("timestep")
    ax1.set_ylabel("mean position error (in pixels)")
    ax1.set_title(f"[{role.upper()}] Position error over time")

    # theta error plot
    ax2.plot(time, theta_mean)
    ax2.fill_between(time, theta_mean - theta_std, theta_mean + theta_std, alpha=0.12)
    ax2.axhline(0, color="red", linewidth=3)
    ax2.grid(True, linewidth=0.8, alpha=0.5)
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(0)
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("mean theta error (in degrees)")
    ax2.set_title(f"[{role.upper()}] Theta error over time")

    plt.tight_layout()
    plt.show()

    return (gen_xs, gen_ys, gen_thetas, exp_xs, exp_ys, exp_thetas)



def align_to_start_heading(x, y, theta):
    """
    Aligns trajectory to start at origin and heading upwards

    Input: x, y, theta
    Output: aligned x, y
    """

    # convert to relative coordinates with start at (0,0)
    start_x = x[0]
    start_y = y[0]
    x_norm = x - start_x
    y_norm = y - start_y

    # rotate so the heading points up (positive y-axis)
    rotation_angle = (np.pi / 2.0) - theta
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)

    # apply rotation
    x_rotated = cos_angle * x_norm - sin_angle * y_norm
    y_rotated = sin_angle * x_norm + cos_angle * y_norm

    return x_rotated, y_rotated


def trajectory_plot(ax, agent_idx, role, positions, scale=5):
    """
    Plots trajectories for a specific agent

    Input: x, y, theta
    Output: aligned x, y
    """

    # unpack positions and get dimensions
    gen_xs, gen_ys, gen_thetas, exp_xs, exp_ys, exp_thetas = positions
    n_rollouts, agents, coordinates = gen_xs.shape

    # prepare rollouts
    rollouts = np.arange(n_rollouts)

    all_x = []
    all_y = []

    # draw generated trajectories in grey
    for r in rollouts:
        # get x, y, theta for this agent, scale to environment size
        x = np.asarray(gen_xs[r, :, agent_idx], dtype=np.float32) * 2160
        y = np.asarray(gen_ys[r, :, agent_idx], dtype=np.float32) * 2160
        theta = float(gen_thetas[r, 0, agent_idx])

        # align trajectory
        x_rotated, y_rotated = align_to_start_heading(x, y, theta)
        all_x.append(x_rotated)
        all_y.append(y_rotated)

        # plot generated trajectories
        ax.plot(x_rotated, y_rotated, color="0.55", linewidth=0.9, alpha=0.25, zorder=1)

    # draw expert trajectory in red, scale is necessary to fix step size differences
    exp_x = np.asarray(exp_xs[:, agent_idx], dtype=np.float32) * scale * 2160
    exp_y = np.asarray(exp_ys[:, agent_idx], dtype=np.float32) * scale * 2160
    exp_theta = float(exp_thetas[0, agent_idx])

    # align expert trajectory
    align_exp_x, align_exp_y = align_to_start_heading(exp_x, exp_y, exp_theta)
    all_x.append(align_exp_x) 
    all_y.append(align_exp_y)

    # plot expert trajectory
    ax.plot(align_exp_x, align_exp_y, color="red", linewidth=0.9, alpha=1.0, zorder=10)

    # set startpoint
    ax.scatter([0.0], [0.0], s=20, color="black", zorder=12)
    ax.set_title(role.upper())
    ax.set_xlabel("x (in pixels)")
    ax.set_ylabel("y (in pixels)")
    ax.set_aspect("auto")

    # widen x-limits and y-limits based on data
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)

    # determine limits
    x_abs = float(np.max(np.abs(x_all))) if x_all.size else 0.0
    y_min = float(np.min(y_all)) if y_all.size else 0.0
    y_max = float(np.max(y_all)) if y_all.size else 1.0

    # set limits with some padding (mainly due to size differences)
    x_span = max(0.10, 1.15 * x_abs)
    ax.set_xlim(-x_span, x_span)

    # set y-limits with padding (mainly due to size differences)
    y_pad = 0.02 * (y_max - y_min + 1e-9)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)



##########################
#####  SOCIAL ROLES  #####
##########################

def compute_incoming_weights(frame_metrics):
    """
    Computes incoming attention weights for each agent
    Outgoing weights are always one, to get attention of fish in comparison to others, incoming weights are needed

    Input: frame metrics
    Output: incoming weights dictionary
    """

    # extract weights and indices info
    weights_pred, weights_prey = frame_metrics["weights"]
    idx_info = frame_metrics["weights_idx"]

    # extract indices
    pred_self = idx_info["pred"]["self"]
    pred_neighbors = np.array(idx_info["pred"]["neighbors"])
    prey_self = np.array(idx_info["prey"]["self"])
    prey_neighbors = np.array(idx_info["prey"]["neighbors"])

    incoming = np.zeros(33, dtype=float)

    # predator incoming weights
    for pred_k, neighbor_id in enumerate(pred_neighbors):
        incoming[neighbor_id] += float(weights_pred[0, pred_k])

    # prey incoming weights
    for focal_k, focal_id in enumerate(prey_self):
        for nbr_k, neighbor_id in enumerate(prey_neighbors[focal_k]):
            incoming[neighbor_id] += float(weights_prey[focal_k, nbr_k])

    # build incoming weights dictionary
    incoming_dict = {f"predator_{pred_self}": 0.999} # predator always visible
    for i in prey_self:
        incoming_dict[f"prey_{i}"] = incoming[i]

    # normalize incoming weights with power renormalization, necessary due to weight imbalance
    keys = list(incoming_dict.keys())
    weights = np.array([float(incoming_dict[k]) for k in keys], dtype=np.float64)
    clipped_weights = np.clip(weights, 0.0, None)

    # power renormalization with exponent 0.001
    powered = (clipped_weights + 1e-12) ** 0.001
    powered_weights = powered / (powered.sum() + 1e-12)

    # scale prey incoming weights to [0.2, 1.0] for better visualization, if 0 fish invisible
    prey_incoming = powered_weights[1:]
    incoming_min = prey_incoming.min()
    incoming_max = prey_incoming.max()
    incoming_scaled = 0.2 + (prey_incoming - incoming_min) / (incoming_max - incoming_min) * (1.0 - 0.2) # scale to alpha range for transparency

    # build final incoming weights dictionary
    final_dict = {f"predator_{pred_self}": 0.999}
    for idx, i in enumerate(prey_self):
        final_dict[f"prey_{i}"] = float(incoming_scaled[idx])

    return final_dict


def draw_attention_graph(metrics_weights, frame_idx=1, pred_img_path=None, prey_img_path=None):
    """
    Draws attention graph for a specific frame

    Input: weights, frame index, predator image path, prey image path
    Output: attention graph plot
    """

    # get weights and metrics for the specified frame
    metrics_list, weights_list = metrics_weights
    metrics = metrics_list[frame_idx]
    weights = weights_list[frame_idx]    

    # extract positions and directions
    xs = np.array(metrics["xs"])
    ys = np.array(metrics["ys"])
    directions = np.array(metrics["theta"])

    # center the graph
    x_center = xs.mean()
    y_center = ys.mean()
    xs_centered = xs - x_center
    ys_centered = ys - y_center

    # determine max range for scaling
    max_range = max(max(xs_centered), max(ys_centered), abs(min(xs_centered)), abs(min(ys_centered))) + 0.05 # automated scaling + margin

    n_agents = len(xs_centered)

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw edges
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            ax.plot([xs_centered[i], xs_centered[j]],
                    [ys_centered[i], ys_centered[j]],
                    color="gray", linewidth=1, alpha=0.2)

    # change nodes to images
    pred_image = Image.open(pred_img_path).convert("RGBA")
    prey_image = Image.open(prey_img_path).convert("RGBA")

    # compute incoming weights for transparency
    alphas = compute_incoming_weights(weights)  # to get incoming weights for transparency

    for i in range(n_agents):
        # compute rotation angle in degrees
        angle_deg = np.degrees(directions[i])

        # select base image
        base_img = pred_image if i == 0 else prey_image

        # get alpha value
        alpha = alphas[f"predator_0"] if i == 0 else alphas[f"prey_{i}"]

        # rotate image to match direction
        rotated_img = base_img.rotate(angle_deg+180, resample=Image.BICUBIC, expand=True)
        rotated_img = np.asarray(rotated_img).copy()

        # mark leader prey
        if alpha == 1.0:
            rotated_img = rotated_img.astype(float)
            rotated_img[..., 0] *= 1.8
            rotated_img[..., 1] *= 0.6
            rotated_img[..., 2] *= 0.6
            rotated_img = np.clip(rotated_img, 0, 255).astype(np.uint8)

        # apply transparency
        rotated_img[:, :, 3] = (rotated_img[:, :, 3].astype(float) * alpha).astype(np.uint8)

        # add image to plot
        imgbox = OffsetImage(rotated_img, zoom=0.45)
        imgbox.rotation = angle_deg
        ann_box = AnnotationBbox(imgbox, (xs_centered[i], ys_centered[i]), frameon=False, xycoords='data', zorder=2)
        ax.add_artist(ann_box)

    ax.set_title(f"Attention Graph")
    ax.set_facecolor("#ACCEE7") # blue from marl_aquarium
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.show()



def draw_predator_attention_graph(metrics_weights, frame_idx=1, pred_img_path=None, prey_img_path=None):
    """
    Draws attention graph for a specific frame, focusing on predator's attention

    Input: weights, frame index, predator image path, prey image path
    Output: predator attention graph plot
    """
    
    # get weights and metrics for the specified frame
    metrics_list, weights_list = metrics_weights
    metrics = metrics_list[frame_idx]
    weights = weights_list[frame_idx]    

    # extract positions and directions
    xs = np.array(metrics["xs"])
    ys = np.array(metrics["ys"])
    directions = np.array(metrics["theta"])

    # center the graph
    x_center = xs.mean()
    y_center = ys.mean()
    xs_centered = xs - x_center
    ys_centered = ys - y_center

    # determine max range for scaling
    max_range = max(max(xs_centered), max(ys_centered), abs(min(xs_centered)), abs(min(ys_centered))) + 0.05 # automated scaling + margin

    n_agents = len(xs_centered)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges
    for i in range(1): # only predator's edges
        for j in range(i+1, n_agents):
            ax.plot([xs_centered[i], xs_centered[j]],
                    [ys_centered[i], ys_centered[j]],
                    color="gray", linewidth=1, alpha=0.2)

    # change nodes to images
    pred_image = Image.open(pred_img_path).convert("RGBA")
    prey_image = Image.open(prey_img_path).convert("RGBA")

    # compute incoming weights for transparency
    alphas = compute_incoming_weights(weights)

    for i in range(n_agents):
        # compute rotation angle in degrees
        angle_deg = np.degrees(directions[i])

        # select base image
        base_img = pred_image if i == 0 else prey_image

        # get alpha value from incoming-weight dict
        alpha = alphas[f"predator_0"] if i == 0 else alphas[f"prey_{i}"]

        # rotate image to match direction
        rotated_img = base_img.rotate(angle_deg+180, resample=Image.BICUBIC, expand=True)
        rotated_img = np.asarray(rotated_img).copy()

        # mark leader prey (use threshold instead of == 1.0)
        if i != 0 and alpha >= 0.999:
            rotated_img = rotated_img.astype(float)
            rotated_img[..., 0] *= 1.8
            rotated_img[..., 1] *= 0.6
            rotated_img[..., 2] *= 0.6
            rotated_img = np.clip(rotated_img, 0, 255).astype(np.uint8)

        # apply transparency
        rotated_img[:, :, 3] = (rotated_img[:, :, 3].astype(float) * alpha).astype(np.uint8)

        # add image to plot
        imgbox = OffsetImage(rotated_img, zoom=0.45)
        imgbox.rotation = angle_deg
        ann_box = AnnotationBbox(imgbox, (xs_centered[i], ys_centered[i]), frameon=False, xycoords='data', zorder=2)
        ax.add_artist(ann_box)

    ax.set_title(f"Predator Attention Graph")
    ax.set_facecolor("#ACCEE7")
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.tight_layout()
    plt.show()



def make_video(frames_dir):
    """
    Makes a video to see attention graph changes over time

    Input: frames directory
    Output: video file
    """

    # gather frame files
    files = sorted(os.listdir(frames_dir), reverse=True)
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    height, width, _ = first.shape

    # create video writer
    video = cv2.VideoWriter(os.path.join(frames_dir, "video.mp4"), 
                            cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

    for f in files:
        # read each frame and write to video
        frame = cv2.imread(os.path.join(frames_dir, f))
        video.write(frame)

    video.release()



def record_attn_graph_video(metrics_weights, num_steps=50, pred_img_path=None, prey_img_path=None, save_dir=None):
    """
    Records attention graph video over multiple frames

    Input: weights, number of steps, predator image path, prey image path, save directory
    Output: video
    """

    # create save directory with timestamp
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")
    folder_name = f"Experiment Garcia - {timestamp}"
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # iterate over frames to create attention graphs
    for frame_idx in range(num_steps):
        # get weights and metrics for the specified frame
        metrics_list, weights_list = metrics_weights
        metrics = metrics_list[frame_idx]
        weights = weights_list[frame_idx]

        # extract positions and directions
        xs = np.array(metrics["xs"])
        ys = np.array(metrics["ys"])
        directions = np.array(metrics["theta"])

        n_agents = len(xs)

        fig, ax = plt.subplots(figsize=(7, 7))

        # draw edges
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                ax.plot([xs[i], xs[j]],
                        [ys[i], ys[j]],
                        color="gray", linewidth=1, alpha=0.2)

        # change nodes to images
        pred_image = Image.open(pred_img_path).convert("RGBA")
        prey_image = Image.open(prey_img_path).convert("RGBA")

        # compute incoming weights for transparency
        alphas = compute_incoming_weights(weights)

        for i in range(n_agents):
            # compute rotation angle in degrees
            angle_deg = np.degrees(directions[i])

            # select base image
            base_img = pred_image if i == 0 else prey_image

            # get alpha value
            alpha = alphas["predator_0"] if i == 0 else alphas[f"prey_{i}"]

            # rotate image to match direction
            rotated_img = base_img.rotate(angle_deg+180, resample=Image.BICUBIC, expand=True)
            rotated_img = np.asarray(rotated_img).copy()

            if i > 0 and alpha >= 0.999:
                # mark leader prey
                rotated_img = rotated_img.astype(float)
                rotated_img[..., 0] *= 1.8
                rotated_img[..., 1] *= 0.6
                rotated_img[..., 2] *= 0.6
                rotated_img = np.clip(rotated_img, 0, 255).astype(np.uint8)

            # apply transparency
            rotated_img[:, :, 3] = (rotated_img[:, :, 3].astype(float) * alpha).astype(np.uint8)

            # add image to plot
            imgbox = OffsetImage(rotated_img, zoom=0.45)
            ann_box = AnnotationBbox(imgbox, (xs[i], ys[i]), frameon=False, xycoords='data', zorder=2)
            ax.add_artist(ann_box)

        ax.set_title("Attention Graph")
        ax.set_facecolor("#ACCEE7")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()

        # save frame
        output_path = os.path.join(folder_path, f"attention_graph_{frame_idx:03d}.png")
        fig.savefig(output_path, dpi="figure")
        plt.close(fig)

        # progress update
        if frame_idx % 25 == 0:
            print(f"Saved frame {frame_idx}/{num_steps}")

    print("Video rendered!")
    make_video(folder_path) # create video from frames