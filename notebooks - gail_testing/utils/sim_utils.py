import os
import time
import pylab
import torch
import pickle
import numpy as np
from math import *
import matplotlib as mpl
from cycler import cycler
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy


class Agent:
    def __init__(self, agent_id, speed, area_width, area_height):
        self.id = agent_id
        self.pos = np.array([np.random.uniform(0, area_width),
                             np.random.uniform(0, area_height)], dtype=np.float64)

        self.theta = np.random.uniform(-np.pi, np.pi)
        self.vel = np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float64) * speed

    def update_position(self, step_size):
        self.pos += self.vel * step_size


def apply_turnrate_on_theta(agent, action, speed, max_turn=np.pi):
    dtheta = (action - 0.5) * 2.0 * max_turn
    agent.theta = (agent.theta + dtheta + np.pi) % (2*np.pi) - np.pi
    agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)]) * speed



def enforce_walls(agent, area_width, area_height):
    bounced = False

    if agent.pos[0] < 0:
        agent.pos[0] = 0
        agent.theta = np.pi - agent.theta
        bounced = True
    elif agent.pos[0] > area_width:
        agent.pos[0] = area_width
        agent.theta = np.pi - agent.theta
        bounced = True

    if agent.pos[1] < 0:
        agent.pos[1] = 0
        agent.theta = -agent.theta
        bounced = True
    elif agent.pos[1] > area_height:
        agent.pos[1] = area_height
        agent.theta = -agent.theta
        bounced = True

    if bounced:
        agent.theta = (agent.theta + np.pi) % (2*np.pi) - np.pi
        speed = float(norm(agent.vel)) 
        agent.vel = np.array([np.cos(agent.theta), np.sin(agent.theta)], dtype=np.float64) * speed


def get_state_tensors(prey_log_step, pred_log_step, n_pred=1, 
                      area_width=50, area_height=50,
                      prey_speed=5, pred_speed=5, mask=None):
    
    combined = np.vstack([pred_log_step, prey_log_step]).astype(np.float32)  # [N,6]
    n_agents = combined.shape[0]

    xs, ys  = combined[:, 0], combined[:, 1]
    vxs, vys = combined[:, 2], combined[:, 3]
    dir_x, dir_y = combined[:, 4], combined[:, 5]

    cos_t = dir_x.astype(np.float32)
    sin_t = dir_y.astype(np.float32)

    xs_scaled = xs / float(area_width)
    ys_scaled = ys / float(area_height)

    dx = xs_scaled[None, :] - xs_scaled[:, None]
    dy = ys_scaled[None, :] - ys_scaled[:, None]

    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    speed = max(prey_speed, pred_speed)
    rel_vx = np.clip(rel_vx, -speed, speed) / speed
    rel_vy = np.clip(rel_vy, -speed, speed) / speed

    features = np.stack([dx, dy, rel_vx, rel_vy], axis=-1).astype(np.float32)
    neigh = features[mask].reshape(n_agents, n_agents-1, 4)

    tensor = torch.from_numpy(neigh)   # already float32

    pred_tensor = tensor[:n_pred]
    prey_tensor = tensor[n_pred:]

    if n_pred > 0:
        agents, neighs, _ = prey_tensor.shape
        flag = torch.zeros((agents, neighs, 1), dtype=prey_tensor.dtype, device=prey_tensor.device)
        flag[:, :n_pred, 0] = 1
        prey_tensor = torch.cat([flag, prey_tensor], dim=-1)

    return pred_tensor, prey_tensor


def apply_init_pool(init_pool, pred, prey, area_width=50, area_height=50):
    steps, agents, coordinates = init_pool.shape
    agents = len(pred) + len(prey)

    positions = init_pool[np.random.randint(steps), :agents]  # [agents, 2]

    # predators
    for i in range(len(pred)):
        x = float(np.clip(positions[i, 0], 0.0, area_width))
        y = float(np.clip(positions[i, 1], 0.0, area_height))
        pred[i].pos = np.array([x, y], dtype=np.float64)

    # prey
    for i in range(len(prey)):
        j = len(pred) + i
        x = float(np.clip(positions[j, 0], 0.0, area_width))
        y = float(np.clip(positions[j, 1], 0.0, area_height))
        prey[i].pos = np.array([x, y], dtype=np.float64)


def run_env_simulation(prey_policy=None, pred_policy=None, 
                       n_prey=32, n_pred=1, step_size=0.5,
                       max_steps=100, seed=None, deterministic=False,
                       prey_speed=15, pred_speed=15, 
                       area_width=50, area_height=50, 
                       visualization='off', init_pool=None):

    if seed is not None:
        np.random.seed(seed) # agent init
        torch.manual_seed(seed) # CPU

    prey_policy = deepcopy(prey_policy).to("cpu")
    pred_policy = deepcopy(pred_policy).to("cpu") if pred_policy is not None else None

    prey = [Agent(i, prey_speed, area_width, area_height) for i in range(n_prey)]
    pred = [Agent(i, pred_speed, area_width, area_height) for i in range(n_pred)]

    if init_pool is not None:
        apply_init_pool(init_pool, pred, prey, area_width=area_width, area_height=area_height)

    n_agents = n_prey + n_pred
    neigh = n_agents - 1
    prey_traj = torch.empty((max_steps, n_prey, neigh, 6), dtype=torch.float32) if n_pred > 0 else torch.empty((max_steps, n_prey, neigh, 5), dtype=torch.float32)
    pred_traj = torch.empty((max_steps, n_pred, neigh, 5), dtype=torch.float32) if n_pred > 0 else None

    if visualization == 'on':
        prey_pos_vis = np.zeros((n_prey, 2), dtype=np.float32)
        prey_vel_vis = np.zeros((n_prey, 2), dtype=np.float32)
        pred_pos_vis = np.zeros((n_pred, 2), dtype=np.float32) if n_pred > 0 else None
        pred_vel_vis = np.zeros((n_pred, 2), dtype=np.float32) if n_pred > 0 else None
        fig, ax = plt.subplots()

    mask = ~np.eye(n_agents, dtype=bool)
    t = 0

    while t < max_steps:
        prey_pos_now = np.asarray([a.pos for a in prey], dtype=np.float32)  # [n_prey,2]
        prey_vel_now = np.asarray([a.vel for a in prey], dtype=np.float32)  # [n_prey,2]
        prey_dir = prey_vel_now / (np.linalg.norm(prey_vel_now, axis=1, keepdims=True) + 1e-12)
        prey_log_t = np.concatenate([prey_pos_now, prey_vel_now, prey_dir], axis=1)  # [n_prey,6]

        if n_pred > 0:
            pred_pos_now = np.asarray([a.pos for a in pred], dtype=np.float32)
            pred_vel_now = np.asarray([a.vel for a in pred], dtype=np.float32)
            pred_dir = pred_vel_now / (np.linalg.norm(pred_vel_now, axis=1, keepdims=True) + 1e-12)
            predator_log_t = np.concatenate([pred_pos_now, pred_vel_now, pred_dir], axis=1)
        else:
            predator_log_t = np.empty((0, 6), dtype=np.float32)

        # --- Visualization ---
        if visualization == 'on':
            prey_pos_vis[:, :] = prey_pos_now
            prey_vel_vis[:, :] = prey_vel_now

            if n_pred > 0:
                pred_pos_vis[:, :] = pred_pos_now
                pred_vel_vis[:, :] = pred_vel_now

            ax.clear()
            pylab.quiver(
                prey_pos_vis[:, 0], prey_pos_vis[:, 1],
                prey_vel_vis[:, 0], prey_vel_vis[:, 1],
                scale=120,
                width=0.01,
                headwidth=3,
                headlength=3,
                headaxislength=3,
            )

            if n_pred > 0:
                pylab.quiver(
                    pred_pos_vis[:, 0], pred_pos_vis[:, 1],
                    pred_vel_vis[:, 0], pred_vel_vis[:, 1],
                    color="#FF0000",
                    scale=15,
                    width=0.01,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                )
            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, area_width)
            ax.set_ylim(0, area_height)

            plt.pause(0.00000001)

        pred_states, prey_states = get_state_tensors(prey_log_t, predator_log_t, 
                                                     n_pred=n_pred, 
                                                     area_width=area_width, area_height=area_height, 
                                                     prey_speed=prey_speed, pred_speed=pred_speed,
                                                     mask=mask)

        if n_pred > 0:
            with torch.inference_mode():
                pred_actions = pred_policy.forward(pred_states, deterministic=deterministic)
                pred_traj[t, :, :, :4] = pred_states
                pred_traj[t, :, :, 4:] = pred_actions.unsqueeze(1).expand(-1, neigh, -1)

            with torch.inference_mode():
                prey_actions = prey_policy.forward(prey_states, deterministic=deterministic)
                prey_traj[t, :, :, :5] = prey_states
                prey_traj[t, :, :, 5:] = prey_actions.unsqueeze(1).expand(-1, neigh, -1)
        else:
            with torch.inference_mode():
                prey_actions = prey_policy.forward(prey_states, deterministic=deterministic)
                prey_traj[t, :, :, :4] = prey_states
                prey_traj[t, :, :, 4:] = prey_actions.unsqueeze(1).expand(-1, neigh, -1)

        prey_actions = prey_actions.squeeze(-1).detach().cpu().numpy()
        for i, agent in enumerate(prey):
            apply_turnrate_on_theta(agent, prey_actions[i], prey_speed)

        if n_pred > 0:
            pred_actions = pred_actions.squeeze(-1).detach().cpu().numpy()
            for i, predator in enumerate(pred):
                apply_turnrate_on_theta(predator, pred_actions[i], pred_speed)

        for agent in prey:
            enforce_walls(agent, area_width, area_height)
            agent.update_position(step_size=step_size)

        if n_pred > 0:
            for predator in pred:
                enforce_walls(predator, area_width, area_height)
                predator.update_position(step_size=step_size)

        t += 1

        prey_tensor = prey_traj[:t]
        pred_tensor = pred_traj[:t] if n_pred > 0 else None

    return pred_tensor, prey_tensor