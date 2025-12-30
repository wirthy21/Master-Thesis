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
from utils.eval_utils import compute_polarization, compute_angular_momentum, degree_of_sparsity, distance_to_predator, escape_alignment

# https://github.com/hossein-haeri/couzin_swarm_model/blob/master/swarm_pray_predator.py

class Agent:
    def __init__(self, agent_id, speed, area_width, area_height, dimension='2d'):
        self.id = agent_id
        self.pos = np.array([
            np.random.uniform(0, area_width),
            np.random.uniform(0, area_height)
        ])

        self.vel = np.random.uniform(-1, 1, 2)
        self.is_alive = 1

        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
        if self.is_alive:
            self.pos = self.pos + self.vel * delta_t


def rotation_matrix_about(z, theta):
    z = float(z)
    sgn = np.sign(z)
    if sgn == 0.0 or theta == 0.0:
        return np.eye(2, dtype=np.float64)

    ang = sgn * theta
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def enforce_walls(agent, area_width, area_height):
    if agent.pos[0] < 0:
        agent.pos[0] = 0
        agent.vel[0] *= -1
    elif agent.pos[0] > area_width:
        agent.pos[0] = area_width
        agent.vel[0] *= -1

    if agent.pos[1] < 0:
        agent.pos[1] = 0
        agent.vel[1] *= -1
    elif agent.pos[1] > area_height:
        agent.pos[1] = area_height
        agent.vel[1] *= -1


def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def run_couzin_simulation(
    visualization='on', n=32, max_steps=1000, dt=0.5, free_offset=10, number_of_sharks=1, alpha=0.1,
    r_r=2, r_o=10, r_a=40, r_thr=30, r_lethal=1, field_of_view=3*np.pi/2, field_of_view_shark=2*np.pi,
    theta_dot_max=np.pi/0.5, theta_dot_max_shark=np.pi/0.5, constant_speed=5, shark_speed=5,
    area_width=50, area_height=50):

    swarm = [Agent(i, constant_speed, area_width, area_height) for i in range(n)]
    sharks = [Agent(i, shark_speed, area_width, area_height) for i in range(number_of_sharks)]

    swarm_pos = np.zeros((n, 2))
    swarm_vel = np.zeros((n, 2))

    sharks_pos = np.zeros((number_of_sharks, 2))
    sharks_vel = np.zeros((number_of_sharks, 2))

    t = 0

    # Figure
    if visualization == 'on':
        fig, ax = plt.subplots()

    # --- Simulation loop ---
    prey_log = np.zeros((max_steps, n, 6))
    predator_log = np.zeros((max_steps, number_of_sharks, 6))

    pred_tensor_list = []
    prey_tensor_list = []
    metrics_list = []

    max_turn_prey  = float(theta_dot_max * dt) + 1e-12
    max_turn_shark = float(theta_dot_max_shark * dt) + 1e-12

    while t < max_steps:

        # --- Log current state (s_t) ---
        for i, agent in enumerate(swarm):
            swarm_pos[i, :] = agent.pos
            swarm_vel[i, :] = agent.vel

            vel_norm = norm(agent.vel[0:2])
            if vel_norm > 1e-12:
                dir_xy = agent.vel[0:2] / vel_norm
            else:
                dir_xy = np.zeros(2)

            prey_log[t, i, 0:2] = agent.pos[0:2]   # x, y
            prey_log[t, i, 2:4] = agent.vel[0:2]   # vx, vy
            prey_log[t, i, 4:6] = dir_xy           # direction_x, direction_y

        if number_of_sharks > 0:
            for i, shark in enumerate(sharks):
                sharks_pos[i, :] = shark.pos
                sharks_vel[i, :] = shark.vel / (norm(shark.vel) + 1e-12) / 80 * area_width

                vel_norm_s = norm(shark.vel[0:2])
                if vel_norm_s > 1e-12:
                    dir_xy_s = shark.vel[0:2] / vel_norm_s
                else:
                    dir_xy_s = np.zeros(2)

                predator_log[t, i, 0:2] = shark.pos[0:2]    # x, y
                predator_log[t, i, 2:4] = shark.vel[0:2]    # vx, vy
                predator_log[t, i, 4:6] = dir_xy_s          # direction_x, direction_y

        prey_theta_prev = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        if number_of_sharks > 0:
            shark_theta_prev = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
        else:
            shark_theta_prev = None

        # --- Visualization (optional, zeigt s_t) ---
        if visualization == 'on':
            ax.clear()

            pylab.quiver(
                swarm_pos[:, 0], swarm_pos[:, 1],
                swarm_vel[:, 0], swarm_vel[:, 1],
                scale=120,
                width=0.01,
                headwidth=3,
                headlength=3,
                headaxislength=3,
            )

            if number_of_sharks > 0:
                pylab.quiver(
                    sharks_pos[:, 0], sharks_pos[:, 1],
                    sharks_vel[:, 0], sharks_vel[:, 1],
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

        # --- Prey update ---
        for agent in swarm:
            d = np.zeros(2)
            d_social = np.zeros(2)
            d_r = np.zeros(2)
            d_o = np.zeros(2)
            d_a = np.zeros(2)
            d_thr = np.zeros(2)

            if agent.is_alive:
                # social interactions (prey–prey)
                for neighbor in swarm:
                    if agent.id == neighbor.id or not neighbor.is_alive:
                        continue

                    r_vec = neighbor.pos - agent.pos
                    dist = norm(r_vec)
                    if dist >= r_a:
                        continue

                    r_normalized = r_vec / (dist + 1e-12)
                    agent_vel_normalized = agent.vel / (norm(agent.vel) + 1e-12)

                    # field of view
                    dot_val = float(np.dot(r_normalized, agent_vel_normalized)) 
                    dot_val = np.clip(dot_val, -1.0, 1.0) 
                    if acos(dot_val) >= field_of_view / 2:
                        continue

                    if dist < r_r:
                        d_r -= r_normalized
                    elif dist < r_o:
                        d_o += neighbor.vel / (norm(neighbor.vel) + 1e-12)
                    elif dist < r_a:
                        d_a += r_normalized

                if norm(d_r) != 0:
                    d_social = d_r
                elif norm(d_a) != 0 and norm(d_o) != 0:
                    d_social = (d_o + d_a) / 2
                elif norm(d_o) != 0:
                    d_social = d_o
                elif norm(d_a) != 0:
                    d_social = d_a

                if number_of_sharks > 0:
                    for each_shark in sharks:
                        diff = each_shark.pos - agent.pos
                        dist = norm(diff)
                        if dist <= r_thr:
                            d_thr -= diff / (dist ** 2 + 1e-12)

                # combine directions
                if norm(d_social) != 0:
                    d = alpha * d_social / (norm(d_social) + 1e-12)
                if norm(d_thr) != 0:
                    d += (1 - alpha) / 2 * d_thr / (norm(d_thr) + 1e-12)

                # steering
                if norm(d) != 0:
                    z = np.cross(d / (norm(d) + 1e-12), agent.vel / (norm(agent.vel) + 1e-12))
                    angle_between = asin(min(1.0, abs(z)))

                    if angle_between >= theta_dot_max * dt:
                        rot = rotation_matrix_about(z, theta_dot_max * dt)
                        agent.vel = np.asarray(np.asmatrix(agent.vel) * rot)[0]
                    else:
                        agent.vel = d / (norm(d) + 1e-12) * constant_speed

        # --- Predator update ---
        if number_of_sharks > 0:
            for each_shark in sharks:
                d = np.zeros(2)
                closest_dist = np.inf

                for prey in swarm:
                    if not prey.is_alive:
                        continue

                    r_vec = prey.pos - each_shark.pos
                    dist = norm(r_vec)
                    if dist < closest_dist:
                        direction = r_vec / (dist + 1e-12)
                        # field of view shark

                        shark_dir = each_shark.vel / (norm(each_shark.vel) + 1e-12)
                        dot_val = float(np.dot(direction, shark_dir))
                        dot_val = np.clip(dot_val, -1.0, 1.0)
                        if acos(dot_val) < field_of_view_shark / 2:
                            closest_dist = dist
                            d = direction

                if norm(d) != 0:
                    z = np.cross(d / (norm(d) + 1e-12), each_shark.vel / (norm(each_shark.vel) + 1e-12))
                    angle_between = asin(min(1.0, abs(z)))

                    if angle_between >= theta_dot_max_shark * dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark * dt)
                        each_shark.vel = np.asarray(np.asmatrix(each_shark.vel) * rot)[0]
                    else:
                        each_shark.vel = d / (norm(d) + 1e-12) * shark_speed


        prey_theta_new = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        dtheta_prey = wrap_to_pi(prey_theta_new - prey_theta_prev)

        prey_actions = np.clip(dtheta_prey / (2.0 * max_turn_prey) + 0.5, 0.0, 1.0).astype(np.float32)

        if number_of_sharks > 0:
            shark_theta_new = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
            dtheta_shark = wrap_to_pi(shark_theta_new - shark_theta_prev)

            shark_actions = np.clip(dtheta_shark / (2.0 * max_turn_shark) + 0.5, 0.0, 1.0).astype(np.float32)
        else:
            shark_actions = None

        pred_tensor, prey_tensor, metrics = get_state_tensors(
            prey_log[t], predator_log[t],
            prey_actions=prey_actions, shark_actions=shark_actions, 
            area_width=area_width, area_height=area_height,
            constant_speed=constant_speed, shark_speed=shark_speed,
            number_of_sharks=number_of_sharks)

        pred_tensor_list.append(pred_tensor)
        prey_tensor_list.append(prey_tensor)
        metrics_list.append(metrics)

        # --- Walls & movement ---
        for agent in swarm:
            enforce_walls(agent, area_width, area_height)
            agent.update_position(dt)

        if number_of_sharks > 0:
            for shark in sharks:
                enforce_walls(shark, area_width, area_height)
                shark.update_position(dt)

        t += 1

    if number_of_sharks > 0:
        final_pred_tensor = torch.stack(pred_tensor_list, dim=0)
        stacked_prey_tensor = torch.stack(prey_tensor_list, dim=0)

        frames, agents, neigh, feat = stacked_prey_tensor.shape
        flag = torch.zeros(
            (frames, agents, neigh, 1),
            dtype=stacked_prey_tensor.dtype,
            device=stacked_prey_tensor.device)
        flag[:, :, :number_of_sharks, 0] = 1
        final_prey_tensor = torch.cat([flag, stacked_prey_tensor], dim=-1)

    else:
        final_pred_tensor = None
        final_prey_tensor = torch.stack(prey_tensor_list, dim=0)

    if number_of_sharks > 0:
        combined = np.concatenate([predator_log[:t], prey_log[:t]], axis=1).astype(np.float32)
        xs = combined[..., 0].astype(np.float32)
        ys = combined[..., 1].astype(np.float32)
    else:
        xs = prey_log[..., 0].astype(np.float32)
        ys = prey_log[..., 1].astype(np.float32)

    init_pool = np.stack([xs, ys], axis=-1).astype(np.float32)
    init_pool = init_pool[50:] # cut 50 off due to random init

    return final_pred_tensor, final_prey_tensor, metrics_list, init_pool



def get_state_tensors(prey_log_step, shark_log_step, 
                      prey_actions=None, shark_actions=None, 
                      area_width=50, area_height=50, 
                      constant_speed=15, shark_speed=15, 
                      number_of_sharks=1):

    combined = np.vstack([shark_log_step, prey_log_step])
    N = combined.shape[0]

    xs = combined[:, 0].astype(np.float32)
    ys = combined[:, 1].astype(np.float32)
    vxs = combined[:, 2].astype(np.float32)
    vys = combined[:, 3].astype(np.float32)
    dir_x = combined[:, 4].astype(np.float32)
    dir_y = combined[:, 5].astype(np.float32)

    thetas = np.arctan2(dir_y, dir_x).astype(np.float32)
    theta_norm = thetas / np.pi
    cos_t = np.cos(thetas).astype(np.float32)
    sin_t = np.sin(thetas).astype(np.float32)

    xs_scaled = np.clip(xs, 0, area_width)  / float(area_width)
    ys_scaled = np.clip(ys, 0, area_height) / float(area_height)

    dx = xs_scaled[None, :] - xs_scaled[:, None]
    dy = ys_scaled[None, :] - ys_scaled[:, None]

    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    speed = max(constant_speed, shark_speed)
    rel_vx = np.clip(rel_vx, -speed, speed) / speed
    rel_vy = np.clip(rel_vy, -speed, speed) / speed

    if shark_actions is None:
        shark_actions = np.full((number_of_sharks,), 0.5, dtype=np.float32)
    if prey_actions is None:
        prey_actions = np.full((N - number_of_sharks,), 0.5, dtype=np.float32)

    actions = np.concatenate([shark_actions, prey_actions]).astype(np.float32)  # [N]
    action_mat = np.tile(actions[:, None], (1, N))     

    features = np.stack([dx, dy, rel_vx, rel_vy, action_mat], axis=-1).astype(np.float32)

    mask = ~np.eye(N, dtype=bool)
    neigh = features[mask].reshape(N, N - 1, 5)

    pred_tensor = torch.from_numpy(neigh[:number_of_sharks]).to(dtype=torch.float32)
    prey_tensor = torch.from_numpy(neigh[number_of_sharks:]).to(dtype=torch.float32)

    polarization = compute_polarization(vxs, vys)
    angular_momentum_val = compute_angular_momentum(xs, ys, vxs, vys)
    sparsity = degree_of_sparsity(xs, ys)
    dist_pred = distance_to_predator(xs, ys)
    escape_align = escape_alignment(xs, ys, vxs, vys)

    metrics = {
        "polarization": polarization,
        "angular_momentum": angular_momentum_val,
        "degree_of_sparsity": sparsity,
        "distance_to_predator": dist_pred,
        "escape_alignment": escape_align,
        "xs": xs_scaled,
        "ys": ys_scaled,
        "dx": dx,
        "dy": dy,
        "vxs": vxs,
        "vys": vys,
        "features": features,
    }

    return pred_tensor, prey_tensor, metrics



def run_circular_simulation(
    visualization='on',
    n=32,
    max_steps=100,
    dt=0.5,
    number_of_sharks=0,
    theta_dot_max=np.pi/0.5,
    theta_dot_max_shark=np.pi/0.5,
    constant_speed=5,
    shark_speed=5,
    area_width=50,
    area_height=50,
    orbit_radius=None,          # CHANGED: gewünschter Kreisradius (None => automatisch)
    orbit_radius_shark=None,    # CHANGED: optional eigener Radius für Sharks
    radial_gain=1.0,            # CHANGED: wie stark zur Soll-Bahn gezogen wird
    radial_gain_shark=1.0,      # CHANGED
    clockwise=False,            # CHANGED: Drehrichtung
    device="cpu"
):
    """
    Simple 'circular/orbit' behavior:
    - Agents orbit around the center with constant speed
    - A radial correction term keeps them near a target radius
    - Outputs match run_couzin_simulation: (pred_tensor, prey_tensor, metrics_list)

    NOTE: This simulation is intended to run on CPU. The returned tensors are CPU tensors,
          consistent with run_couzin_simulation in this file.
    """

    swarm = [Agent(i, constant_speed, area_width, area_height) for i in range(n)]
    sharks = [Agent(i, shark_speed, area_width, area_height) for i in range(number_of_sharks)]

    swarm_pos = np.zeros((n, 2), dtype=np.float64)
    swarm_vel = np.zeros((n, 2), dtype=np.float64)

    sharks_pos = np.zeros((number_of_sharks, 2), dtype=np.float64)
    sharks_vel = np.zeros((number_of_sharks, 2), dtype=np.float64)

    # Figure
    if visualization == 'on':
        fig, ax = plt.subplots()

    # Logs (match run_couzin_simulation)
    prey_log = np.zeros((max_steps, n, 6), dtype=np.float32)
    predator_log = np.zeros((max_steps, number_of_sharks, 6), dtype=np.float32)

    pred_tensor_list = []
    prey_tensor_list = []
    metrics_list = []

    # center + default orbit radius
    center = np.array([area_width * 0.5, area_height * 0.5], dtype=np.float64)
    min_dim = float(min(area_width, area_height))

    if orbit_radius is None:
        orbit_radius = 0.35 * min_dim  # CHANGED: default
    orbit_radius = float(orbit_radius)

    if orbit_radius_shark is None:
        orbit_radius_shark = 0.20 * min_dim  # CHANGED: default
    orbit_radius_shark = float(orbit_radius_shark)

    max_turn_prey  = float(theta_dot_max * dt) + 1e-12
    max_turn_shark = float(theta_dot_max_shark * dt) + 1e-12

    t = 0
    while t < max_steps:

        # --- Log current state (s_t) ---
        for i, agent in enumerate(swarm):
            swarm_pos[i, :] = agent.pos
            swarm_vel[i, :] = agent.vel

            vel_norm = norm(agent.vel[0:2])
            if vel_norm > 1e-12:
                dir_xy = agent.vel[0:2] / vel_norm
            else:
                dir_xy = np.zeros(2, dtype=np.float64)

            prey_log[t, i, 0:2] = agent.pos[0:2]
            prey_log[t, i, 2:4] = agent.vel[0:2]
            prey_log[t, i, 4:6] = dir_xy

        if number_of_sharks > 0:
            for i, shark in enumerate(sharks):
                sharks_pos[i, :] = shark.pos
                sharks_vel[i, :] = shark.vel / (norm(shark.vel) + 1e-12) / 80 * area_width

                vel_norm_s = norm(shark.vel[0:2])
                if vel_norm_s > 1e-12:
                    dir_xy_s = shark.vel[0:2] / vel_norm_s
                else:
                    dir_xy_s = np.zeros(2, dtype=np.float64)

                predator_log[t, i, 0:2] = shark.pos[0:2]
                predator_log[t, i, 2:4] = shark.vel[0:2]
                predator_log[t, i, 4:6] = dir_xy_s

        prey_theta_prev = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        if number_of_sharks > 0:
            shark_theta_prev = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
        else:
            shark_theta_prev = None

        # --- Visualization (optional, shows s_t) ---
        if visualization == 'on':
            ax.clear()

            pylab.quiver(
                swarm_pos[:, 0], swarm_pos[:, 1],
                swarm_vel[:, 0], swarm_vel[:, 1],
                scale=120, width=0.01,
                headwidth=3, headlength=3, headaxislength=3
            )

            if number_of_sharks > 0:
                pylab.quiver(
                    sharks_pos[:, 0], sharks_pos[:, 1],
                    sharks_vel[:, 0], sharks_vel[:, 1],
                    color="#FF0000",
                    scale=15, width=0.01,
                    headwidth=3, headlength=3, headaxislength=3
                )

            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, area_width)
            ax.set_ylim(0, area_height)
            plt.pause(0.00000001)

        # --- Prey circular steering update (updates vel only) ---
        for agent in swarm:
            r = agent.pos - center
            dist = float(norm(r) + 1e-12)
            radial_dir = r / dist

            # tangential direction (CCW); flip for clockwise
            tangential_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float64)
            if clockwise:
                tangential_dir *= -1.0

            # radial correction keeps orbit radius
            radial_error = dist - orbit_radius
            desired = tangential_dir - (radial_gain * (radial_error / (orbit_radius + 1e-12))) * radial_dir

            if norm(desired) > 1e-12:
                desired = desired / (norm(desired) + 1e-12)

                # same turn-rate limiting logic as in run_couzin_simulation
                z = np.cross(desired, agent.vel / (norm(agent.vel) + 1e-12))
                angle_between = asin(min(1.0, abs(float(z))))

                if angle_between >= theta_dot_max * dt:
                    rot = rotation_matrix_about(z, theta_dot_max * dt)
                    agent.vel = np.asarray(np.asmatrix(agent.vel) * rot)[0]
                    agent.vel = agent.vel / (norm(agent.vel) + 1e-12) * constant_speed
                else:
                    agent.vel = desired * constant_speed

        # --- Shark circular steering update (optional) ---
        if number_of_sharks > 0:
            for shark in sharks:
                r = shark.pos - center
                dist = float(norm(r) + 1e-12)
                radial_dir = r / dist

                tangential_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float64)
                if clockwise:
                    tangential_dir *= -1.0

                radial_error = dist - orbit_radius_shark
                desired = tangential_dir - (radial_gain_shark * (radial_error / (orbit_radius_shark + 1e-12))) * radial_dir

                if norm(desired) > 1e-12:
                    desired = desired / (norm(desired) + 1e-12)

                    z = np.cross(desired, shark.vel / (norm(shark.vel) + 1e-12))
                    angle_between = asin(min(1.0, abs(float(z))))

                    if angle_between >= theta_dot_max_shark * dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark * dt)
                        shark.vel = np.asarray(np.asmatrix(shark.vel) * rot)[0]
                        shark.vel = shark.vel / (norm(shark.vel) + 1e-12) * shark_speed
                    else:
                        shark.vel = desired * shark_speed

        # --- Actions from heading change (same as Couzin) ---
        prey_theta_new = np.array([np.arctan2(a.vel[1], a.vel[0]) for a in swarm], dtype=np.float32)
        dtheta_prey = wrap_to_pi(prey_theta_new - prey_theta_prev)
        prey_actions = np.clip(dtheta_prey / (2.0 * max_turn_prey) + 0.5, 0.0, 1.0).astype(np.float32)

        if number_of_sharks > 0:
            shark_theta_new = np.array([np.arctan2(s.vel[1], s.vel[0]) for s in sharks], dtype=np.float32)
            dtheta_shark = wrap_to_pi(shark_theta_new - shark_theta_prev)
            shark_actions = np.clip(dtheta_shark / (2.0 * max_turn_shark) + 0.5, 0.0, 1.0).astype(np.float32)
        else:
            shark_actions = None

        # --- Build tensors for this step (state is s_t, action is dtheta from s_t -> s_{t+1}) ---
        pred_tensor, prey_tensor, metrics = get_state_tensors(
            prey_log[t], predator_log[t],
            prey_actions=prey_actions, shark_actions=shark_actions,
            area_width=area_width, area_height=area_height,
            constant_speed=constant_speed, shark_speed=shark_speed,
            number_of_sharks=number_of_sharks
        )

        pred_tensor_list.append(pred_tensor)
        prey_tensor_list.append(prey_tensor)
        metrics_list.append(metrics)

        # --- Walls & movement ---
        for agent in swarm:
            enforce_walls(agent, area_width, area_height)
            agent.update_position(dt)

        if number_of_sharks > 0:
            for shark in sharks:
                enforce_walls(shark, area_width, area_height)
                shark.update_position(dt)

        t += 1

    final_pred_tensor = torch.stack(pred_tensor_list, dim=0) if number_of_sharks > 0 else 0
    final_prey_tensor = torch.stack(prey_tensor_list, dim=0)

    if number_of_sharks > 0:
        combined = np.concatenate([predator_log[:t], prey_log[:t]], axis=1).astype(np.float32)
        xs = combined[..., 0].astype(np.float32)
        ys = combined[..., 1].astype(np.float32)
    else:
        xs = prey_log[..., 0].astype(np.float32)
        ys = prey_log[..., 1].astype(np.float32)

    init_pool = np.stack([xs, ys], axis=-1).astype(np.float32)
    init_pool = init_pool[50:] # cut 50 off due to random init

    return final_pred_tensor, final_prey_tensor, metrics_list, init_pool

