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
    def __init__(self, agent_id, speed, area_width, area_height, area_depth, dimension='2d'):
        self.id = agent_id
        self.pos = np.array([
            np.random.uniform(0, area_width),
            np.random.uniform(0, area_height),
            np.random.uniform(0, area_depth)
        ])
        self.vel = np.random.uniform(-1, 1, 3)
        self.is_alive = 1

        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0

        self.vel = self.vel / norm(self.vel) * speed

    def update_position(self, delta_t):
        if self.is_alive:
            self.pos = self.pos + self.vel * delta_t


def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)],
        [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc]
    ])


def enforce_walls(agent, area_width, area_height):
    # X axis
    if agent.pos[0] < 0:
        agent.pos[0] = 0
        agent.vel[0] *= -1
    elif agent.pos[0] > area_width:
        agent.pos[0] = area_width
        agent.vel[0] *= -1

    # Y axis
    if agent.pos[1] < 0:
        agent.pos[1] = 0
        agent.vel[1] *= -1
    elif agent.pos[1] > area_height:
        agent.pos[1] = area_height
        agent.vel[1] *= -1


def run_couzin_simulation(visualization='on', dimension='2d', n=32, max_steps=1000, dt=0.5, free_offset=10, number_of_sharks=1, alpha=0.1, 
                          r_r=2, r_o=10, r_a=40, r_thr=30, r_lethal=1, field_of_view=3*np.pi/2, field_of_view_shark=2*np.pi, theta_dot_max=0.5, 
                          theta_dot_max_shark=0.3, constant_speed=15, shark_speed=15, area_width=2160, area_height=2160, area_depth=50):
    start = time.time()

    swarm = [Agent(i, constant_speed, area_width, area_height, area_depth, dimension=dimension) for i in range(n)]
    sharks = [Agent(i, shark_speed, area_width, area_height, area_depth, dimension=dimension) for i in range(number_of_sharks)]

    swarm_pos = np.zeros((n, 3))
    swarm_vel = np.zeros((n, 3))
    swarm_color = np.zeros(n)
    sharks_pos = np.zeros((number_of_sharks, 3))
    sharks_vel = np.zeros((number_of_sharks, 3))

    t = 0

    # Figure
    if visualization == 'on':
        fig, ax = plt.subplots()
        if dimension == '3d':
            ax = fig.add_subplot(111, projection='3d')


    # --- Simulation loop ---

    prey_log = np.zeros((max_steps, n, 6))
    predator_log = np.zeros((max_steps, number_of_sharks, 6))

    while t < max_steps:

        for i, agent in enumerate(swarm):
            swarm_pos[i, :] = agent.pos
            swarm_vel[i, :] = agent.vel
            swarm_color[i] = 2 - agent.is_alive

            # direction = normalisiertes velocity (nur x,y)
            vel_norm = norm(agent.vel[0:2])
            if vel_norm > 0:
                dir_xy = agent.vel[0:2] / vel_norm
            else:
                dir_xy = np.zeros(2)

            prey_log[t, i, 0:2] = agent.pos[0:2]   # x, y
            prey_log[t, i, 2:4] = agent.vel[0:2]   # vx, vy
            prey_log[t, i, 4:6] = dir_xy           # direction_x, direction_y


        for i, shark in enumerate(sharks):
            sharks_pos[i, :] = shark.pos
            sharks_vel[i, :] = shark.vel / norm(shark.vel) / 80 * area_width

            vel_norm_s = norm(shark.vel[0:2])
            if vel_norm_s > 0:
                dir_xy_s = shark.vel[0:2] / vel_norm_s
            else:
                dir_xy_s = np.zeros(2)

            predator_log[t, i, 0:2] = shark.pos[0:2]    # x, y
            predator_log[t, i, 2:4] = shark.vel[0:2]    # vx, vy
            predator_log[t, i, 4:6] = dir_xy_s          # direction_x, direction_y

        # --- Visualization ---
        if visualization == 'on':
            ax.clear()

            if dimension == '2d':
                pylab.quiver(
                    swarm_pos[:, 0], swarm_pos[:, 1],
                    swarm_vel[:, 0], swarm_vel[:, 1],
                    swarm_color
                )

                ax.plot(sharks_pos[:, 0], sharks_pos[:, 1], 'o', color='#FF0000')
                ax.set_aspect('equal', 'box')
                ax.set_xlim(0, area_width)
                ax.set_ylim(0, area_height)

            else:
                q = ax.quiver(
                    swarm_pos[:, 0], swarm_pos[:, 1], swarm_pos[:, 2],
                    swarm_vel[:, 0], swarm_vel[:, 1], swarm_vel[:, 2]
                )
                q.set_array(swarm_color)
                ax.plot(
                    sharks_pos[:, 0], sharks_pos[:, 1], sharks_pos[:, 2],
                    'o', color='#FF0000'
                )
                ax.set_xlim(-free_offset, area_width + free_offset)
                ax.set_ylim(-free_offset, area_height + free_offset)
                ax.set_zlim(-free_offset, area_depth + free_offset)

            plt.pause(0.00000001)

        # --- Prey update ---
        for agent in swarm:
            d = np.zeros(3)
            d_social = np.zeros(3)
            d_r = np.zeros(3)
            d_o = np.zeros(3)
            d_a = np.zeros(3)
            d_thr = np.zeros(3)

            if agent.is_alive:
                # social interactions (prey–prey)
                for neighbor in swarm:
                    if agent.id == neighbor.id or not neighbor.is_alive:
                        continue

                    r_vec = neighbor.pos - agent.pos
                    dist = norm(r_vec)
                    if dist >= r_a:
                        continue

                    r_normalized = r_vec / dist
                    agent_vel_normalized = agent.vel / norm(agent.vel)

                    # field of view
                    if acos(np.dot(r_normalized, agent_vel_normalized)) >= field_of_view / 2:
                        continue

                    if dist < r_r:
                        d_r -= r_normalized
                    elif dist < r_o:
                        d_o += neighbor.vel / norm(neighbor.vel)
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

                # predator avoidance
                for each_shark in sharks:
                    #diff = agent.pos - each_shark.pos # ÄNDERN WENN BEUTE SHARK JAGEN SOLL
                    diff = each_shark.pos - agent.pos
                    dist = norm(diff)
                    if dist <= r_thr:
                        d_thr -= diff / (dist ** 2)
                    #if dist <= r_lethal:
                    #    agent.is_alive = 0
                    #    number_of_alives -= 1

                # combine directions
                if norm(d_social) != 0:
                    d = alpha * d_social / norm(d_social)
                if norm(d_thr) != 0:
                    d += (1 - alpha) / 2 * d_thr / norm(d_thr)

                # steering
                if norm(d) != 0:
                    z = np.cross(d / norm(d), agent.vel / norm(agent.vel))
                    angle_between = asin(norm(z))
                    if angle_between >= theta_dot_max * dt:
                        rot = rotation_matrix_about(z, theta_dot_max * dt)
                        agent.vel = np.asarray(np.asmatrix(agent.vel) * rot)[0]
                    elif abs(angle_between) - pi > 0:
                        agent.vel = d / norm(d) * constant_speed

        # --- Predator update ---
        for each_shark in sharks:
            d = np.zeros(3)
            closest_dist = np.inf

            for prey in swarm:
                if not prey.is_alive:
                    continue

                r_vec = prey.pos - each_shark.pos
                dist = norm(r_vec)
                if dist < closest_dist:
                    direction = r_vec / dist
                    # field of view shark
                    if acos(np.dot(direction, each_shark.vel / norm(each_shark.vel))) < field_of_view_shark / 2:
                        closest_dist = dist
                        d = direction

            if norm(d) != 0:
                z = np.cross(d / norm(d), each_shark.vel / norm(each_shark.vel))
                angle_between = asin(norm(z))
                if angle_between >= theta_dot_max_shark * dt:
                    rot = rotation_matrix_about(z, theta_dot_max_shark * dt)
                    each_shark.vel = np.asarray(np.asmatrix(each_shark.vel) * rot)[0]
                elif abs(angle_between) - pi > 0:
                    each_shark.vel = d / norm(d) * shark_speed

        # --- Walls & movement ---
        for agent in swarm:
            enforce_walls(agent, area_width, area_height)
            agent.update_position(dt)

        for shark in sharks:
            enforce_walls(shark, area_width, area_height)
            shark.update_position(dt)

        t += 1

    final_pred_tensor = torch.zeros((max_steps, number_of_sharks, n, 5), dtype=torch.float32)
    final_prey_tensor = torch.zeros((max_steps, n, n, 5), dtype=torch.float32)
    metrics = []

    for step in range(max_steps):
        prey_log_step = prey_log[step]
        shark_log_step = predator_log[step]
        pred_tensor_step, prey_tensor_step, metric_step = get_features_from_logs(prey_log_step, shark_log_step, area_width, area_height, constant_speed, shark_speed)

        final_pred_tensor[step] = pred_tensor_step
        final_prey_tensor[step] = prey_tensor_step
        metrics.append(metric_step)

    return final_pred_tensor, final_prey_tensor, metrics


def get_features_from_logs(prey_log_step, shark_log_step, area_width, area_height, constant_speed, shark_speed):

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
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    xs_scaled = np.clip(xs, 0, area_width) / float(area_width)
    ys_scaled = np.clip(ys, 0, area_height) / float(area_height)

    dx = xs_scaled[None, :] - xs_scaled[:, None]
    dy = ys_scaled[None, :] - ys_scaled[:, None]

    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    max_speed = float(max(constant_speed, shark_speed))
    rel_vx = np.clip(rel_vx, -max_speed, max_speed) / max_speed
    rel_vy = np.clip(rel_vy, -max_speed, max_speed) / max_speed

    theta_mat = np.tile(theta_norm[:, None], (1, N))
    features = np.stack([dx, dy, rel_vx, rel_vy, theta_mat], axis=-1)

    mask = ~np.eye(N, dtype=bool)
    neigh = features[mask].reshape(N, N - 1, 5)

    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:])

    polarization = compute_polarization(vxs, vys)
    angular_momentum_val = compute_angular_momentum(xs_scaled, ys_scaled, vxs, vys)
    sparsity = degree_of_sparsity(xs_scaled, ys_scaled)
    dist_pred = distance_to_predator(xs_scaled, ys_scaled)
    escape_align = escape_alignment(xs_scaled, ys_scaled, vxs, vys)

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

