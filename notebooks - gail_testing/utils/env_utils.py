import os
import math
import torch
import numpy as np
from marl_aquarium.aquarium_v0 import parallel_env
from concurrent.futures import ThreadPoolExecutor, as_completed

# [Agent, scaled_position_x, scaled_position_y, scaled_direction (-180:180), scaled_speed]

def get_features(global_state, max_speed=15.0):
    sorted_gs = dict(sorted(global_state.items()))
    items = list(sorted_gs.items())
    agents, raw = zip(*items)

    n = len(raw)

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

    rel_vx = np.clip(rel_vx, -max_speed, max_speed) / max_speed # range [-1,1]
    rel_vy = np.clip(rel_vy, -max_speed, max_speed) / max_speed # range [-1,1]
    
    n = xs.shape[0]
    thetas_mat = np.tile(thetas_norm[:, None], (1, n))
    features = np.stack([dx, dy, rel_vx, rel_vy, thetas_mat], axis=-1)

    mask = ~np.eye(n, dtype=bool) # shape (N, N)
    neigh = features[mask].reshape(n, n-1, 5)

    prey_tensor = torch.from_numpy(neigh) 
    
    return prey_tensor



def continuous_to_discrete(actions, action_count=360):
    low, high = -math.pi, math.pi
    scaled = (actions - low) * (action_count - 1) / (high - low)
    discrete_action = scaled.round().long().clamp(0, action_count - 1)
    return discrete_action.flatten().tolist()


def parallel_get_rollouts(env, prey_policy=None, clip_length=30):
    
    prey_tensors = []
    
    for frame_idx in range(clip_length):

        global_state = env.state().item()
        prey_tensor = get_features(global_state)
        prey_tensors.append(prey_tensor)

        prey_states = prey_tensor[..., :4]
        prey_actions = prey_policy.forward(prey_states)
        dis_prey = continuous_to_discrete(prey_actions, 360)

        action_dict = {}
        for agent in env.agents:
            idx = int(agent.split("_")[1])
            action_dict[agent] = dis_prey[idx]

        env.step(action_dict)

    prey_tensor = torch.stack(prey_tensors, dim=0).squeeze(1)

    return prey_tensor.float()


# One env for each thread
def make_env(pred_count=0, use_walls=True, start_frame_pool=None):
    env = parallel_env(predator_count=pred_count, use_walls=use_walls)
    positions = start_frame_pool.sample(n=1)
    obs, infos = env.reset(options=positions)
    return env


def generate_trajectories(buffer, start_frame_pool, prey_policy, clip_length=30, num_generative_episodes=1, use_walls=True):

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(parallel_get_rollouts, make_env(use_walls=use_walls, start_frame_pool=start_frame_pool), 
                                   prey_policy, clip_length) for i in range(num_generative_episodes)]

    successful = 0
    for future in as_completed(futures):
        try:
            prey_tensor = future.result()
            buffer.add_generative(prey_tensor)
            successful += 1
        except KeyError:
            continue
    missing = num_generative_episodes - successful
    if missing > 0:
        generate_trajectories(buffer=buffer, start_frame_pool=start_frame_pool,
                              prey_policy=prey_policy, clip_length=clip_length, 
                              num_generative_episodes=missing, use_walls=use_walls)