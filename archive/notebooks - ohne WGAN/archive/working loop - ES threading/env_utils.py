import torch
import numpy as np
import pandas as pd
import math
import torch

def get_env_data(global_state, frame):
    rows = []
    data_dict = global_state.item()

    for row in data_dict.keys():
        name = row
        agent = data_dict[row]
        id = agent[0]
        cx, cy = agent[1], agent[2]
        vx, vy = agent[3], agent[4]
        angle = math.atan2(vy, vx)

        rows.append({
            'frame': frame,
            'agent_id': name,
            'x': cx,
            'y': cy,
            'vx': vx,
            'vy': vy,
            'angle': angle,
        })

    return rows


def get_env_state_actions(full_tracks_df):
    rows = []
    frames = sorted(full_tracks_df['frame'].unique())

    frame = frames[0]
    df = full_tracks_df
    for _, agent in df.iterrows():
        agent_id    = agent['agent_id']
        agent_x     = float(agent['x'])
        agent_y     = float(agent['y'])
        agent_theta = float(agent['angle'])
        cos_t, sin_t = math.cos(agent_theta), math.sin(agent_theta)

        for _, neigh in df[df['agent_id'] != agent_id].iterrows():
            neigh_id    = neigh['agent_id']
            dx = float(neigh['x']) - agent_x
            dy = float(neigh['y']) - agent_y
            rel_v_x = cos_t * float(neigh['vx']) + sin_t * float(neigh['vy'])
            rel_v_y = -sin_t * float(neigh['vx']) + cos_t * float(neigh['vy'])
            rows.append({
                'frame': frame,
                'agent_id': agent_id,
                'neighagent_id': neigh_id,
                'dx': dx,
                'dy': dy,
                'rel_v_x': rel_v_x,
                'rel_v_y': rel_v_y,
                # no theta for single frame
            })

    return pd.DataFrame(rows,
        columns=['frame','agent_id','neighagent_id','dx','dy','rel_v_x','rel_v_y','theta']
    )


def env_pred_to_tensor(df, type):
    feature_map = {'state':  ['dx', 'dy', 'rel_v_x', 'rel_v_y'], 'action': ['theta']}

    cols = feature_map[type]
    n_features = len(cols)

    frames    = np.sort(df['frame'].unique())
    agents    = np.sort(df['agent_id'].unique())
    neighbors = np.sort(df['neighagent_id'].unique())

    frame_idx    = {f: i for i, f in enumerate(frames)}
    agent_idx    = {a: j for j, a in enumerate(agents)}
    neighbor_idx = {n: k for k, n in enumerate(neighbors)}

    array = np.full((len(frames), len(agents), len(neighbors), n_features), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        i = frame_idx[row['frame']]
        j = agent_idx[row['agent_id']]
        k = neighbor_idx[row['neighagent_id']]

        for fi, col in enumerate(cols):
            array[i, j, k, fi] = row[col]

    tensor = torch.from_numpy(array)

    return tensor


def env_prey_to_tensor(df, type):
    feature_map = {'state':  ['dx', 'dy', 'rel_v_x', 'rel_v_y'], 'action': ['theta']}

    cols = feature_map[type]
    n_features = len(cols)

    frames_all = np.sort(df['frame'].unique())
    agents_all = np.sort(df['agent_id'].unique())

    n_frames   = len(frames_all)
    n_agents   = len(agents_all)
    n_neighbors = n_agents - 1

    frame_idx = {f: i for i, f in enumerate(frames_all)}
    agent_idx = {a: j for j, a in enumerate(agents_all)}

    neighbor_idx = {}
    for a in agents_all:
        neigh_list = [x for x in agents_all if x != a]
        neighbor_idx[a] = {n: k for k, n in enumerate(neigh_list)}

    array = np.full((n_frames, n_agents, n_neighbors, n_features), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        f_id = row['frame']
        a_id = row['agent_id']
        n_id = row['neighagent_id']

        if n_id == a_id:
            continue

        i = frame_idx[f_id]
        j = agent_idx[a_id]

        if n_id not in neighbor_idx[a_id]:
            continue
        k = neighbor_idx[a_id][n_id]

        for fi, col in enumerate(cols):
            array[i, j, k, fi] = row[col]

    tensor = torch.from_numpy(array)  # dtype=float32

    return tensor


def get_state_action_tensors(records):
    if len(records) > 0 and isinstance(records[0], dict):
        flat_records = records
    else:
        flat_records = [item for sublist in records for item in sublist]

    data = pd.DataFrame(flat_records)

    env_state_actions = get_env_state_actions(data)

    df_pred = env_state_actions[env_state_actions["agent_id"] == "predator_0"].copy()
    env_pred_state  = env_pred_to_tensor(df_pred, "state")
    env_pred_action = env_pred_to_tensor(df_pred, "action")
    env_pred_tensor = torch.cat([env_pred_state, env_pred_action], dim=-1)

    df_prey = env_state_actions[env_state_actions["agent_id"] != "predator_0"].copy()
    env_prey_state  = env_prey_to_tensor(df_prey, "state")
    env_prey_action = env_prey_to_tensor(df_prey, "action")
    env_prey_tensor = torch.cat([env_prey_state, env_prey_action], dim=-1)

    return env_pred_tensor, env_prey_tensor


def continuous_to_discrete(actions, action_count):
    low, high = -math.pi, math.pi
    scaled = (actions - low) * (action_count - 1) / (high - low)
    idxs   = scaled.round().long().clamp(0, action_count - 1)
    return idxs


def get_rollouts(env, pred_policy, prey_policy, num_frames=9, render=False, randomized_seed=None):
    if randomized_seed is not None:
        env.reset(seed=randomized_seed)
    else:
        env.reset()

    last_agent = env.agents[-1]
    pred_tensors, prey_tensors = [], []

    for frame_idx in range(num_frames):
        global_state = env.state()
        env_data = get_env_data(global_state, frame_idx)
        env_pred_tensor, env_prey_tensor = get_state_action_tensors(env_data)

        pred_states = env_pred_tensor[..., :4]
        prey_states = env_prey_tensor[..., :4]

        action_count = env.unwrapped.action_count

        # PREDATOR
        con_pred_action = pred_policy.forward_pred(pred_states)
        pred_disc = continuous_to_discrete(con_pred_action, action_count)
        pred_action = pred_disc.item()

        con_pred = con_pred_action.unsqueeze(-1).unsqueeze(-1)
        con_pred = con_pred.expand(-1, -1, pred_states.size(2), -1)
        pred_tensor = torch.cat([pred_states, con_pred], dim=-1)
        pred_tensors.append(pred_tensor)


        # PREY
        con_prey_action = prey_policy.forward_prey(prey_states)
        prey_disc = continuous_to_discrete(con_prey_action, action_count).flatten()

        con_prey = con_prey_action.unsqueeze(-1).unsqueeze(-1)
        con_prey = con_prey.expand(-1, -1, prey_states.size(2), -1)
        prey_tensor = torch.cat([prey_states, con_prey], dim=-1)
        prey_tensors.append(prey_tensor)


        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                env.step(None)
            else:
                if agent.startswith("predator"):
                    env.step(pred_action)
                else:
                    idx = int(agent.split("_")[1])
                    env.step(prey_disc[idx].item())

            if render:
                env.render()

            if agent == last_agent:
                break

    if render:
        try:
            env.close()
        except SystemExit:
            pass

    pred_stacked = torch.stack(pred_tensors, dim=0)
    prey_stacked = torch.stack(prey_tensors, dim=0)

    pred_tensor = pred_stacked.squeeze(1)
    prey_tensor = prey_stacked.squeeze(1) 

    return pred_tensor, prey_tensor