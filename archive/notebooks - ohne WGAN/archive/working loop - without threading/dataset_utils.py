import os
import cv2
import numpy as np
import pandas as pd
import torch
import math
from collections import defaultdict
from ultralytics import YOLO
from torch.utils.data import Dataset
from deep_sort_realtime.deepsort_tracker import DeepSort


def get_swarm_data(frame, model, tracker, frame_idx):
    records = []

    # YOLO inference
    results = model(frame)[0]
    bboxes, confidences, class_ids = [], [], []

    for box, score, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        label = model.names[int(cls)]
        
        if label not in ("Prey", "Predator Head"):
            continue
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        bboxes.append([x1, y1, w, h])
        confidences.append(float(score))
        class_ids.append(label)

    detections = list(zip(bboxes, confidences, class_ids))
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        tid = t.track_id
        cx, cy = t.mean[0], t.mean[1]
        vx, vy = float(t.mean[4]), float(t.mean[5])
        speed = np.hypot(vx, vy)
        angle = math.atan2(vy, vx)  # in radians
        label = t.det_class
        conf = t.det_conf

        records.append({
            "frame":    int(frame_idx),
            "track_id": int(tid),
            "label":    str(label),
            "conf":     conf,
            "x":        float(cx),
            "y":        float(cy),
            "vx":       float(vx),
            "vy":       float(vy),
            "speed":    float(speed),
            "angle":    float(angle),
        })

    return pd.DataFrame(records)


def get_quality_score(df, n=10):
    track_counts = df.groupby("track_id")["frame"].nunique()
    mean_track_visibility = track_counts.mean()

    num_full_tracks = (track_counts == n).sum()

    mean_confidence = df["conf"].mean()

    print(f"Mean track visibility: {mean_track_visibility}")
    print(f"Number of full tracks: {num_full_tracks}")
    print(f"Mean confidence: {mean_confidence}")

    return mean_track_visibility, num_full_tracks, mean_confidence


def get_full_tracks(df, n):
    df_pred = df[df["label"] == "Predator Head"]
    idx_best_pred = (df_pred.groupby("frame", sort=False)["conf"].idxmax().dropna().astype(int))
    df_pred_best = df.loc[idx_best_pred]
    df_non_pred = df[df["label"] != "Predator Head"]
    df_filtered = pd.concat([df_non_pred, df_pred_best], ignore_index=True)

    counts = df_filtered.groupby("track_id")["frame"].nunique()
    good_ids = counts[counts == n].index
    full_tracks_df = df_filtered[df_filtered["track_id"].isin(good_ids)]

    if full_tracks_df.empty:
        print("No full tracks found.")
        return pd.DataFrame(columns=df.columns)
    elif full_tracks_df[full_tracks_df["label"] == "Predator Head"].empty:
        print("No predator found in full tracks.")
        return pd.DataFrame(columns=df.columns)
    else:
        return full_tracks_df


def get_state_actions(full_tracks_df):
    rows = []
    all_frames = sorted(full_tracks_df['frame'].unique())
    frame_idx = {f: full_tracks_df[full_tracks_df['frame'] == f] for f in all_frames}

    for i, frame in enumerate(all_frames[:-1]):
        next_frame = all_frames[i + 1]
        df_frame = frame_idx[frame]
        df_next = frame_idx[next_frame]

        for _, agent in df_frame.iterrows():
            agent_id = agent['track_id']
            agent_label = agent['label']
            agent_x = float(agent['x'])
            agent_y = float(agent['y'])
            agent_vx = float(agent['vx'])
            agent_vy = float(agent['vy'])
            agent_theta = float(agent['angle'])

            next_row = df_next[df_next['track_id'] == agent_id]
            if next_row.empty:
                continue
            theta_next = float(next_row.iloc[0]['angle'])
            d = theta_next - agent_theta
            delta_theta = (d + np.pi) % (2 * np.pi) - np.pi

            df_neighbors = df_frame[df_frame['track_id'] != agent_id]
            cos_t = math.cos(agent_theta)
            sin_t = math.sin(agent_theta)

            for _, neigh in df_neighbors.iterrows():
                neighagent_label = neigh['label']
                neighagent_id = neigh['track_id']
                neighagent_x = float(neigh['x'])
                neighagent_y = float(neigh['y'])
                neighagent_vx = float(neigh['vx'])
                neighagent_vy = float(neigh['vy'])

                dx = neighagent_x - agent_x
                dy = neighagent_y - agent_y
                rel_v_x = cos_t * neighagent_vx + sin_t * neighagent_vy
                rel_v_y = -sin_t * neighagent_vx + cos_t * neighagent_vy

                rows.append({
                    'frame': frame,
                    'agent_label': agent_label,
                    'agent_id': agent_id,
                    'neighagent_label': neighagent_label,
                    'neighagent_id': neighagent_id,
                    'dx': dx,
                    'dy': dy,
                    'rel_v_x': rel_v_x,
                    'rel_v_y': rel_v_y,
                    'theta': delta_theta
                })

    df_state_actions = pd.DataFrame(rows, columns=['frame', 'agent_label', 'agent_id', 'neighagent_label', 'neighagent_id', 'dx', 'dy', 'rel_v_x', 'rel_v_y', 'theta'])
    return df_state_actions


def pred_to_tensor(df, type):
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


def prey_to_tensor(df, type):
    feature_map = {'state':  ['dx', 'dy', 'rel_v_x', 'rel_v_y'], 'action': ['theta']}

    cols = feature_map[type]
    n_features = len(cols)

    # 2) Einmalige, sortierte Indizes
    frames_all = np.sort(df['frame'].unique())
    agents_all = np.sort(df['agent_id'].unique())

    n_frames   = len(frames_all)
    n_agents   = len(agents_all)
    n_neighbors = n_agents - 1

    # 2a) Maps für Frame- und Agent-Indizes
    frame_idx = {f: i for i, f in enumerate(frames_all)}
    agent_idx = {a: j for j, a in enumerate(agents_all)}

    # 2b) Pro-Agent: Mapping Nachbar-ID -> Position (0..n_neighbors-1)
    neighbor_idx = {}
    for a in agents_all:
        neigh_list = [x for x in agents_all if x != a]
        neighbor_idx[a] = {n: k for k, n in enumerate(neigh_list)}

    # 3) Array initialisieren (mit NaN)
    array = np.full((n_frames, n_agents, n_neighbors, n_features), np.nan, dtype=np.float32)

    # 4) DataFrame-Zeilen durchgehen und Werte eintragen
    for _, row in df.iterrows():
        f_id = row['frame']
        a_id = row['agent_id']
        n_id = row['neighagent_id']
        # Self-Interaktion überspringen (falls vorhanden)
        if n_id == a_id:
            continue

        i = frame_idx[f_id]
        j = agent_idx[a_id]
        # falls aus irgendeinem Grund kein Eintrag, skippen
        if n_id not in neighbor_idx[a_id]:
            continue
        k = neighbor_idx[a_id][n_id]

        # Features setzen
        for fi, col in enumerate(cols):
            array[i, j, k, fi] = row[col]

    # 5) In PyTorch-Tensor umwandeln
    tensor = torch.from_numpy(array)  # dtype=float32

    return tensor

def shape_str(tensor):
    return "(" + ", ".join(str(dim) for dim in tensor.shape) + ")"