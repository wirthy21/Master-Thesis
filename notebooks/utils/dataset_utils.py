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
from marl_aquarium.env.utils import scale


# names: {0: 'Predator', 1: 'Predator Head', 2: 'Prey', 3: 'Prey Head'}

def process_frame(cap, model, tracker, frame_idx, device="cpu"):
    _, frame = cap.read()
    height, width = frame.shape[:2]
    result = model(frame, verbose=False, device=device)[0]

    xywh = result.boxes.xywh.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    raw_detections = list(zip(xywh, confs, cls_ids))
    tracks = tracker.update_tracks(raw_detections, frame=frame)

    records = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        x_raw = track.mean[0]
        y_raw = track.mean[1]
        x = float(np.clip(x_raw, 0, width))
        y = float(np.clip(y_raw, 0, height))

        records.append({"frame":    int(frame_idx),
                        "track_id": int(track.track_id),
                        "label":    str(track.det_class),
                        "conf":     track.det_conf,
                        "x":        x,
                        "y":        y,
                        "vx":       float(track.mean[4]),
                        "vy":       float(track.mean[5]),
                        "speed":    abs(float(math.hypot(track.mean[4], track.mean[5]))), #vx, vy
                        "angle":    float(math.atan2(track.mean[5], track.mean[4]))}) #vy, vx = radians
        
    return records


def filter_frames(total_frames):
    # Use only necessary detections of Pred Head and Prey
    pred_prey_frames = [frame for frame in total_frames if frame['label'] in ('1', '2')] #Pred Head 1, Prey 2

    # Filter detections with None confidence
    filtered_conf = [frame for frame in pred_prey_frames if frame['conf'] is not None]

    # Drop multiple Pred Detections
    best_pred_label = {}
    preys = []

    for data in filtered_conf:
        frame = data['frame']
        if data['label'] == '1': #Pred Head 1, Prey 2
            if frame not in best_pred_label or data['conf'] > best_pred_label[frame]['conf']:
                best_pred_label[frame] = data
        else:
            preys.append(data)

    best_pred_prey_frames = list(best_pred_label.values()) + preys

    all_speeds = [det["speed"] for det in best_pred_prey_frames]
    max_speed = max(all_speeds)

    return best_pred_prey_frames, max_speed


def find_valid_windows(filtered_frames, window_len=10, total_detections=33):
    # track ids pro frame
    ids_by_frame = defaultdict(set)
    for d in filtered_frames:
        ids_by_frame[int(d["frame"])].add(int(d["track_id"]))

    frames = sorted(ids_by_frame.keys())
    
    if not frames:
        return []

    episodes = []
    i = 0

    while i < len(frames):
        frame = frames[i]
        inital_ids = ids_by_frame[frame]

        if len(inital_ids) != total_detections:
            i += 1
            continue

        start = frame
        end = frame

        while i + 1 < len(frames):
            f_next = frames[i + 1]
            if f_next != end + 1:
                break
            if ids_by_frame[f_next] != inital_ids:
                break
            i += 1
            end = f_next

        length = end - start + 1
        
        if length >= window_len:
            episodes.append({
                "start_frame": start,
                "end_frame": end,
                "length": length,
                "ids": sorted(inital_ids)
            })

        i += 1

    return episodes


def extract_windows(episodes, window_len=5):
    windows = []

    for episode in episodes:
        clip_len = episode["length"]
        if clip_len < window_len:
            continue

        clip_start = episode["start_frame"]
        ids = episode["ids"]

        num_windows = clip_len - window_len + 1
        for offset in range(num_windows):
            window_start = clip_start + offset
            window_end = window_start + window_len - 1

            windows.append({
                "start_frame": window_start,
                "end_frame": window_end,
                "length": window_len,
                "ids": ids
            })

    return windows



def get_expert_features(frame, width, height, max_speed=10):    
    frame = sorted(frame, key=lambda det: (det['label'] != '1', int(det['track_id']))) # sort so that Pred Head is always first

    vscale = np.vectorize(scale)

    xs = np.array([det['x'] for det in frame])
    ys = np.array([det['y'] for det in frame])

    clipped_xs = np.clip(xs, 0, width)
    clipped_ys = np.clip(ys, 0, height)

    scaled_xs = vscale(clipped_xs, 0, width, 0, 1) # [0, 1]
    scaled_ys = vscale(clipped_ys, 0, height, 0, 1) # [0, 1]

    vxs = np.array([det['vx'] for det in frame]) # Range [-10.961784489688785 : 13.193770118386169]
    vys = np.array([det['vy'] for det in frame]) # Range [-9.267164570677894 : 11.038460817819471]

    thetas = np.array([det['angle'] for det in frame])
    scaled_thetas = vscale(thetas, -np.pi, np.pi, 0, 1)

    cos_t = np.cos(thetas)                        
    sin_t = np.sin(thetas)

    # pairwise distances
    dx = scaled_xs[None, :] - scaled_xs[:, None] # [-1, 1]
    dy = scaled_ys[None, :] - scaled_ys[:, None] # [-1, 1]

    # relative velocities
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    scaled_rel_vx = np.clip(rel_vx, -max_speed, max_speed) / max_speed
    scaled_rel_vy = np.clip(rel_vy, -max_speed, max_speed) / max_speed

    n = scaled_xs.shape[0]
    thetas_mat = np.tile(scaled_thetas[:, None], (1, n))
    features = np.stack([dx, dy, scaled_rel_vx, scaled_rel_vy, thetas_mat], axis=-1)

    mask = ~np.eye(n, dtype=bool) # shape (N, N)
    neigh = features[mask].reshape(n, n-1, 5)

    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:]) # shape (N-1, N-1, 5)

    return pred_tensor, prey_tensor, scaled_xs.tolist(), scaled_ys.tolist(), thetas.tolist()


def get_expert_tensors(filtered_frames, extracted_windows, width, height, max_speed=10, window_size=5):
    if len(extracted_windows) == 0:
        return torch.empty(0), torch.empty(0)
    
    dets_by_frame = defaultdict(list)
    for det in filtered_frames:
        dets_by_frame[int(det["frame"])].append(det)
    
    start_frames = [window['start_frame'] for window in extracted_windows]
    pred_windows = []
    prey_windows = []
    window_coordinates = []

    for idx, start in enumerate(start_frames):
        window_detections = []
        for frame in range(start, start + window_size):
            dets = dets_by_frame[int(frame)]
            window_detections.append(dets)

        preds = []
        preys = []
        frame_coordinates = []

        for dets in window_detections:
            pred_tensor, prey_tensor, xs, ys, thetas = get_expert_features(dets, width, height, max_speed)

            preds.append(pred_tensor)
            preys.append(prey_tensor)
            
            xy = torch.from_numpy(np.stack([xs, ys, thetas], axis=-1)).float()
            frame_coordinates.append(xy)

        pred_windows.append(torch.stack(preds, dim=0))
        prey_windows.append(torch.stack(preys, dim=0))
        window_coordinates.append(torch.stack(frame_coordinates, dim=0))

    pred_tensor = torch.stack(pred_windows, dim=0)
    prey_tensor = torch.stack(prey_windows, dim=0)
    coordinates = torch.stack(window_coordinates, dim=0)

    return pred_tensor, prey_tensor, coordinates
