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

def process_frame(cap, model, tracker, frame_idx):
    _, frame = cap.read()
    height, width = frame.shape[:2]
    result = model(frame, verbose=False)[0]

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


def find_valid_windows(filtered_frames, num_frames=9, total_detections=33):
    det_by_frame = defaultdict(list)
    for d in filtered_frames:
        det_by_frame[d['frame']].append(d['track_id'])
    
    all_frames = sorted(det_by_frame.keys())
    
    valid_windows =[]
    for start in range(all_frames[0], all_frames[-1] - num_frames + 1):
        track_ids = set(det_by_frame.get(start, [])) #get track_ids for the start frame
        
        # update track_ids for the next num_frames
        for f in range(start + 1, start + num_frames):
            frame_ids = set(det_by_frame.get(f, []))
            track_ids &= frame_ids
            if not track_ids:
                break #if no track_ids left, break early
        
        if len(list(track_ids)) == total_detections:
            valid_windows.append({"start_frame": start, "ids": list(track_ids)}) #save start frame of the valid window
    
    full_track_windows = []
    for window in valid_windows:
        start = window['start_frame']
        ids = set(window['ids'])
        frames = set(range(start, start + num_frames))

        window_data = [data for data in filtered_frames if data['frame'] in frames and data['track_id'] in ids]
        full_track_windows.append(window_data)
        
    return full_track_windows, valid_windows


def get_expert_features(frame, width, height, max_speed=25):    
    frame = sorted(frame, key=lambda det: det['label'] != '1') # sort so that Pred Head is always first

    vscale = np.vectorize(scale)

    xs = np.array([det['x'] for det in frame])
    ys = np.array([det['y'] for det in frame])

    clipped_xs = np.clip(xs, 0, width)
    clipped_ys = np.clip(ys, 0, height)

    scaled_xs = vscale(clipped_xs, 0, width, 0, 1)
    scaled_ys = vscale(clipped_ys, 0, height, 0, 1)

    vxs = np.array([det['vx'] for det in frame])
    vys = np.array([det['vy'] for det in frame])

    thetas = np.array([det['angle'] for det in frame])
    scaled_thetas = vscale(thetas, -np.pi, np.pi, -1, 1)

    cos_t = np.cos(thetas)                        
    sin_t = np.sin(thetas)

    # pairwise distances
    dx = scaled_xs[None, :] - scaled_xs[:, None]
    dy = scaled_ys[None, :] - scaled_ys[:, None]

    # relative velocities
    rel_vx = cos_t[:, None] * vxs[None, :] + sin_t[:, None] * vys[None, :]
    rel_vy = -sin_t[:, None] * vxs[None, :] + cos_t[:, None] * vys[None, :]

    clipped_vx = np.clip(rel_vx, -max_speed, max_speed)
    clipped_vy = np.clip(rel_vy, -max_speed, max_speed)

    scaled_rel_vx = vscale(clipped_vx, -max_speed, max_speed, -1, 1)
    scaled_rel_vy = vscale(clipped_vy, -max_speed, max_speed, -1, 1)

    n = scaled_xs.shape[0]
    thetas_mat = np.tile(scaled_thetas[:, None], (1, n))
    features = np.stack([dx, dy, scaled_rel_vx, scaled_rel_vy, thetas_mat], axis=-1)

    mask = ~np.eye(n, dtype=bool) # shape (N, N)
    neigh = features[mask].reshape(n, n-1, 5)

    pred_tensor = torch.from_numpy(neigh[0]).unsqueeze(0)
    prey_tensor = torch.from_numpy(neigh[1:]) # shape (N-1, N-1, 5)

    return pred_tensor, prey_tensor


def get_expert_tensors(full_track_windows, valid_windows, width, height, max_speed=25, window_size=9):
    if len(valid_windows) == 0:
        return torch.empty(0), torch.empty(0)
    
    else:
        start_frames = [vw['start_frame'] for vw in valid_windows]
        pred_windows = []
        prey_windows = []

        for idx, start in enumerate(start_frames):
            window_detections = []
            for frame in range(start, start + window_size):
                dets = [det for det in full_track_windows[idx] if det['frame'] == frame]
                window_detections.append(dets)

            preds = []
            preys = []
            for dets in window_detections:
                pred_tensor, prey_tensor = get_expert_features(dets, width, height, max_speed)
                preds.append(pred_tensor)
                preys.append(prey_tensor)

            pred_windows.append(torch.stack(preds, dim=0))
            prey_windows.append(torch.stack(preys, dim=0))

        pred_tensor = torch.stack(pred_windows, dim=0)
        prey_tensor = torch.stack(prey_windows, dim=0)

        total, n_clips, agent, neigh, feat = pred_tensor.shape
        pred_tensors = pred_tensor.reshape(total * n_clips, agent, neigh, feat)

        total, n_clips, agent, neigh, feat = prey_tensor.shape
        prey_tensors = prey_tensor.reshape(total * n_clips, agent, neigh, feat)

        return pred_tensors, prey_tensors