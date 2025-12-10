import os
import cv2
import torch
import pickle
import random
import numpy as np
from collections import deque
import custom_marl_aquarium
#from marl_aquarium.env.utils import scale

class Buffer:
    def __init__(self, pred_max_length, prey_max_length, device="cpu"):
        self.pred_buffer = deque(maxlen=pred_max_length)
        self.prey_buffer = deque(maxlen=prey_max_length)
        self.device = torch.device(device)

    def add_expert(self, path):
        for file in os.listdir(path):
            if file.startswith("pred"):
                pred_path = os.path.join(path, file)
                with open(pred_path, "rb") as f:
                    pred_tensors = pickle.load(f)

                for single in pred_tensors:
                    flat = single.squeeze(1)
                    for clip in flat:
                        self.pred_buffer.append(clip)

            elif file.startswith("prey"):
                prey_path = os.path.join(path, file)
                with open(prey_path, "rb") as f:
                    prey_tensors = pickle.load(f)

                n_clips, agent, neigh, feat = prey_tensors.shape
                flat = prey_tensors.reshape(n_clips * agent, neigh, feat)
                for single_tensor in flat:
                    self.prey_buffer.append(single_tensor.to(self.device).float())

        return pred_tensors, prey_tensors



    def add_generative(self, pred_tensor, prey_tensor):
        flat_pred = pred_tensor.squeeze(1)
        for single_tensor in flat_pred:
            self.pred_buffer.append(single_tensor.to(self.device).float())

        n_clips, agent, neigh, feat = prey_tensor.shape
        flat_prey  = prey_tensor.reshape(n_clips * agent, neigh, feat)
        for single_tensor in flat_prey:
            self.prey_buffer.append(single_tensor.to(self.device).float())


    def get_latest(self, n=10, pred_count=1, prey_count=32, device="cpu"):
        pred_n = n * pred_count
        prey_n = n * prey_count
        pred_list = list(self.pred_buffer)[-pred_n:]
        prey_list = list(self.prey_buffer)[-prey_n:]

        pred_tensor = torch.stack(pred_list, dim=0).to(device)
        prey_tensor = torch.stack(prey_list, dim=0).to(device)
        return pred_tensor, prey_tensor
        

    def sample(self, pred_batch_size, prey_batch_size):
        len_pred, len_prey = self.lengths()

        pred_idx = random.sample(range(len_pred), pred_batch_size)
        prey_idx = random.sample(range(len_prey), prey_batch_size)

        pred_batch = torch.stack([self.pred_buffer[i] for i in pred_idx], dim=0).float()
        prey_batch = torch.stack([self.prey_buffer[i] for i in prey_idx], dim=0).float()

        return pred_batch, prey_batch


    def lengths(self):
        return len(self.pred_buffer), len(self.prey_buffer)


    def load(self, path, type="expert", device="cpu"):
        try:
            if type == "expert":
                pred_path = os.path.join(path, "pred_expert_buffer.pt")
                prey_path = os.path.join(path, "prey_expert_buffer.pt")  
            else:
                pred_path = os.path.join(path, "pred_generative_buffer.pt")
                prey_path = os.path.join(path, "prey_generative_buffer.pt")

            pred_tensor = torch.load(pred_path, map_location=device, weights_only=False)
            prey_tensor = torch.load(prey_path, map_location=device, weights_only=False)

            self.pred_buffer.clear()
            self.prey_buffer.clear()

            self.pred_buffer.extend(pred_tensor)
            self.prey_buffer.extend(prey_tensor)
        except:
            pass


    def save(self, path, type="expert"):       
        if type == "expert":
            pred_path = os.path.join(path, "pred_expert_buffer.pt")
            prey_path = os.path.join(path, "prey_expert_buffer.pt")
            
            torch.save(self.pred_buffer, pred_path)
            torch.save(self.prey_buffer, prey_path)

        else:
            pred_path = os.path.join(path, "pred_generative_buffer.pt")
            prey_path = os.path.join(path, "prey_generative_buffer.pt")
            
            torch.save(self.pred_buffer, pred_path)
            torch.save(self.prey_buffer, prey_path)
            

    def clear(self, p=None):
        if p is None:
            self.pred_buffer.clear()
            self.prey_buffer.clear()
        else:
            pred_remove = int(len(self.pred_buffer) * (p / 100.0))
            prey_remove = int(len(self.prey_buffer) * (p / 100.0))

            pred_idx = set(random.sample(range(len(self.pred_buffer)), pred_remove)) if pred_remove > 0 else set()
            prey_idx = set(random.sample(range(len(self.prey_buffer)), prey_remove)) if prey_remove > 0 else set()

            new_pred = deque([x for i, x in enumerate(self.pred_buffer) if i not in pred_idx], maxlen=self.pred_buffer.maxlen)
            new_prey = deque([x for i, x in enumerate(self.prey_buffer) if i not in prey_idx], maxlen=self.prey_buffer.maxlen)

            # Buffers überschreiben
            self.pred_buffer = new_pred
            self.prey_buffer = new_prey


class Pool:
    def __init__(self, max_length, device="cpu"):
        self.pool = deque(maxlen=max_length)
        self.device = torch.device(device)


    def generate_startframes_old(self, video_path, full_track_windows):
        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i in range(len(full_track_windows)):
            xs = [f["x"] for f in full_track_windows[i]]
            ys = [f["y"] for f in full_track_windows[i]]
            angles = [f["angle"] for f in full_track_windows[i]]

            vscale = np.vectorize(scale)
            scaled_xs = vscale(xs, 0, width,  0, 1)
            scaled_ys = vscale(ys, 0, height, 0, 1)
            scaled_angle = vscale(angles, -np.pi, np.pi, 0, 1)

            frame_positions = list(zip(scaled_xs, scaled_ys, scaled_angle)) #zip object only usable once - after use = 0
            self.pool.append(frame_positions)


    def generate_startframes(self, ftw_path):
        width, height = 2160, 2160

        for file in os.listdir(ftw_path):
            if file.startswith("full") and file.endswith(".pkl"):
                ftw_file = os.path.join(ftw_path, file)
                with open(ftw_file, "rb") as f:
                    full_track_windows = pickle.load(f)

                for window in full_track_windows:
                    xs = [f["x"] for f in window]
                    ys = [f["y"] for f in window]
                    angles = [f["angle"] for f in window]

                    vscale = np.vectorize(scale)

                    clipped_xs = np.clip(xs, 0, width)
                    clipped_ys = np.clip(ys, 0, height)

                    scaled_xs = vscale(clipped_xs, 0, width,  0, 1)
                    scaled_ys = vscale(clipped_ys, 0, height, 0, 1)
                    scaled_angle = vscale(angles, -np.pi, np.pi, 0, 1)

                    frame_positions = list(zip(scaled_xs, scaled_ys, scaled_angle)) #zip object only usable once - after use = 0
                    self.pool.append(frame_positions)


    def sample(self, n=1):
        sampled_frame = random.sample(list(self.pool), n)
        return sampled_frame[0]


    def __len__(self):
        return len(self.pool)


    def clear(self):
        self.pool.clear()