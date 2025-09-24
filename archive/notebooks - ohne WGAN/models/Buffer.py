import os
import cv2
import torch
import pickle
import random
import numpy as np
from collections import deque
from marl_aquarium.env.utils import scale

class Buffer:
    def __init__(self, pred_max_length, prey_max_length, device="cpu"):
        self.pred_buffer = deque(maxlen=pred_max_length)
        self.prey_buffer = deque(maxlen=prey_max_length)
        self.device = torch.device(device)


    def add_expert(self, path, window_size, detections):
        pred_file = os.path.join(path, f"pred_tensors_{window_size}_{detections}.pkl")
        with open(pred_file, "rb") as f:
            pred_list = pickle.load(f)

        for single_tensor in pred_list:
            flat = single_tensor.squeeze(1)
            for clip in flat:
                self.pred_buffer.append(clip.to(self.device).float())

        prey_file = os.path.join(path, f"prey_tensors_{window_size}_{detections}.pkl")
        with open(prey_file, "rb") as f:
            prey_list = pickle.load(f)

        for single_tensor in prey_list:
            n_clips, agent, neigh, feat = single_tensor.shape
            flat = single_tensor.reshape(n_clips * agent, neigh, feat)
            for clip in flat:
                self.prey_buffer.append(clip.to(self.device).float())


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
            

    def clear(self):
        self.pred_buffer.clear()
        self.prey_buffer.clear()


class Pool:
    def __init__(self, max_length, device="cpu"):
        self.pool = deque(maxlen=max_length)
        self.device = torch.device(device)


    def generate_startframes(self, video_path, full_track_windows):
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


    def sample(self, n=1):
        sampled_frame = random.sample(list(self.pool), n)
        return sampled_frame[0]


    def __len__(self):
        return len(self.pool)


    def clear(self):
        self.pool.clear()