from datetime import datetime
import random
from collections import deque
import os
import torch
import pickle

class Buffer:
    def __init__(self, clip_length, max_length, device="cpu"):
        self.clip_length = clip_length
        self.pred_buffer = deque(maxlen=max_length)
        self.prey_buffer = deque(maxlen=max_length)
        self.device = torch.device(device)


    def add_expert(self, path, detections):
        for name in os.listdir(path):
            file_path = os.path.join(path, name)
            if name.startswith("pred"):
                with open(os.path.join(path, f"pred_tensors_{detections}.pkl"), "rb") as f_pred:
                    pred_exp = pickle.load(f_pred)

                for single_tensor in pred_exp:                 # single: (9,1,32,5)
                    num_clips, agents, neigh, feat = single_tensor.shape
                    flat = single_tensor.reshape(num_clips * agents, neigh, feat)
                    self.pred_buffer.append(flat)

            else:
                with open(os.path.join(path, f"prey_tensors_{detections}.pkl"), "rb") as f_prey:
                    prey_exp = pickle.load(f_prey)

                for single_tensor in prey_exp:                 # single: (9,32,32,5)
                    flat = single_tensor.reshape(num_clips * agents, neigh, feat)
                    self.prey_buffer.append(flat)


    def add_generative(self, pred_tensor, prey_tensor):
        self.pred_buffer.append(pred_tensor.to(self.device))
        self.prey_buffer.append(prey_tensor.to(self.device))


    def get_latest(self, n=10, device="cpu"):
        pred_list = list(self.pred_buffer)[-n:]
        prey_list = list(self.prey_buffer)[-n:]

        pred_tensor = torch.stack(pred_list, dim=0).to(device)
        prey_tensor = torch.stack(prey_list, dim=0).to(device)
        return pred_tensor, prey_tensor
        

    def sample(self, batch_size):
        buf_len = min(len(self.pred_buffer), len(self.prey_buffer))
        idx = random.sample(range(buf_len), batch_size)
        return (torch.stack([self.pred_buffer[i] for i in idx]).float(),
                torch.stack([self.prey_buffer[i] for i in idx]).float())


    def __len__(self):
        return len(self.pred_buffer)


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

        
    def trim(self, num=100):
        to_remove = min(num, len(self))
        for _ in range(to_remove):
            self.pred_buffer.popleft()
            self.prey_buffer.popleft()
            

    def clear(self):
        self.pred_buffer.clear()
        self.prey_buffer.clear()