from datetime import datetime
import random
from collections import deque
import os
import torch

class Buffer:
    def __init__(self, clip_length, max_length):
        self.clip_length = clip_length
        self.pred_buffer = deque(maxlen=max_length)
        self.prey_buffer = deque(maxlen=max_length)


    def add_expert(self, path):
        for name in os.listdir(path):
            file_path = os.path.join(path, name)
            if name.endswith("pred.pt"):
                pred_tensor = torch.load(file_path, weights_only=False)
                self.pred_buffer.append(pred_tensor)
            else:
                prey_tensor = torch.load(file_path, weights_only=False)
                self.prey_buffer.append(prey_tensor)


    def add_generative(self, pred_tensor, prey_tensor):
        self.pred_buffer.append(pred_tensor)
        self.prey_buffer.append(prey_tensor)



    def sample(self, batch_size, device="cpu"):
        buf_len = min(len(self.pred_buffer), len(self.prey_buffer))
        idx = random.sample(range(buf_len), batch_size)
        return (torch.stack([self.pred_buffer[i] for i in idx]).to(device),
                torch.stack([self.prey_buffer[i] for i in idx]).to(device))


    def __len__(self):
        return len(self.pred_buffer)


    def shuffle(self):
        paired = list(zip(self.pred_buffer, self.prey_buffer))
        random.shuffle(paired)

        self.pred_buffer.clear()
        self.prey_buffer.clear()

        for pred, prey in paired:
            self.pred_buffer.append(pred)
            self.prey_buffer.append(prey)


    def load(self, path, type="expert"):
        if type == "expert":
            pred_path = os.path.join(path, "pred_expert_buffer.pt")
            prey_path = os.path.join(path, "prey_expert_buffer.pt")  
        else:
            pred_path = os.path.join(path, "pred_generative_buffer.pt")
            prey_path = os.path.join(path, "prey_generative_buffer.pt")

            pred_tensor = torch.load(pred_path)
            prey_tensor = torch.load(prey_path)

            self.pred_buffer.clear()
            self.prey_buffer.clear()

            self.pred_buffer.extend(loaded_pred)
            self.prey_buffer.extend(loaded_prey) 


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