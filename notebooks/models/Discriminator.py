import torch
import torch.nn as nn
from utils.es_utils import *
from utils.train_utils import *

# pref ref: https://github.com/Bigpig4396/PyTorch-Generative-Adversarial-Imitation-Learning-GAIL/blob/master/GAIL_OppositeV4.py
# ref: https://github.com/deligentfool/GAIL_pytorch/blob/master/net.py
# ref: https://github.com/jatinarora2702/gail-pytorch/blob/master/gail/main.py

# Always in Fisch and all neighbors
# Input Layer: neigh * features = 5
# Output Layer: Binary Classification = 1
# Hidden layers: like majoritiy of references

class Discriminator(nn.Module):
    def __init__(self, neigh=32, features=5):
        super(Discriminator, self).__init__()
        input_dim = neigh * features
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, tensor):
        batch_size, neigh, features = tensor.shape
        x = tensor.view(batch_size, neigh * features) # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(batch_size) # D(s,a) = P(„kommt von Expert“∣(s,a))

    def set_parameters(self, init=False):
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def update(self, expert_batch, policy_batch, optim_dis, lambda_gp):
        expert_batch = expert_batch.detach()
        policy_batch = policy_batch.detach()
        
        exp_scores = self.forward(expert_batch)
        gen_scores = self.forward(policy_batch)
        grad_penalty = gradient_penalty(self, expert_batch, policy_batch)

        wasserstein_loss = compute_wasserstein_loss(exp_scores, gen_scores, lambda_gp, grad_penalty)
        
        optim_dis.zero_grad()
        wasserstein_loss.backward()
        optim_dis.step()

        return (wasserstein_loss.item(), grad_penalty.item(), exp_scores.mean().item(), gen_scores.mean().item())