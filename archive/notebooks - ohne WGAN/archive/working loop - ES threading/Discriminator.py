import torch
import torch.nn as nn
from notebooks.utils.OpenAI_ES import *
from notebooks.utils.train_utils import *

# pref ref: https://github.com/Bigpig4396/PyTorch-Generative-Adversarial-Imitation-Learning-GAIL/blob/master/GAIL_OppositeV4.py
# ref: https://github.com/deligentfool/GAIL_pytorch/blob/master/net.py
# ref: https://github.com/jatinarora2702/gail-pytorch/blob/master/gail/main.py

# Input Layer: state_dim + action_dim = 5
# Output Layer: Binary Classification = 1
# Hidden layers: like majoritiy of references

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, tensor):
        x = torch.relu(self.fc1(tensor))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.flatten() # D(s,a) = P(„kommt von Expert“∣(s,a))

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
        loss_discriminator = compute_discriminator_loss(exp_scores, gen_scores, lambda_gp, grad_penalty) #ggf. WGAN-Loss direkt
        
        optim_dis.zero_grad()
        loss_discriminator.backward()
        optim_dis.step()

        return loss_discriminator.item()