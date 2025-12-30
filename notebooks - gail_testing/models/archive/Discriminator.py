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
    def __init__(self, neigh=31, features=5):
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
        return x.view(batch_size) # f(s,a): Wasserstein-Critic-Score (higher = closer to Expert)

    def update(self, expert_batch, policy_batch, optim_dis, lambda_gp, debug_logs=False):
        expert_batch = expert_batch.detach()
        policy_batch = policy_batch.detach()
        
        exp_scores = self.forward(expert_batch)
        gen_scores = self.forward(policy_batch)
        grad_penalty = gradient_penalty(self, expert_batch, policy_batch)
        critic_term = exp_scores.mean() - gen_scores.mean()

        loss, loss_gp = compute_wasserstein_loss(exp_scores, gen_scores, lambda_gp, grad_penalty)
        
        optim_dis.zero_grad()
        loss_gp.backward()

        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if debug_logs is True:
            print("[DISC]")
            print("Critic term (Exp - Gen):", critic_term.item())
            print("Lambda GP * GP:", lambda_gp, "*", grad_penalty.item(), "=", lambda_gp * grad_penalty.item())
            print("  W-loss:", loss.item())
            print("  W-loss+GP:", loss_gp.item())
            print("  GP    :", grad_penalty.item())
            print("  grad‖ :", total_norm)
            print("  exp/gen:", exp_scores.mean().item(), gen_scores.mean().item())

        optim_dis.step()

        return (loss.item(), loss_gp.item(), grad_penalty.item(), exp_scores.mean().item(), gen_scores.mean().item())
    
    # ref: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
    def set_parameters(self, init=False):
        if init is True:
            for layer in self.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()