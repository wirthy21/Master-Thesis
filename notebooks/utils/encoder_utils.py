import sys, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.ModularNetworks import Attention

### Encoder
class NeighborPooling(nn.Module):
    def __init__(self, features=4, embd_dim=32):
        super().__init__()

        in_dim = features * 2

        self.embed = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, embd_dim),
            nn.LeakyReLU(0.1),
        )

        self.attention = Attention(in_dim)

    def forward(self, states, neigh_mask=None, feat_mask=None):
        if feat_mask is None:
            feat_mask = torch.ones_like(states)

        states = states * feat_mask

        cat_states = torch.cat([states, feat_mask], dim=-1)

        embed = self.embed(cat_states)
        weights_logit = self.attention(cat_states)

        if neigh_mask is not None:
            weights_logit = weights_logit.masked_fill(neigh_mask == 0, float("-inf"))

        weights = torch.softmax(weights_logit, dim=2)
        pooled = (embed * weights).sum(dim=2)

        return pooled
    

class AgentEmbedding(nn.Module):
    def __init__(self, embd_dim=32, z=32):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(embd_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, z)
        )

    def forward(self, pooled_embd):
        embed = self.embed(pooled_embd)
        return embed


class TransitionEncoder(nn.Module):
    def __init__(self, features=4, embd_dim=32, z=32):
        super().__init__()
        self.z = z
        self.neigh_pooling = NeighborPooling(features=features, embd_dim=embd_dim)
        self.agent_pooling = AgentEmbedding(embd_dim=embd_dim, z=z)

    def forward(self, states, neigh_mask=None, feat_mask=None):
        batch, frames, agents, neigh, features = states.shape
        flat = states.reshape(batch * frames, agents, neigh, features)

        if neigh_mask is not None:
            if neigh_mask.dim() == 4:
                neigh_mask = neigh_mask.unsqueeze(-1)
            neigh_mask = neigh_mask.reshape(batch * frames, agents, neigh, 1)

        if feat_mask is not None:
            if feat_mask.size(3) == 1 and neigh != 1:
                feat_mask = feat_mask.expand(batch, frames, agents, neigh, features)
            feat_mask = feat_mask.reshape(batch * frames, agents, neigh, features)

        pooled_embd = self.neigh_pooling(flat, neigh_mask=neigh_mask, feat_mask=feat_mask)
        pooled = pooled_embd.view(batch, frames, agents, -1)

        z_state = self.agent_pooling(pooled)
        
        z_t   = z_state[:, :-1]
        z_tp1 = z_state[:,  1:]
        dz = z_tp1 - z_t
        transition_feature = torch.cat([z_t, dz], dim=-1)
        return z_state, transition_feature
    

class TrajectoryAugmentation(nn.Module):
    def __init__(self, noise_std=0.01, neigh_drop=0.10, feat_drop=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.neigh_drop = neigh_drop
        self.feat_drop = feat_drop

    def forward(self, states):
        batch, frames, agents, neigh, features = states.shape
        device = states.device

        neigh_mask = torch.ones((batch, frames, agents, neigh, 1), device=device, dtype=states.dtype)
        feat_mask  = torch.ones((batch, frames, agents, neigh, features), device=device, dtype=states.dtype)

        if self.neigh_drop > 0:
            neigh_mask = (torch.rand(batch, frames, agents, neigh, 1, device=device) > self.neigh_drop).float()

        if self.feat_drop > 0:
            feat_mask = (torch.rand(batch, frames, agents, 1, features, device=device) > self.feat_drop).float()
            feat_mask = feat_mask.expand(batch, frames, agents, neigh, features)

        if features == 5:
            feat_mask[..., 0] = 1.0

        if self.noise_std > 0:
            noise = torch.randn_like(states) * self.noise_std
            if features == 5:
                noise[..., 0] = 0.0
            states = states + noise

        return states, neigh_mask, feat_mask


class VicRegProjector(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128))
        
    def forward(self, states):
        return self.net(states)


# https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
def off_diagonal(tensor):
    dim = tensor.size(0)
    mask = ~torch.eye(dim, dtype=torch.bool, device=tensor.device)
    return tensor[mask]

def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=15.0, cov_coeff=5.0, eps=1e-4):

    sim_loss = torch.mean((z1 - z2) ** 2)

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    batch, dim = z1.shape
    cov_z1 = (z1.T @ z1) / (batch - 1)
    cov_z2 = (z2.T @ z2) / (batch - 1)
    cov_loss = (off_diagonal(cov_z1).pow(2).sum() / dim) + (off_diagonal(cov_z2).pow(2).sum() / dim)

    loss = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
    logs = {
        "sim": sim_loss.detach(),
        "std": std_loss.detach(),
        "cov": cov_loss.detach(),
        "std_mean": 0.5 * (std_z1.mean().detach() + std_z2.mean().detach()),
    }
    return loss, logs


def sample_data(data, batch_size=10, window_len=10):
    if data.dim() == 5: # expert_case
        idx = torch.randint(0, data.size(0), (batch_size,), device=data.device)
        return data[idx]
    
    if data.dim() == 4: # generative case
        n_samples = data.size(0)
        idx = torch.randint(n_samples - window_len + 1, (batch_size,), device=data.device)
        return torch.stack([data[i:i + window_len] for i in idx.tolist()])


def train_encoder(encoder, projector, aug, exp_tensor, epochs, optimizer, role="prey"):
    device = next(encoder.parameters()).device

    for epoch in range(1, epochs + 1):
        encoder.train()
        projector.train()

        expert_batch = sample_data(exp_tensor, batch_size=10, window_len=10)
        expert_batch = expert_batch.to(device, non_blocking=True)

        states = expert_batch[..., :-1]

        x1, neigh_mask1, feat_mask1 = aug(states)
        x2, neigh_mask2, feat_mask2 = aug(states)

        z_state1, trans1 = encoder(x1, neigh_mask=neigh_mask1, feat_mask=feat_mask1)
        z_state2, trans2 = encoder(x2, neigh_mask=neigh_mask2, feat_mask=feat_mask2)
        
        r1 = trans1.reshape(-1, trans1.size(-1))
        r2 = trans2.reshape(-1, trans2.size(-1))

        y1 = projector(r1)
        y2 = projector(r2)

        loss, logs = vicreg_loss(y1, y2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(projector.parameters()), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch:03d}: loss={loss.item():.6f} "
                f"sim={logs['sim'].item():.4f} std={logs['std'].item():.4f} "
                f"cov={logs['cov'].item():.4f} std_mean={logs['std_mean'].item():.3f}")