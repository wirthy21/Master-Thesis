import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseInteraction(nn.Module):
    def __init__(self, features):
        super(PairwiseInteraction, self).__init__()
        self.fc1 = nn.Linear(features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)  # mu, sigma

    def forward(self, states):
        params = F.relu(self.fc1(states))
        params = F.relu(self.fc2(params))
        params = F.relu(self.fc3(params))
        params = F.relu(self.fc4(params))
        params = F.relu(self.fc5(params))
        params = self.fc6(params)

        mu, var_logit = params[:, :1], params[:, 1:]
        var = F.softplus(var_logit) + 1e-6 # ensure positivity
        sigma = var.sqrt()
        return mu, sigma


class Attention(nn.Module):
    def __init__(self, features):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1) # weight

    def forward(self, states):
        weight = F.relu(self.fc1(states))
        weight = F.relu(self.fc2(weight))
        weight = F.relu(self.fc3(weight))
        weight = F.relu(self.fc4(weight))
        weight = F.relu(self.fc5(weight))
        weight = self.fc6(weight)
        return weight
    

class PredatorInteraction(nn.Module):
    def __init__(self, features):
        super(PredatorInteraction, self).__init__()
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # mu, sigma

    def forward(self, states):
        params = F.relu(self.fc1(states))
        params = F.relu(self.fc2(params))
        params = self.fc3(params)

        mu, var_logit = params[:, :1], params[:, 1:]
        var = F.softplus(var_logit) + 1e-6
        sigma = var.sqrt()
        return mu, sigma