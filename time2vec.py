import torch
from torch import nn


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.linear_0 = nn.Linear(in_features, 1)
        self.linear = nn.Linear(in_features, out_features - 1)
        self.f = torch.sin

    def forward(self, x):
        # k-1 periodic features
        v1 = self.f(self.linear(x))
        # One Non-periodic feature
        v2 = self.linear_0(x)
        return torch.cat([v1, v2], -1)
