import numpy as np
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, config, act_layer=nn.GELU):
        super().__init__()
        # out_features = config.mlp.out_features or config.mlp.in_features
        # hidden_features = config.mlp.hidden_features or config.mlp.in_features
        self.fc1 = nn.Linear(config.mlp.in_features, config.mlp.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(config.mlp.hidden_features, config.mlp.out_features)
        self.drop = nn.Dropout(config.mlp.drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x