import torch 
import torch.nn as nn

from functools import partial
from model.afno.afno2d import afno2d
# from timm.models.layers import DropPath
from model.afno.mlp import mlp

class block(nn.Module):
    def __init__(
            self,
            config,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        ):
        super().__init__()
        self.norm1 = norm_layer(config.data.emd_dim)
        self.filter = afno2d(config) 
        # self.drop_path = DropPath(config.afno2d.drop_path) if config.afno2d.drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(config.data.emd_dim)
        mlp_hidden_dim = int(config.data.emd_dim * config.afno2d.mlp_ratio)
        self.mlp = mlp(config)
        self.double_skip = config.afno2d.double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

