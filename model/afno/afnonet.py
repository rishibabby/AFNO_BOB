import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from functools import partial
# from timm.models.layers import DropPath, trunc_normal_

from model.afno.patch import patchembed
from model.afno.block import block

class afnonet(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()
        self.img_size = config.data.img_size
        self.patch_size = (config.data.patch_size[0], config.data.patch_size[1])
        self.in_chans = config.data.in_chs
        self.out_chans = config.data.out_chs
        self.num_features = self.embed_dim = config.data.emd_dim
        self.num_blocks = config.afno2d.num_blocks 
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = patchembed(config)
        num_patches = self.patch_embed.num_patches
        # print(num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.data.emd_dim))
        self.pos_drop = nn.Dropout(p=config.data.pos_drop_rate)


        self.h = config.data.img_size[0] // self.patch_size[0]
        self.w = config.data.img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            block(config) 
        for i in range(config.afno2d.n_blocks)])


        self.head = nn.Linear(config.data.emd_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        # trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        

        x = self.patch_embed(x)
        # print(x.shape)
        # print(self.pos_embed.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        # print(x.shape)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

if __name__ == "__main__":

    # Read the configuration file

    from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./temp_afno.yaml", config_name='default', config_folder='cfg/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='cfg/')
        ]
    )
    config = pipe.read_conf()

    model = afnonet(config)
    sample = torch.randn(1, 1, 224, 224)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))