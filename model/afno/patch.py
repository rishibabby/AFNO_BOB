import torch
import torch.nn as nn


class patchembed(nn.Module):
    def __init__(self, config, atm=None):
        super().__init__()
        num_patches = (config.data.img_size[1] // config.data.patch_size[1]) * (config.data.img_size[0] // config.data.patch_size[0])
        self.img_size = config.data.img_size
        self.patch_size = config.data.patch_size
        self.num_patches = num_patches
        self.dynamic_size = config.data.dynamic_size
        if atm:
            config.data.in_chs = config.data.atm_chs
        self.proj = nn.Conv2d(config.data.in_chs, config.data.emd_dim, kernel_size=config.data.patch_size, stride=config.data.patch_size)


    def forward(self, x):
        B, C, H, W = x.shape
        
        # Flexible size checking
        if not self.dynamic_size:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Patch embedding
        x = self.proj(x)
        # print(x.shape)

        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)

        # Batch, _ , embed dimension
        return x

if __name__ == "__main__":

    from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
    
    # Read the configuration file
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

    x = torch.rand(1,1,226,295)

    model = patchembed(config)

    print(model(x).shape)