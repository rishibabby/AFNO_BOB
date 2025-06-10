import torch
import pickle 
import torch.nn as nn
import numpy as np

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from model.afno.afnonet import afnonet
from dataset.preprocessing_afno import load_and_prepare_data
from train.train_afno import Trainer

import wandb

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

# Random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
#torch.set_default_dtype(config.dtype)
torch.cuda.manual_seed_all(config.seed)

# wandb
if config.wb:
    name = f"variable: {config.data.variable}, L: {config.fno.lifting_channels}, H: {config.fno.hidden_channels}, P: {config.fno.projection_channels}, modes: {config.fno.n_modes}, layers: {config.fno.n_layers}, lr: {config.opt.lr}"
    wandb.init(project='FNO for Bay of Bengal',
                name = name,
                config=config
                )

# Load Data
train_data_loader, val_data_loader, test_dataloader, mask = load_and_prepare_data(config)

# Load model
model = afnonet(config)
model = model.to(config.device)

total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print("Total Number of trainable parameters : ", total_params)


### Gflops computation
###########################################################################
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# # Create a dummy input with the appropriate size (e.g., batch_size=1, 3 channels, 224x224 image)
# input_tensor = torch.randn(1, 11, 224, 224).to(config.device)
# # Move model to the same device as input
# model.eval()
# # Calculate FLOPs
# flops = FlopCountAnalysis(model, input_tensor)
# # Print GFLOPs
# print(f"GFLOPs: {flops.total() / 1e9:.2f}")


# ### inference time 
# ###########################################################################
# # Warm-up (especially for GPU)
# for _ in range(10):
#     _ = model(input_tensor)

# # Time measurement
# num_runs = 100
# if torch.cuda.is_available() and "cuda" in str(config.device):
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#     torch.cuda.synchronize()
#     times = []
#     for _ in range(num_runs):
#         starter.record()
#         _ = model(input_tensor)
#         ender.record()
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)  # milliseconds
#         times.append(curr_time)
#     avg_inference_time = sum(times) / len(times)
#     print(f"Average inference time: {avg_inference_time:.3f} ms")
# else:
#     times = []
#     for _ in range(num_runs):
#         start = time.time()
#         _ = model(input_tensor)
#         end = time.time()
#         times.append((end - start) * 1000)  # convert to ms
#     avg_inference_time = sum(times) / len(times)
#     print(f"Average inference time: {avg_inference_time:.3f} ms")
# exit()
##########################################################################################

# Create Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.opt.lr,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "OneCycleLR":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.opt.lr,
        steps_per_epoch = len(data_loader),
        epochs = config.opt.epochs,
    ) 
elif config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

# Loss function
# mse = nn.MSELoss()
mse = nn.L1Loss()

# Mask
trainer = Trainer(model=model, mask=mask, config=config)

if config.verbose:
    print("\n### MODEL ###\n", print(model))
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULAR ###\n", scheduler)
    print("\n### LOSSES ###\n")
    print("\n### Beginning Training ...\n")


# Train
trainer.train(train_data_loader=train_data_loader, val_data_loader=val_data_loader, mse=mse, optimizer=optimizer, scheduler=scheduler)

# wandb finish
if config.wb:
    wandb.finish()