from hydra_zen import MISSING, builds, make_config, make_custom_builds_fn
from hydra.core.config_store import ConfigStore
import numpy as np
from dataset import ConicDiffusionDataset
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher

cs = ConfigStore.instance()
pop_builds = make_custom_builds_fn(populate_full_signature=True)
pop_part_builds = make_custom_builds_fn(populate_full_signature=True, zen_partial=True)

CONIC_DIFFUSION_DATASET = builds(
    ConicDiffusionDataset,
    root="${root_dataset}",
    abs_transform="${abs_transform}"
)

UNET = builds(
    Unet1D,
    dim="${unet_dim}",
    dim_mults=(1, 2, 4, 8),
    channels="${channels}",
    self_condition=False
)

DIFFUSION = builds(
    GaussianDiffusion1D,
    model=UNET,
    seq_length="${seq_length}",
    timesteps=1000,
    objective="pred_v"
)

TRAINER = builds(
    Trainer1D,
    diffusion_model=DIFFUSION,
    dataset=CONIC_DIFFUSION_DATASET,
    train_batch_size="${train_batch_size}",
    save_and_sample_every="${save_and_sample_every}",
    train_lr="${train_lr}",
    train_num_steps="${train_num_steps}",
    gradient_accumulate_every = 2, 
    ema_decay = 0.995,         
    amp = True
)

BASE = make_config(
    trainer=TRAINER,
    global_seed=123,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=50000,
    save_and_sample_every=10000,
    seq_length=360,
    channels=107,
    abs_transform=True,
    root_dataset="/home/gridsan/NE32716/derm/denoising-diffusion-pytorch/conic_10000.nc"
)
