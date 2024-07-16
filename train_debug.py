import torch
import torch as tr
import importlib
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from baselines.datasets.equivariant_mesh_arrow import EquivMeshXarrayDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import xarray as xr
from numpy.random import SeedSequence
from derm_dataset.data.mesh_conic import sample_frusta
from rfsim.functional import compute_range_profile
from rfsim.coordinate_conversions import ar2los
from rfsim.polarization import Polarization
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

class ConicDiffusionDataset(Dataset):
    def __init__(self, root, abs_transform=True, stage="test"):
        """
        abs_transform: takes abs of real and complex channels before flattening
        """
        super().__init__()
        self.dataset = xr.open_dataset(root, engine='netcdf4')
        self.abs_transform = abs_transform
        
    def __len__(self):
        return len(self.dataset.sample)
    
    def __getitem__(self, idx):
        # TODO figure out zero padding for first element
        indexed = self.dataset[dict(sample=idx)]
        length, nr, br = indexed.length.to_numpy(), indexed.nr.to_numpy(), indexed.br.to_numpy()
        rti_complex = indexed.rti.to_numpy()
        if self.abs_transform:
            rti_complex = tr.view_as_complex(tr.stack([tr.tensor(rti_complex)[:, :, 0],tr.tensor(rti_complex)[:, :, 1]]).permute(1,2,0).contiguous())
            rti_complex = tr.abs(rti_complex)
        rti_complex = rti_complex.permute((1, 0))
        cat = tr.cat([
            # tr.tensor(length).repeat(1, rti_complex.size(1)), 
            # tr.tensor(nr).repeat(1, rti_complex.size(1)),
            # tr.tensor(br).repeat(1, rti_complex.size(1)),
            rti_complex
        ], axis=0).float()
        
        # cat = np.concatenate([tr.zeros((1,)), rti_flat, length[np.newaxis], nr[np.newaxis], br[np.newaxis]])
        # cat_t = tr.tensor(cat.reshape(1, -1), dtype=tr.float32)
        # cat_t_pow2 = self._pad_pow_2(cat_t)
        
        return cat
    
    def _pad_pow_2(self, x):
        h, w = x.shape[-2:]
        h_pad = ((h-1) // 32+1) * 32 - h
        w_pad = ((w-1) // 32+1) * 32 - w
        return F.pad(x, (0, w_pad))
        

testds = ConicDiffusionDataset("/home/gridsan/NE32716/derm/denoising-diffusion-pytorch/conic_10000.nc")

model = Unet1D(
    dim = 512,
    # dim=16,
    dim_mults = (1, 2, 4, 8),
    channels = 107,
    self_condition=False
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = testds[0].size(1),
    timesteps = 1000,
    objective = 'pred_v'
)
# trainer = Trainer1D(diffusion, dataset=testds, results_folder="/home/gridsan/NE32716/derm/denoising-diffusion-pytorch/multirun/2024-07-15/15-50-05/0/results")
# trainer = Trainer1D(diffusion, dataset=testds, results_folder="/home/gridsan/NE32716/derm/denoising-diffusion-pytorch/multirun/2024-07-15/16-30-40/0/results")
trainer = Trainer1D(diffusion, dataset=testds)

meta = trainer.train()