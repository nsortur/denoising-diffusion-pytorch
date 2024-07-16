from torch.utils.data import Dataset
import torch as tr
import xarray as xr

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
        
        return cat
    