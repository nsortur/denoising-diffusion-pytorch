import configs
from workflows import TrainDiffusionWorkflow
from rai_toolbox.mushin.workflows import hydra_list, multirun
import numpy as np

def run():
    (outputs,) = TrainDiffusionWorkflow.run(
        configs.BASE,
        overrides={
            # "train_lr" : multirun([8e-5, 8e-6]),
            "train_num_steps": 60000,
            "root_dataset": "/home/gridsan/NE32716/derm/denoising-diffusion-pytorch/conic_25000.nc",
            "+unet_dim": 512,
            # "save_and_sample_every": 10000
        },
    )


if __name__ == "__main__":
    run()
