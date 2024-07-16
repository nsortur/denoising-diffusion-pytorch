import time
from hydra_zen import instantiate, launch, load_from_yaml
from mlflow import MlflowClient, set_tracking_uri
import pytorch_lightning as pl
from typing import Any, Dict, Optional
from rai_toolbox._utils import value_check
from rai_toolbox.mushin.workflows import _task_calls, hydra_list, multirun
from rai_toolbox.mushin.hydra import zen
from omegaconf import OmegaConf
import torch as tr
import logging
from torch import nn
import pickle
import datetime
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class _BaseWorkflow:
    @staticmethod
    def pre_task(global_seed: int, mlflow_tracking_uri: Optional[str] = None):
        pl.seed_everything(global_seed)

        if mlflow_tracking_uri is not None:
            set_tracking_uri(mlflow_tracking_uri)

    @staticmethod
    def task(cfg):
        pass

    @classmethod
    def run(
        cls,
        cfg,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        to_dictconfig: bool = True,
        version_base: str = "1.1",
    ):
        launch_overrides = []
        if overrides is not None:
            for k, v in overrides.items():
                value_check(
                    k, v, type_=(int, float, bool, str, dict, multirun, hydra_list)
                )
                if isinstance(v, multirun):
                    v = ",".join(str(item) for item in v)

                launch_overrides.append(f"{k}={v}")

        return launch(
            cfg,
            _task_calls(
                pre_task=zen(cls.pre_task),
                task=zen(cls.task),
            ),
            overrides=launch_overrides,
            multirun=True,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
        )


class TrainDiffusionWorkflow(_BaseWorkflow):

    @staticmethod
    def task(trainer):
        meta = trainer.train()
        plt.figure()
        plt.plot(meta["losses"])
        plt.xlabel("Step")
        plt.ylabel("Train loss")
        plt.savefig("losses.png")
