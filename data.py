import os.path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from asammdf import MDF

channels = [
    "SPEED",
    "TORQUE",
    "Temp_MotorCoilAve",
    "Temp_MotorMagnetAve",
]


def dataset(filename, observe_timestep=1, prediction_timestep=1):
    _, ext = os.path.splitext(filename)
    if ext == ".mf4":
        f = MDF(filename)
        df = f.to_dataframe(channels, raster="Temp_MotorMagnetAve")
        obs = df.iloc[: df.shape[0] - prediction_timestep]
        pre = df.iloc[observe_timestep:]
        return [(obs, pre)]
    elif ext == ".txt":
        op = []
        with open(filename) as config_file:
            for line in config_file.readlines():
                if line != "":
                    op.extend(
                        dataset(line.rstrip(), observe_timestep, prediction_timestep)
                    )
        return op
    else:
        raise ValueError(f"Unsupported file type: {ext}")


class SerialDataset(Dataset):
    def __init__(self, filename, observe_timestep=1, prediction_timestep=1) -> None:
        data = dataset(filename, observe_timestep, prediction_timestep)
        self.data = None
        self.observe_timestep = observe_timestep
        self.prediction_timestep = prediction_timestep

        for obs, pre in data:
            effective_length, record_length = sorted([obs.shape[0], pre.shape[0]])
            minimum_op_length = min(observe_timestep, prediction_timestep)
            allow_indexing = np.hstack(
                [
                    np.ones(effective_length - minimum_op_length + 1),
                    np.zeros(record_length - effective_length + minimum_op_length - 1),
                ]
            ).astype(int)
            obs.columns = pd.MultiIndex.from_tuples([("obs", x) for x in obs.columns])
            pre.columns = pd.MultiIndex.from_tuples([("pre", x) for x in pre.columns])
            obs.index = np.arange(obs.shape[0])
            pre.index = np.arange(pre.shape[0])
            df = pd.DataFrame({("common", "allow"): allow_indexing})
            df = df.merge(obs, left_index=True, right_index=True, how="outer")
            df = df.merge(pre, left_index=True, right_index=True, how="outer")
            if self.data is None:
                self.data = df
            else:
                self.data = pd.concat([self.data, df], ignore_index=True)

        self.index = self.data[self.data.common.allow == 1].index.to_numpy()
        self.data = self.data.to_numpy()

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, index):
        i = self.index[index]
        x = self.data[i : i + self.observe_timestep, 1:5]
        y = self.data[i : i + self.prediction_timestep, 5:9]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def read(filename):
    f = MDF(filename).resample(raster="Temp_MotorMagnetAve")
    return np.array([x.samples for x in f.select(channels)]).T
