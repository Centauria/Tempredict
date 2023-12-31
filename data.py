import os.path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from asammdf import MDF
from asammdf.mdf import MdfException


def dataset(
    filename,
    observe_channels,
    prediction_channels,
    observe_timestep=1,
    prediction_timestep=1,
    missing_channel_behavior="discard",
):
    _, ext = os.path.splitext(filename)
    if ext == ".mf4":
        with MDF(filename) as f:
            ch = set(f.channels_db.keys())
            ch_req = set(observe_channels).union(prediction_channels)
            if ch.issuperset(ch_req):
                df_in = f.to_dataframe(
                    observe_channels, raster="Temp_MotorMagnetAve"
                ).reindex(columns=observe_channels)
                df_out = f.to_dataframe(
                    prediction_channels, raster="Temp_MotorMagnetAve"
                ).reindex(columns=prediction_channels)
            else:
                if missing_channel_behavior == "discard":
                    print(
                        f"{filename} have no channel named {ch_req.difference(ch)}, ignored"
                    )
                    return []
                elif missing_channel_behavior == "fix":
                    ch_diff_in = set(observe_channels).difference(ch)
                    ch_diff_out = set(prediction_channels).difference(ch)
                    ch_in = [c for c in observe_channels if c not in ch_diff_in]
                    ch_out = [c for c in prediction_channels if c not in ch_diff_out]
                    df_in = f.to_dataframe(ch_in, raster="Temp_MotorMagnetAve")
                    df_out = f.to_dataframe(ch_out, raster="Temp_MotorMagnetAve")
                    for c in ch_diff_in:
                        df_in[c] = 40.0
                    for c in ch_diff_out:
                        df_out[c] = 40.0
                    df_in = df_in.reindex(columns=observe_channels)
                    df_out = df_out.reindex(columns=prediction_channels)
            obs = df_in.iloc[:-prediction_timestep]
            pre = df_out.iloc[observe_timestep:]
        return [(obs, pre)]
    elif ext == ".txt":
        op = []
        with open(filename) as config_file:
            for line in config_file.readlines():
                if line != "":
                    op.extend(
                        dataset(
                            line.rstrip(),
                            observe_channels,
                            prediction_channels,
                            observe_timestep,
                            prediction_timestep,
                            missing_channel_behavior,
                        )
                    )
        return op
    else:
        raise ValueError(f"Unsupported file type: {ext}")


class SerialDataset(Dataset):
    def __init__(
        self,
        filename,
        prediction_channels,
        condition_channels,
        observe_timestep=1,
        prediction_timestep=1,
        missing_channel_behavior="discard",
    ) -> None:
        in_channels = prediction_channels
        out_channels = prediction_channels + condition_channels
        data = dataset(
            filename,
            in_channels,
            out_channels,
            observe_timestep,
            prediction_timestep,
            missing_channel_behavior,
        )
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
        self.in_channel_index = slice(1, 1 + len(in_channels))
        self.prediction_channel_index = slice(
            1 + len(in_channels), 1 + len(in_channels) + len(prediction_channels)
        )
        self.condition_channel_index = slice(-len(condition_channels), None)

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, index):
        i = self.index[index]
        x = self.data[i : i + self.observe_timestep, self.in_channel_index]
        y = self.data[i : i + self.prediction_timestep, self.prediction_channel_index]
        z = self.data[i : i + self.prediction_timestep, self.condition_channel_index]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32),
        )
