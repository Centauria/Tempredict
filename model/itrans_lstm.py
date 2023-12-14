from typing import Any, Union, Sequence

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

from model.blocks import ITransformerBlock, mlp_layers


class ITransLSTM(L.LightningModule):
    def __init__(
        self,
        input_length,
        predict_length,
        variate_num,
        condition_num,
        mlp_layer_num,
        token_dim=128,
        heads=32,
        lstm_hidden_dim=64,
        block_num=4,
        lr=1e-4,
    ):
        super().__init__()
        self.extract = nn.LSTM(variate_num, lstm_hidden_dim, batch_first=True)
        self.t2v = mlp_layers(mlp_layer_num, predict_length, token_dim)
        self.itrans_blocks_z = nn.ModuleList(
            [
                ITransformerBlock(condition_num, token_dim, heads)
                for _ in range(block_num)
            ]
        )
        self.v2t = mlp_layers(mlp_layer_num, token_dim, predict_length)
        self.lstm = nn.LSTM(
            condition_num, lstm_hidden_dim, proj_size=variate_num, batch_first=True
        )
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x, z):
        # x: (batch, T, N)
        hidden_size = (1, x.shape[0], x.shape[2])
        _, (_, x) = self.extract(x)
        # x: (1, batch, H)
        # z: (batch, S, M)
        z = z.transpose(1, 2)
        # z: (batch, M, S)
        z = self.t2v(z)
        # z: (batch, M, D)
        for block in self.itrans_blocks_z:
            z = block(z)
        # z: (batch, M, D)
        z = self.v2t(z)
        z = z.transpose(1, 2)
        # z: (batch, S, M)
        x, _ = self.lstm(z, (torch.zeros(hidden_size, device=self.device), x))
        # x: (batch, S, N)
        return x

    def training_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        x, y, z = batch
        output = self(x, z)
        loss = self.criterion(output, y)
        return loss

    def test_step(self, batch, batch_index, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss = self._shared_eval_step(batch, batch_index)
        self.log("test_loss", loss)
        return

    def validation_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self._shared_eval_step(batch, batch_index)
        self.log("val_loss", loss)
        return

    def _shared_eval_step(self, batch, batch_index):
        x, y, z = batch
        output = self(x, z)
        loss = self.criterion(output, y)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", 0.8, 3
                ),
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 500,
            },
        }

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        checkpoint = ModelCheckpoint(monitor="val_loss")
        logging_lr = LearningRateMonitor(logging_interval="step")
        return [early_stop, checkpoint, logging_lr]
