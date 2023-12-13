from typing import Any, Union, Sequence

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
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
        self.mlp = mlp_layers(mlp_layer_num, input_length, token_dim)
        self.t2v = mlp_layers(mlp_layer_num, predict_length, token_dim)
        self.itrans_blocks = nn.ModuleList(
            [ITransformerBlock(variate_num, token_dim, heads) for _ in range(block_num)]
        )
        self.itrans_blocks_z = nn.ModuleList(
            [ITransformerBlock(condition_num, token_dim, heads) for _ in range(block_num)]
        )
        self.lstm = nn.LSTM(variate_num, lstm_hidden_dim, batch_first=True)
        self.c2v = mlp_layers(mlp_layer_num, condition_num, lstm_hidden_dim)
        self.mlp_reduce = mlp_layers(mlp_layer_num, lstm_hidden_dim, variate_num)
        self.mlp_out = mlp_layers(mlp_layer_num, token_dim, predict_length, output_activation=None)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x, z):
        # x: (batch, T, N)
        # z: (batch, S, M)
        x = x.transpose(1, 2)
        z = z.transpose(1, 2)
        # x: (batch, N, T)
        # z: (batch, M, S)
        x = self.mlp(x)
        z = self.t2v(z)
        # x: (batch, N, D)
        # z: (batch, M, D)
        for block in self.itrans_blocks:
            x = block(x)
        for block in self.itrans_blocks_z:
            z = block(z)
        # x: (batch, N, D)
        # z: (batch, M, D)
        x = x.transpose(1, 2)
        z = z.transpose(1, 2)
        # x: (batch, D, N)
        # z: (batch, D, M)
        z = self.c2v(z)
        # z: (batch, D, H)
        z = z.mean(dim=1, keepdim=False).unsqueeze(0)
        # z: (1, batch, H)
        x, (h, c) = self.lstm(x, (torch.zeros_like(z), z))
        # x: (batch, D, H)
        x = self.mlp_reduce(x).transpose(1, 2)
        # x: (batch, N, D)
        x = self.mlp_out(x).transpose(1, 2)
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
