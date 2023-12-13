from typing import Any, Union, Sequence

import torch
from lightning import Callback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim
import lightning as L

from model.blocks import mlp_layers, ITransformerBlock


class Model(nn.Module):
    def __init__(
        self,
        input_length,
        predict_length,
        variate_num,
        condition_num,
        mlp_layer_num=2,
        token_dim=128,
        heads=32,
        block_num=4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mlp = mlp_layers(mlp_layer_num, input_length, token_dim)
        self.t2v = mlp_layers(mlp_layer_num, predict_length, token_dim)
        self.itrans_blocks = nn.ModuleList(
            [ITransformerBlock(variate_num, token_dim, heads) for _ in range(block_num)]
        )
        self.mlp_transform = mlp_layers(mlp_layer_num, variate_num + condition_num, variate_num)
        self.mlp_out = mlp_layers(mlp_layer_num, token_dim, predict_length, output_activation=None)

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
        # x: (batch, N, D)
        x = torch.cat([x, z], 1).transpose(1, 2)
        # x: (batch, D, N+M)
        x = self.mlp_transform(x).transpose(1, 2)
        # x: (batch, N, D)
        x = self.mlp_out(x)
        # x: (batch, N, S)
        x = x.transpose(1, 2)
        # x: (batch, S, N)
        return x


class ITransModel(L.LightningModule):
    def __init__(
        self,
        input_length,
        predict_length,
        variate_num,
        condition_num,
        mlp_layer_num,
        block_num,
        lr=1e-4,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = Model(input_length, predict_length, variate_num, condition_num, mlp_layer_num, block_num=block_num)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args)

    def training_step(
        self, batch, batch_index, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        x, y, z = batch
        output = self.model(x, z)
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
        output = self.model(x, z)
        loss = self.criterion(output, y)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
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
