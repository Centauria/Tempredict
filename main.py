import os
import argparse
import torch
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.utils.data as data

from data import dataset, SerialDataset
from model.itrans_lstm import ITransLSTM

condition_channels = [
    "SPEED",
    "TORQUE",
    "ESS1_ACT_U",
    "WC_Act_TempIn2",
    "WC_Act_TempOut2",
]

prediction_channels = [
    "Temp_MotorCoilAve",
    "Temp_MotorMagnetAve",
    "Temp_MotorBearingAve",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("temp_predict")
    parser.add_argument(
        "--train-data",
        required=True,
        help="Specify training data in mf4 format",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Specify training data in mf4 format",
    )
    parser.add_argument("-o", "--output-model-path")
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("--observe-timestep", type=int, default=1)
    parser.add_argument("--prediction-timestep", type=int, default=50)
    parser.add_argument("--mlp-layer-num", type=int, default=2)
    parser.add_argument("--block-num", type=int, default=4)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--data-fix", action="store_true")
    args = parser.parse_args()

    dataset_train = SerialDataset(
        args.train_data,
        prediction_channels,
        condition_channels,
        observe_timestep=args.observe_timestep,
        prediction_timestep=args.prediction_timestep,
        missing_channel_behavior="fix" if args.data_fix else "discard",
    )
    dataset_test = SerialDataset(
        args.test_data,
        prediction_channels,
        condition_channels,
        observe_timestep=args.observe_timestep,
        prediction_timestep=args.prediction_timestep,
        missing_channel_behavior="fix" if args.data_fix else "discard",
    )

    print(len(dataset_train), len(dataset_test))

    train_set_size = int(len(dataset_train) * 0.8)
    valid_set_size = len(dataset_train) - train_set_size
    seed = torch.Generator().manual_seed(42)
    dataset_train, dataset_valid = data.random_split(
        dataset_train, [train_set_size, valid_set_size], generator=seed
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=256,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=256,
        num_workers=5,
        persistent_workers=True,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=256,
        num_workers=5,
        persistent_workers=True,
    )

    model_lightning = ITransLSTM(
        args.observe_timestep,
        args.prediction_timestep,
        len(prediction_channels),
        len(condition_channels),
        args.mlp_layer_num,
        block_num=args.block_num,
    )
    trainer = Trainer(
        val_check_interval=500,
        max_epochs=50,
        default_root_dir="checkpoints",
    )
    # tuner = Tuner(trainer)
    # tuner.lr_find(model_lightning, loader_train, loader_valid, min_lr=1e-5, max_lr=1e-3)
    trainer.fit(model_lightning, loader_train, loader_valid)
    trainer.test(model_lightning, loader_test)
