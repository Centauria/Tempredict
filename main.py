import os
import argparse
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.utils.data as data

from data import dataset, SerialDataset
from model.itrans import ITransModel

condition_channels = [
    "SPEED",
    "TORQUE",
    "ESS1_ACT_U",
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
    parser.add_argument("-o", "--output-model-path", required=True)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    dataset_train = SerialDataset(
        args.train_data,
        prediction_channels,
        condition_channels,
        observe_timestep=3,
        prediction_timestep=50,
    )
    dataset_test = SerialDataset(
        args.test_data,
        prediction_channels,
        condition_channels,
        observe_timestep=3,
        prediction_timestep=50,
    )

    print(len(dataset_train), len(dataset_test))

    train_set_size = int(len(dataset_train) * 0.8)
    valid_set_size = len(dataset_train) - train_set_size
    seed = torch.Generator().manual_seed(42)
    dataset_train, dataset_valid = data.random_split(dataset_train, [train_set_size, valid_set_size], generator=seed)

    loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=256)

    device = 0 if torch.cuda.is_available() and args.cuda else "cpu"

    model_lightning = ITransModel(3, 50, 3, 3)
    trainer = Trainer(default_root_dir='checkpoints', val_check_interval=1000)
    trainer.fit(model_lightning, loader_train, loader_valid)
