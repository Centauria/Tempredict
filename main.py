import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import dataset, SerialDataset
from model.itrans import model, criterion, optimizer


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

    loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)

    device = "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"

    model.to(device)

    with tqdm(range(args.epochs)) as bar:
        for n in bar:
            for i, (x, y, z) in enumerate(loader_train):
                output = model(x.to(device), z.to(device))
                # print(x.shape, output.shape, y.shape)
                loss = criterion(output, y.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                bar.set_description(
                    f"Epoch: {n}, Iter: {i} loss: {loss.detach().cpu().item():.3f} "
                )

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model, args.output_model_path)

    model.eval()
    loss = 0
    for x, y, z in loader_test:
        output = model(x.to(device), z.to(device))
        loss += criterion(output, y.to(device)).detach().cpu().item()
    loss /= len(loader_test)
    print(f"Test loss: {loss}")
