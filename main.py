import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import dataset, SerialDataset
from model.cnn import model, criterion, optimizer


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
    args = parser.parse_args()

    dataset_train = SerialDataset(
        args.train_data, observe_timestep=10, prediction_timestep=10
    )
    dataset_test = SerialDataset(
        args.test_data, observe_timestep=10, prediction_timestep=10
    )

    print(len(dataset_train), len(dataset_test))

    loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)

    model.to('cuda')

    with tqdm(range(args.epochs)) as bar:
        for n in bar:
            for i, (x, y) in enumerate(loader_train):
                output = model(x.to('cuda'))
                # print(x.shape, output.shape, y.shape)
                loss = criterion(output, y.to('cuda'))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                bar.set_description(f"Epoch: {n}, Iter: {i} loss: {loss.detach().cpu().item():.3f} ")

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model, args.output_model_path)

    model.eval()
    loss = 0
    for x, y in loader_test:
        output = model(x.to('cuda'))
        loss += criterion(output, y.to('cuda')).detach().cpu().item()
    loss /= len(loader_test)
    print(f"Test loss: {loss}")
