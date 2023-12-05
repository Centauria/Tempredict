import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
from main import prediction_channels, condition_channels

if __name__ == "__main__":
    parser = argparse.ArgumentParser("inference")
    parser.add_argument("model_path")
    parser.add_argument("--data", nargs="+")
    args = parser.parse_args()

    model: torch.nn.Module = torch.load(args.model_path, map_location="cpu")
    model.eval()

    prediction_timestep = 50

    for fn in args.data:
        print(f"Processing {fn}")
        test_data = data.SerialDataset(
            fn, prediction_channels, condition_channels, 3, prediction_timestep
        )

        y_real = []
        y_pred = []
        for i in tqdm(range(len(test_data))):
            x, y, z = test_data[i]

            predict = model(x.unsqueeze(0), z.unsqueeze(0)).detach()

            y_real.append(y[prediction_timestep - 1, :])
            y_pred.append(predict[0, prediction_timestep - 1, :])

        y_real = np.array(y_real)
        y_pred = np.array(y_pred)

        for i in range(3):
            plt.plot(y_pred[:, i])
            plt.plot(y_real[:, i])
            plt.show()
