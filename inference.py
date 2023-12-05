import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import data

if __name__ == "__main__":
    parser = argparse.ArgumentParser("inference")
    parser.add_argument("model_path")
    parser.add_argument("--data", nargs="+")
    args = parser.parse_args()

    model: torch.nn.Module = torch.load(args.model_path, map_location="cpu")
    model.eval()

    for fn in args.data:
        print(f"Processing {fn}")
        test_data = torch.tensor(data.read(fn))

        print(test_data.shape)
        test_x = torch.stack(
            [test_data[i : -2 * 10 + i + 1, 3:] for i in range(10)], dim=1
        )
        print(test_x.shape)

        predict = model(test_x).detach()

        print(predict.shape)

        for i in range(3):
            plt.plot(predict[:, 9, i])
            plt.plot(test_data[10 + 9 :, 3 + i])
            plt.show()
