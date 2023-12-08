import argparse
import torch
import os
import ncnn
import shutil

from model.itrans import ITransModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser("deploy")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--output", "-o", default="model-ncnn")
    parser.add_argument("--pnnx-path", default="pnnx")
    args = parser.parse_args()

    itm = ITransModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    itm.eval()
    x = torch.rand(1, 1, 3, dtype=torch.float32)
    z = torch.rand(1, 50, 3, dtype=torch.float32)
    mod = itm.to_torchscript(method="trace", example_inputs=(x, z))
    os.makedirs("tmp", exist_ok=True)
    mod.save("tmp/model.pt")

    os.system(
        f"{args.pnnx_path} tmp/model.pt inputshape={','.join([list(v.shape) for v in (x, z)])}"
    )

    os.makedirs(args.output, exist_ok=True)
    shutil.move("tmp/model.ncnn.param", args.output)
    shutil.move("tmp/model.ncnn.param", args.output)
    shutil.rmtree("tmp")
