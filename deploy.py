import argparse
import torch
import os
import shutil

from model.itrans import ITransModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser("deploy")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--output", "-o", default="trained_models_ncnn")
    parser.add_argument("--pnnx-path", default="pnnx")
    args = parser.parse_args()

    itm: ITransModel = ITransModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    itm.eval()
    x = torch.rand(1, itm.hparams.input_length, itm.hparams.variate_num, dtype=torch.float32)
    z = torch.rand(1, itm.hparams.predict_length, itm.hparams.condition_num, dtype=torch.float32)
    mod = itm.to_torchscript(method="trace", example_inputs=(x, z))
    os.makedirs("tmp", exist_ok=True)
    mod.save("tmp/model.pt")

    cmd = f"cd tmp && {args.pnnx_path} model.pt inputshape=\"{','.join([str(list(v.shape)) for v in (x, z)])}\""
    print(cmd)
    os.system(cmd)

    os.makedirs(args.output, exist_ok=True)
    shutil.copy2("tmp/model.ncnn.param", args.output)
    shutil.copy2("tmp/model.ncnn.bin", args.output)
    shutil.rmtree("tmp")
