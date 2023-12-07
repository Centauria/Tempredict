import argparse
import torch
from onnxscript.function_libs.torch_lib.ops import nn

from model.itrans import ITransModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser("deploy")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--output", "-o", default="model.onnx")
    args = parser.parse_args()

    itm = ITransModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    itm.eval()
    itm.to_onnx(
        args.output,
        input_sample=(torch.zeros(1, 1, 3), torch.zeros(1, 50, 3)),
        input_names=["x", "z"],
        export_params=True,
        opset_version=19,
    )
