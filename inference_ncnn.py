import ncnn
import numpy as np
import argparse

if __name__ == "__main__":
    net = ncnn.Net()
    net.load_param("trained_models_ncnn/model.ncnn.param")
    net.load_model("trained_models_ncnn/model.ncnn.bin")

    x = np.random.rand(1, 1, 3).astype(np.float32)
    z = np.random.rand(1, 50, 3).astype(np.float32)

    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(x).clone())
        ex.input("in1", ncnn.Mat(z).clone())

        ret, out = ex.extract("out0")
        print(ret)
        print(out.numpy())
