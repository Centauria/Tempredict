from typing import Union
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtGui import QPainter, QColor
from PySide6.QtCore import Qt, QTimer, QPointF
from pyqtgraph.parametertree import Parameter, ParameterTree, interact
from asammdf import MDF
import pyqtgraph as pg
import numpy as np
import argparse
import ncnn
import pathlib
import torch
import lightning as L

from main import condition_channels, prediction_channels
from model.itrans import ITransModel


class WaveChart(QChart):
    def __init__(self, data, net: Union[ITransModel, ncnn.Net]):
        super().__init__()

        # 显示的时间范围
        self.t_range = 30
        self.sample_time = 0.1

        self.data = data
        self.net = net

        self.channels = prediction_channels
        self.series = {k: QLineSeries() for k in self.channels}
        for s in self.series.values():
            self.addSeries(s)
        self.prediction_series = {f"{k}_P": QLineSeries() for k in self.channels}
        for s in self.prediction_series.values():
            p = s.pen()
            p.setWidthF(5.0)
            p.setColor(QColor("#223300"))
            p.setStyle(Qt.PenStyle.DashLine)
            s.setPen(p)
            self.addSeries(s)

        # 创建坐标轴
        self.x_axis = QValueAxis()
        self.y_axis = QValueAxis()
        self.addAxis(self.x_axis, Qt.AlignBottom)
        self.addAxis(self.y_axis, Qt.AlignLeft)
        self.x_axis.setTickCount(11)

        for s in self.series.values():
            s.attachAxis(self.x_axis)
            s.attachAxis(self.y_axis)
        for s in self.prediction_series.values():
            s.attachAxis(self.x_axis)
            s.attachAxis(self.y_axis)

        # 初始化x的值
        self.n = 0

        # 设置y轴的范围
        self.x_axis.setMin(self.t)
        self.x_axis.setMax(self.t_range)
        self.y_axis.setMin(0)
        self.y_axis.setMax(120)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleTimeout)
        self.timer.start(50)

        self.resize(800, 500)

    def handleTimeout(self):
        if self.n + int(self.t_range * 10) < self.data.shape[0]:
            if isinstance(self.net, ncnn.Net):
                f = predict_ncnn
            elif isinstance(self.net, ITransModel):
                f = predict
            pred = f(
                self.data[self.n, :3].reshape(1, 3),
                self.data[self.n + 1 : self.n + 101, 3:],
                self.net,
            )
            for i, k in enumerate(self.channels):
                self.series[k].replace(
                    [
                        QPointF(x / 10, self.data[x, i])
                        for x in range(self.n, self.n + int(self.t_range * 10))
                    ]
                )
                self.prediction_series[f"{k}_P"].replace(
                    [
                        QPointF((self.n + index + 1) / 10, pred[index, i])
                        for index in range(100)
                    ]
                )

            self.x_axis.setRange(self.t, self.t + self.t_range)
            self.n += 1

    @property
    def t(self):
        return self.n * self.sample_time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]

        # plot data: x, y values
        self.graphWidget.plot(hour, temperature)


def command(SPEED=0, TORQUE=0, ESS1_ACT_V=0):
    pass


def predict_ncnn(x, z, net):
    x = np.array(x, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(x))
        ex.input("in1", ncnn.Mat(z))

        _, out = ex.extract("out0")
    return out.numpy().transpose()


def predict(x, z, net):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    z = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    return net(x, z).detach().squeeze(0).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("interactive_prediction")
    parser.add_argument("--ncnn-param-file")
    parser.add_argument("--ckpt-file")
    parser.add_argument("--record-file", required=True)
    args = parser.parse_args()

    if args.ncnn_param_file is not None:
        net = ncnn.Net()
        param_file = pathlib.Path(args.ncnn_param_file)
        bin_file = param_file.with_suffix(".bin")
        net.load_param(str(param_file))
        net.load_model(str(bin_file))
    elif args.ckpt_file is not None:
        net = ITransModel.load_from_checkpoint(args.ckpt_file, map_location="cpu")
        net.eval()
    else:
        raise argparse.ArgumentError(
            message="--ncnn-param-file and --ckpt-file cannot be both empty"
        )

    channels = prediction_channels + condition_channels
    with MDF(args.record_file) as f:
        df = f.to_dataframe(channels, raster="Temp_MotorMagnetAve").reindex(
            columns=channels
        )
        dfn = df.to_numpy()

    app = QApplication()

    chart = WaveChart(dfn, net)
    for m, c in zip(
        chart.legend().markers(), chart.channels + [f"{c}_P" for c in chart.channels]
    ):
        m.setLabel(c)
    chart.setTitle("Motor Temparature Simulation")

    chart_view = QChartView(chart)
    chart_view.setRenderHint(QPainter.Antialiasing)

    params = interact(command)
    tree = ParameterTree()
    tree.setParameters(params)
    # w = MainWindow()
    # w.show()
    chart_view.show()
    # tree.show()
    app.exec()
