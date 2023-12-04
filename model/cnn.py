from torch import nn, optim
from time2vec import SineActivation


class Model(nn.Module):
    def __init__(self, observe_timestep, prediction_timestep, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.t2v = SineActivation(6, 32)
        self.cnn = nn.Sequential(
            nn.LayerNorm([32, observe_timestep]),
            nn.ConvTranspose1d(32, 32, 12 - observe_timestep),
            nn.SELU(),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.Conv1d(32, 64, 7),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.Conv1d(64, 128, 7),
            nn.SELU(),
            nn.ConvTranspose1d(128, 3, prediction_timestep),
        )

    def forward(self, x):
        x = self.t2v(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        return x


model = Model(observe_timestep=10, prediction_timestep=10)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
