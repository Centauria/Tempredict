from torch import nn, optim


class Model(nn.Module):
    def __init__(self, observe_timestep, prediction_timestep, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(9, 8, 11 - observe_timestep),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 3, prediction_timestep),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        return x


model = Model(observe_timestep=10, prediction_timestep=10)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
