from torch import nn, optim


class Model(nn.Module):
    def __init__(self, observe_timestep, prediction_timestep, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(9, 8, (11 - observe_timestep, 10)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, (prediction_timestep, 1)),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x.unsqueeze_(-1)
        x = self.cnn(x)
        x.squeeze_(-1)
        x.transpose_(1, 2)
        return x


model = Model(observe_timestep=10, prediction_timestep=10)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
