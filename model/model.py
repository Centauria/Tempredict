from torch import nn, optim

model = nn.Sequential(
    nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
