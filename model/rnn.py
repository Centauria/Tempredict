import torch
from torch import nn, optim


class Model(nn.Module):
    def __init__(self, prediction_timestep, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prediction_timestep = prediction_timestep
        self.rnn = nn.RNN(4, 10, batch_first=True)
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, hidden = self.rnn(x)
        # output: (batch, seq_len, hidden_size)
        # hidden: (1, hidden_size)
        # x_last_frame = x[:, -1, :].repeat(1, self.prediction_timestep, 1)
        zero_input = torch.zeros(x.shape[0], self.prediction_timestep, x.shape[2])
        output, _ = self.rnn(zero_input, hidden)
        output = self.fc(output)
        return output


model = Model(prediction_timestep=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
