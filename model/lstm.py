import torch
from torch import nn, optim

from time2vec import SineActivation


class Model(nn.Module):
    def __init__(self, prediction_timestep, out_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prediction_timestep = prediction_timestep
        self.out_channel = out_channel
        self.t2v = SineActivation(6, 64)
        self.rnn = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, out_channel)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.t2v(x)
        _, hidden = self.rnn(x)
        # output: (batch, seq_len, hidden_size)
        # hidden: (1, hidden_size)
        # x_last_frame = x[:, -1, :].repeat(1, self.prediction_timestep, 1)
        zero_input = torch.zeros(x.shape[0], self.prediction_timestep, x.shape[2])
        output, _ = self.rnn(zero_input, hidden)
        output = self.fc(output)
        return output


model = Model(prediction_timestep=10, out_channel=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
