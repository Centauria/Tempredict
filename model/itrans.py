import torch
from torch import nn, optim


class ITransformerBlock(nn.Module):
    def __init__(self, variate_num, token_dim, heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mha = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.ln1 = nn.LayerNorm([variate_num, token_dim])
        self.ln2 = nn.LayerNorm([variate_num, token_dim])

    def forward(self, x):
        x = self.ln1(x + self.mha(x, x, x, need_weights=False)[0])
        x = self.ln2(x + self.ff(x))
        return x


class Model(nn.Module):
    def __init__(
        self,
        input_length,
        predict_length,
        variate_num,
        token_dim=128,
        heads=32,
        block_num=4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, token_dim),
            nn.ReLU(),
        )
        self.itrans_blocks = [
            ITransformerBlock(variate_num, token_dim, heads) for _ in range(block_num)
        ]
        self.mlp_out = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, predict_length),
        )

    def forward(self, x):
        # x: (batch, T, N)
        x = x.transpose(1, 2)
        # x: (batch, N, T)
        x = self.mlp(x)
        # x: (batch, N, D)
        for block in self.itrans_blocks:
            x = block(x)
        # x: (batch, N, D)
        x = self.mlp_out(x)
        # x: (batch, N, S)
        x = x.transpose(1, 2)
        # x: (batch, S, N)
        return x


model = Model(10, 10, 3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)