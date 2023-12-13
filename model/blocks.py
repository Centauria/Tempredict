from typing import Optional

from torch import nn


def mlp_layers(
        layer_num,
        in_dim,
        out_dim,
        hidden_dim=128,
        activation=nn.ReLU,
        output_activation: Optional[type(nn.ReLU)] = nn.ReLU
):
    if layer_num <= 0:
        raise ValueError("layer num must be > 0")
    dim_list = [(in_dim, hidden_dim)] + [(hidden_dim, hidden_dim)] * (layer_num - 1)
    layer_list = [[nn.Linear(*x), activation()] for x in dim_list]
    layer_list = sum(layer_list, [])
    layer_list.append(nn.Linear(hidden_dim, out_dim))
    if output_activation is not None:
        layer_list.append(output_activation())
    return nn.Sequential(*layer_list)


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
