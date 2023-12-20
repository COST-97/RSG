from torch import nn
import numpy as np
import torch


class Encoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(layer_init(nn.Linear(input_dim, hidden_dim)),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(hidden_dim, hidden_dim)),
                                 nn.Tanh(),
                                 layer_init(nn.Linear(hidden_dim, out_dim)),
                                 nn.Tanh())

    def forward(self, entities: torch.Tensor):
        return self.net(entities)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer