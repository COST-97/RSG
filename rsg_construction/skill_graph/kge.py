from typing import Dict, Any
import torch
from torch import nn
from torch.nn.parameter import Parameter
from .common import DEVICE, SCORE_FACTOR
import torch.nn.functional as F


def transH(h, w, d, t):
    assert len(h.shape) == 2 == len(w.shape) == len(d.shape) == len(t.shape)
    batch_size = h.size(0)
    embed_dim = h.size(1)
    _h = h
    _w = w / torch.linalg.norm(w, dim=-1, keepdim=True)
    _d = d
    _t = t
    scores: torch.Tensor = ((_h - (_w * _h).sum(-1, keepdim=True) * _w) + _d -
                            (_t - (_w * _t).sum(-1, keepdim=True) * _w))

    assert scores.shape == (batch_size, embed_dim)

    scores = torch.linalg.norm(scores, dim=-1)
    # scores = (scores ** 2).sum(-1)

    scores = torch.exp(-SCORE_FACTOR * scores)
    assert scores.shape == (batch_size, )
    return scores