import torch
from torch import nn


def normalize_embeds(embeds: nn.Embedding):
    # embeds.weight = torch.tensor([
    #     embeds.weight[i] / torch.linalg.norm(embeds.weight[i])
    #     for i in range(embeds.weight.shape[0])
    # ])
    # embeds.requires_grad_(False)
    with torch.no_grad():
        for i in range(embeds.weight.size(0)):
            _norm = torch.linalg.norm(embeds.weight[i])
            embeds.weight[i].div_(_norm +
                                  torch.randn_like(_norm) * 0.2 * _norm)
    # embeds.requires_grad_(True)


def orth_to(u: torch.Tensor, v: torch.Tensor):
    # assert u.inner(v).abs() > 0.2

    orth_vector = u.inner(u) * v - v.inner(u) * u
    assert u.inner(orth_vector) < 1e-4

    return orth_vector + torch.randn_like(orth_vector) * 0.2 * orth_vector
