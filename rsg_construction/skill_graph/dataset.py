import torch
from typing import List, Tuple, Dict, Any
from typing_extensions import Self
import random
import numpy as np
from .utils.sample import random_with_fixed_sum
from copy import deepcopy


class Dataset:

    def __init__(self, ) -> None:
        self._h = None
        self._w = None
        self._d = None
        self._t = None
        self._i = None
        self._len = 0

    def add(self, hwdti: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, List[Dict[str, Any]]]):
        h, w, d, t, i = hwdti
        assert h.size(0) == w.size(0) == d.size(0) == t.size(0) == len(i)
        if self._len == 0:
            self._h = h
            self._w = w
            self._d = d
            self._t = t
            self._i = i
            self._len += h.size(0)
            return self

        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

        self._h = torch.concatenate((self._h, h))
        self._w = torch.concatenate((self._w, w))
        self._d = torch.concatenate((self._d, d))
        self._t = torch.concatenate((self._t, t))
        self._i.extend(i)
        self._len += h.size(0)

        return self

    def len(self):
        return self._len

    def sample(self, batch_size: int, repeats: int = 1, all: bool = False):
        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

        indices = random.choices(list(range(self._len)), k=batch_size) if not all else list(range(self._len))
        _is = []
        for id in indices:
            for _ in range(repeats):
                _is.append(self._i[id])

        return (self._h[indices].repeat_interleave(repeats, dim=0),
                self._w[indices].repeat_interleave(repeats, dim=0),
                self._d[indices].repeat_interleave(repeats, dim=0),
                self._t[indices].repeat_interleave(repeats, dim=0), _is)

    def _validate(self):
        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

    def split(self, percents: Tuple[float, float]) -> Tuple[Self, Self]:

        assert self._h is not None and self._w is not None and self._d is not None and self._t is not None and self._i is not None

        l = self._len
        ns = tuple(map(lambda n: int(n * l), percents))

        assert sum(ns) == l

        ids = []

        all_ids = list(range(l))
        random.shuffle(all_ids)
        for i in range(len(percents)):
            ids.append(all_ids[-ns[i]:])
            del all_ids[-ns[i]:]

        assert len(all_ids) == 0
        assert sum(map(len, ids)) == l

        rlts = tuple([Dataset() for _ in range(len(percents))])

        for i, r in enumerate(rlts):
            id = ids[i]
            _is = [self._i[j] for j in id]
            r.add((self._h[id], self._w[id], self._d[id], self._t[id], _is))

        return rlts
