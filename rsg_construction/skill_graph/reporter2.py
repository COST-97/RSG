from typing import Dict, Any, Optional, List, Callable, Tuple
from os import path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

def get_current_ms() -> str:
    return datetime.now().strftime("%f")

class Reporter:
    def __init__(self, writer, counter):
        self._writer = writer
        self._times_counter = counter

    def add_scalars(self, info: Dict[str, Any], prefix: str):
        for _k, v in info.items():
            k = f"{prefix}/{_k}"

            if isinstance(v, tuple):
                assert isinstance(v[1], int)
                assert k not in self._times_counter or self._times_counter[k] < v[1]
                val = v[0]
                self._writer.add_scalar(k, val, v[1])
                self._times_counter[k] = v[1]
            else:
                if not k in self._times_counter:
                    self._times_counter[k] = 0
                val = v
                self._writer.add_scalar(k, val, self._times_counter[k])
                self._times_counter[k] += 1

    def add_distributions(self, info: Dict[str, Any], prefix: str):
        for _k, v in info.items():
            k = f"{prefix}/{_k}"


            if isinstance(v, tuple):
                assert isinstance(v[1], int)
                assert k not in self._times_counter or self._times_counter[k] < v[1]
                val = v[0]
                self._writer.add_histogram(k, val, v[1])
                self._times_counter[k] = v[1]
            else:
                if not k in self._times_counter:
                    self._times_counter[k] = 0
                val = v
                self._writer.add_histogram(k, val, self._times_counter[k])
                self._times_counter[k] += 1

    def add_videos(self, info: Dict[str, Tuple[np.ndarray, int]], prefix: str):
        for _k, (video, step) in info.items():
            k = f"{prefix}/{_k}"

            self._writer.add_video(k, video, step)


def get_reporter(name: str, desc: Optional[str] = None):
    writer = SummaryWriter(comment="_" + get_current_ms() + "_" + name)
    times_counter: Dict[str, int] = dict()

    if desc is not None:
        assert writer.log_dir is not None
        with open(path.join(writer.log_dir, "desc.txt"), "w") as f:
            f.write(desc)

    return Reporter(writer, times_counter), writer.get_logdir()

