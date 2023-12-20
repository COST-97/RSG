from typing import Dict, Any, Optional, List, Callable
from os import path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

Reporter = Callable[[Dict[str, Any], Optional[str]], None]

def get_current_ms() -> str:
    return datetime.now().strftime("%f")

def get_reporter(name: str, desc: Optional[str] = None):
    writer = SummaryWriter(comment="_" + get_current_ms() + "_" + name)
    times_counter: Dict[str, int] = dict()

    if desc is not None:
        assert writer.log_dir is not None
        with open(path.join(writer.log_dir, "desc.txt"), "w") as f:
            f.write(desc)

    def reporter(info: Dict[str, Any], prefix: Optional[str] = None):
        nonlocal times_counter

        for _k, v in info.items():
            k = f"{prefix or 'train'}/{_k}"
            if not k in times_counter:
                times_counter[k] = 0
            writer.add_scalar(k, v, times_counter[k])
            times_counter[k] = times_counter[k] + 1

    return reporter

