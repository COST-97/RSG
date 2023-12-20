import os
from os import path
import torch
from pathlib import Path
from datetime import datetime
#from tqdm import tqdm

SOURCE_FOLDER = "./logs_to_be_extracted"
TARGET_FOLDER = './new_extracted_logs'

cnt = 0
for (r, dirs, ffiles) in os.walk(SOURCE_FOLDER):
    _r = r.split('/')[-1]
    if "ActorCritic" not in _r:
        continue

    if "roll" not in r and "up" not in r and "slow" not in r and "SkillID_3" not in _r:
        continue

    # assert len(dirs) == 1, f"wrong numbers of {dirs} in {r}"
    s_dirs = list(
        sorted(dirs, key=lambda d: datetime.strptime(d, "%b%d_%H-%M-%S_")))

    dir = s_dirs[-1]
    if len(s_dirs) > 1:
        print(f"use {dir} in {s_dirs}")
    m = torch.load(
        f'{r}/{dir}/{"model_350.pt" if ("up" not in r and "roll" not in r and "slow" not in r) else "model_1000.pt"}'
    )
    Path(f'{TARGET_FOLDER}/{r}/{dir}').mkdir(parents=True, exist_ok=True)
    torch.save({k: v
                for (k, v) in m.items()},
               f'{TARGET_FOLDER}/{r}/{dir}/model.pt')
    cnt += 1
    print(cnt)
