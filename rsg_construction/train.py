import setup
import csv
from typing import Tuple, Optional
import numpy as np
from common import seed
import pickle
from skill_graph import SkillGraph
from os import path
import os
import torch
from torch import nn
from args import args

FOLDER_NAME = "./param"

if __name__ == "__main__":
    print(f'in {args.name}, seed is {args.seed}')
    seed(args.seed)
    sg = SkillGraph({"hidden_dim": 256, "skill_dim": 48, "read_skill": True, "name": args.seed, 'eval': False})
    # sg.vis("task")
    # sg.vis("env")
    print('train start')
    sg.train_kgc({"batch_size": 256, "train_iters": int(1.2e4)})
    print('end')
    exit(0)
    # sg.read(read_skill=True)
    print(sg.get_statistics())
    # sg.draw_in_neo4j()
    # exit(0)
    # sg.pretrain_encoder()
    # sg.perform_tsne('env')
    # print(sg.get_statistics())
    # sg.update_min_max()
    # sg.train_kgc(
    #     {"hidden_dim": 8, "skill_dim": 8, "batch_size": 64, "train_iters": int(1.5e4)}
    # )
    # sg.perform_tsne("env", latent=True)
    for id in range(10):
        query_env = {
            "foot_force": np.random.uniform(-50, 50, size=(N, 4)),
            "mass_central": np.random.uniform(-50, 50, size=(N, 6)),
        }

        query_task = {
            "body_move": np.random.uniform(-50, 50, size=(N, 6)),
            "legs_move": np.random.uniform(-50, 50, size=(N, 4, 3)),
        }

        # find, _ = sg.kgc(
        #     4,
        #     query_env=query_env,
        #     query_task=query_task,
        # )
        sg.knn(sg.data_len,
               query_env=query_env,
               query_task=query_task,
               reverse=id % 2 == 0)
        # idxs = [all.index(f) for f in find]
        # print(f"nearest neighbors that kgc find: {idxs}")
        sg.random_knn(5)
    print("end")

    # model_params = pick
