import os
from dataclasses import dataclass
from datetime import datetime
from os import path
from typing import (Any, Dict, List, Optional, OrderedDict, Tuple, Union,
                    Literal)

import numpy as np
import torch
from torch import optim
from pathlib import Path
from .encoder import Encoder
from .reporter2 import get_reporter
from .uid import (ENV_UUID, SKILL_UUID, TASK_UUID, UUID, gen_uuid)
from pathlib import Path
from torch import nn
import pickle
from datetime import datetime, date
from .utils.net import normalize_embeds
from .envs import ENV_DESC, ENV_DIM, ENV_NUM, ENV_SAMPLE_NUM, env_mapper
from .tasks import TOTAL_ROLLOUT, TASK_DESC, SEQUENCE_INDEX, ROLLOUT_NUM, SEQUENCE_LEN, RAW_TASK_DIM, is_simple_task, TASK_DIM, process
from itertools import product
from tqdm import tqdm
from dtaidistance.dtw_ndim import distance
from .common import DEVICE

FOLDER_NAME = "./param"


def get_current_datetime_str() -> str:
    return datetime.now().strftime("%m-%d:%H:%M:%S:%f")


@dataclass
class Task:
    uuid: TASK_UUID

    desc: str

    # 质心的期望位移，维度： N x 6
    body_move: np.ndarray

    skill_uuids: List[SKILL_UUID]

    label = "Task"

    @property
    def details(self):
        dtl = self.body_move.flatten()
        assert dtl.shape == (SEQUENCE_LEN * TASK_DIM, )
        return dtl


@dataclass
class Env:
    uuid: ENV_UUID

    desc: str

    # 摩擦系数
    friction: Tuple[float, float]

    # 平整度
    uniformness: Tuple[float, float]

    # 坡度
    slope: Tuple[float, float]

    skill_uuids: List[SKILL_UUID]
    label = "Env"

    @property
    def details(self):
        return np.hstack((self.friction, self.uniformness, self.slope))


@dataclass
class Skill:
    uuid: SKILL_UUID
    desc: str
    model_weight: Any

    env_uuids: List[ENV_UUID]
    task_uuids: List[TASK_UUID]
    skill_folder: str
    label = "Skill"

    def get_model_weight(self, folder: str):
        file = f'{folder}/{self.skill_folder}'
        dirs = os.listdir(file)
        assert len(dirs) == 1
        dir = dirs[0]
        return torch.load(f'{file}/{dir}/model.pt')


NNSkills = Tuple[List[Tuple[Env, Task, Skill]], np.ndarray]


def split(low: float, upp: float, num: int = 10):
    assert num > 1
    return [low + (upp - low) * idx / (num - 1) for idx in range(num)]


class SkillGraphBase:

    def __init__(self, config: Dict[str, Any]) -> None:

        self.envs: Dict[ENV_UUID, Env] = {}
        self.raw_env_desc_index: Dict[str, List[ENV_UUID]] = {}

        self.tasks: Dict[TASK_UUID, Task] = {}
        self.raw_task_desc_index: Dict[str, List[ENV_UUID]] = {}

        self.skills: Dict[SKILL_UUID, Skill] = {}
        self.env_task_skill_index: Dict[Tuple[ENV_UUID, TASK_UUID],
                                        SKILL_UUID] = {}

        self.data_len = 0
        self.reporter, self.log_dir = get_reporter(
            f"kgc_train_{config['name']}")

        self.hidden_dim = config["hidden_dim"]
        self.skill_dim = config["skill_dim"]

        self.env_encoder = Encoder(
            ENV_DIM,
            self.hidden_dim,
            self.skill_dim,
        ).to(DEVICE)
        self.env_optim = optim.AdamW(self.env_encoder.parameters(), lr=3e-4)

        self.task_encoder = Encoder(
            SEQUENCE_LEN * TASK_DIM,
            self.hidden_dim,
            self.skill_dim,
        ).to(DEVICE)
        self.task_optim = optim.AdamW(self.task_encoder.parameters(), lr=3e-4)

        self.save_obj = {
            "envs", "raw_env_desc_index", "tasks", "raw_task_desc_index",
            "skills", "env_task_skill_index", "data_len", "hidden_dim",
            "skill_dim", "skill_index_mapper", "max_env_delta",
            "max_env_delta_all", "max_task_delta"
        }

        self._read(config['eval'])

        self.skill_embeds = nn.Embedding(len(self.skills),
                                         self.skill_dim,
                                         device=DEVICE)
        normalize_embeds(self.skill_embeds)
        self.skill_optim = optim.AdamW(self.skill_embeds.parameters(), 3e-4)

        self.skill_index_mapper = {
            suid: i
            # for i, suid in enumerate(set(self.env_task_skill_index.values()))
            for i, suid in enumerate(self.skills.keys())
        }

    def _add_env(self, env_info: Dict[str, Any]) -> ENV_UUID:
        friction, desc, uniformness, slope = (env_info["friction"],
                                              env_info["desc"],
                                              env_info["uniformness"],
                                              env_info["slope"])

        uuid = gen_uuid()
        self.envs[uuid] = Env(uuid,
                              desc,
                              friction=friction,
                              uniformness=uniformness,
                              slope=slope,
                              skill_uuids=[])

        return uuid

    def _add_task(self, task_info: Dict[str, Any]) -> TASK_UUID:
        body_move, desc = (
            task_info["body_move"],
            task_info["desc"],
        )

        # assert sum(reward_weight) == 1.

        uuid = gen_uuid()
        self.tasks[uuid] = Task(uuid,
                                desc,
                                body_move=body_move,
                                skill_uuids=[])

        return uuid

    def _add_skill(self, skill_info: Dict[str, Any]) -> SKILL_UUID:
        env_uuids, task_uuids, model_weight, env_desc, task_desc, env_id = (
            skill_info["env_uuids"],
            skill_info["task_uuids"],
            skill_info["model_weight"],
            skill_info["env_desc"],
            skill_info["task_desc"],
            skill_info['env_id'],
        )

        skill_uuid = gen_uuid()

        skill = Skill(
            skill_uuid,
            f"{task_desc}_{env_desc}",
            model_weight,
            env_uuids=env_uuids,
            task_uuids=task_uuids,
            skill_folder=
            # f'{task_desc}/{f"ActorCritic_EnvID_{env_id}_SkillID_3" if ("roll" not in task_desc and "up" not in task_desc) else "ActorCritic" }'
            f'{task_desc}/{f"ActorCritic_EnvID_{env_id}_SkillID_3" if not is_simple_task(task_desc) else "ActorCritic" }'
        )
        self.skills[skill_uuid] = skill
        for task_uuid in task_uuids:
            task = self.tasks[task_uuid]
            task.skill_uuids.append(skill_uuid)

        for env_uuid in env_uuids:
            env = self.envs[env_uuid]
            env.skill_uuids.append(skill_uuid)

        # skill.model_weight = skill

        for env_uuid, task_uuid in product(env_uuids, task_uuids):
            self.env_task_skill_index[(env_uuid, task_uuid)] = skill_uuid

        return skill_uuid

    def _read(self, eval: bool):
        for env_name, props in tqdm(ENV_DESC.items(), desc="添加环境"):
            (_friction, _uniformness, _slope) = props

            friction = lambda idx: _friction if not isinstance(
                _friction, tuple) else np.random.uniform(low=_friction[0],
                                                         high=_friction[1])
            uniformness = lambda idx: _uniformness if not isinstance(
                _uniformness, tuple) else np.random.uniform(
                    low=_uniformness[0], high=_uniformness[1])

            slope = lambda idx: _slope if not isinstance(
                _slope, tuple) else np.random.uniform(low=_slope[0],
                                                      high=_slope[1])

            for idx in range(ENV_SAMPLE_NUM):
                f = friction(idx)
                u = uniformness(idx)
                s = slope(idx)

                env_uuid = self._add_env(
                    dict(
                        desc=env_name,
                        friction=f,
                        uniformness=u,
                        slope=s,
                    ))
                if env_name not in self.raw_env_desc_index:
                    self.raw_env_desc_index[env_name] = [env_uuid]
                else:
                    self.raw_env_desc_index[env_name].append(env_uuid)

        self.env_max = torch.as_tensor(np.vstack(
            [e.details for e in self.envs.values()]),
                                       dtype=torch.float32,
                                       device=DEVICE).max(dim=0).values + 1e-6
        self.env_min = torch.as_tensor(np.vstack(
            [e.details for e in self.envs.values()]),
                                       dtype=torch.float32,
                                       device=DEVICE).min(dim=0).values - 1e-6

        if not eval:
            _e = np.vstack([e.details for e in self.envs.values()])
            e1 = np.repeat(_e, len(self.envs), axis=0)
            e2 = np.vstack((_e, ) * len(self.envs))

            assert len(e1.shape) == 2 == len(e2.shape)
            self.max_env_delta_all = np.max(np.linalg.norm(self.norm_env(e1) -
                                                           self.norm_env(e2),
                                                           axis=-1),
                                            axis=0)
            print(f"max_env_delta_all的值是：{self.max_env_delta_all}")

            self.max_env_delta = np.max(np.linalg.norm(
                self.norm_env(e1)[:, :-1] - self.norm_env(e2)[:, :-1],
                axis=-1),
                                        axis=0)
            print(f"max_env_delta的值是：{self.max_env_delta}")
            del _e, e1, e2
        print(f"共计添加了{len(self.envs)}个环境")

        for task_name, sequence in tqdm(TASK_DESC.items(), desc="添加任务"):

            for rollout_idx in np.random.choice(TOTAL_ROLLOUT,
                                                size=ROLLOUT_NUM,
                                                replace=False).tolist():

                seq = sequence[rollout_idx, SEQUENCE_INDEX]
                assert seq.shape == (SEQUENCE_LEN, RAW_TASK_DIM)
                task_uuid = self._add_task(
                    dict(
                        body_move=process(seq),
                        desc=task_name,
                    ))
                if task_name not in self.raw_task_desc_index:
                    self.raw_task_desc_index[task_name] = [task_uuid]
                else:
                    self.raw_task_desc_index[task_name].append(task_uuid)

        self.task_max = torch.as_tensor(np.vstack(
            [t.details for t in self.tasks.values()]),
                                        dtype=torch.float32,
                                        device=DEVICE).max(dim=0).values + 1e-6

        self.task_min = torch.as_tensor(np.vstack(
            [t.details for t in self.tasks.values()]),
                                        dtype=torch.float32,
                                        device=DEVICE).min(dim=0).values - 1e-6

        if not eval:
            _t = np.vstack([t.details for t in self.tasks.values()])
            t1 = np.repeat(_t, len(self.tasks), axis=0)
            t2 = np.vstack((_t, ) * len(self.tasks))
            assert len(t1.shape) == 2 == len(t2.shape)
            self.max_task_delta = np.max(np.linalg.norm(self.norm_task(t1) -
                                                        self.norm_task(t2),
                                                        axis=-1),
                                         axis=0)
            print(f"max_task_delta的值是：{self.max_task_delta}")
            del _t, t1, t2
        print(f"共计添加了{len(self.tasks)}个任务")

        for (env_desc, _), (task_desc, _) in product(ENV_DESC.items(),
                                                     TASK_DESC.items()):
            env_uuids = self.raw_env_desc_index[env_desc]
            task_uuids = self.raw_task_desc_index[task_desc]
            self._add_skill(
                dict(
                    env_uuids=env_uuids,
                    task_uuids=task_uuids,
                    model_weight=np.random.random((32, 32, 16)),
                    env_desc=env_desc,
                    task_desc=task_desc,
                    env_id=env_mapper[env_desc],
                ))
        print("技能添加完毕")

    def norm_env(self, envs: np.ndarray):
        return (envs - self.env_min.detach().cpu().numpy()) * 2 / (
            self.env_max.detach().cpu().numpy() -
            self.env_min.detach().cpu().numpy()) - 1

    def norm_task(self, tasks: np.ndarray):
        # return (tasks - self.task_min.detach().cpu().numpy()) * 2 / (
        #     self.task_max.detach().cpu().numpy() -
        #     self.task_min.detach().cpu().numpy()) - 1
        return tasks

    def get_statistics(self):
        return {
            "num_of_envs": len(self.envs),
            "desc_of_envs": [env.desc for env in self.envs.values()],
            "num_of_tasks": len(self.tasks),
            "desc_of_tasks": [t.desc for t in self.tasks.values()],
            "num_of_skills": len(self.env_task_skill_index),
        }

    def save(self, counter: int, folder: Optional[str] = None):
        if folder is None:
            folder = f'{self.log_dir}/save/{counter}'
        path = folder
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.env_encoder.state_dict(), f"{path}/env_encoder.pt")
        torch.save(self.env_max, f"{path}/env_max.pt")
        torch.save(self.env_min, f'{path}/env_min.pt')
        assert self.env_max.shape == self.env_min.shape == (ENV_DIM, )
        torch.save(self.task_max, f'{path}/task_max.pt')
        torch.save(self.task_min, f'{path}/task_min.pt')
        assert self.task_max.shape == self.task_min.shape == (SEQUENCE_LEN *
                                                              TASK_DIM, )
        torch.save(self.task_encoder.state_dict(), f"{path}/task_encoder.pt")
        torch.save(self.skill_embeds.state_dict(), f"{path}/skill_embeds.pt")

        for v in self.save_obj:
            self._pickle_save(f"{path}/{v}.pkl", getattr(self, v))
        print(f"skill_graph保存成功，目录为：{path}")

    def load(self, folder):
        print(f"skill_graph 从文件{folder}中load")
        for v in self.save_obj:
            setattr(self, v, self._pickle_load(f"{folder}/{v}.pkl"))

        self.env_encoder.load_state_dict(
            torch.load(f"{folder}/env_encoder.pt"))
        self.task_encoder.load_state_dict(
            torch.load(f"{folder}/task_encoder.pt"))
        self.skill_embeds.load_state_dict(
            torch.load(f"{folder}/skill_embeds.pt"))
        self.env_max = torch.load(f"{folder}/env_max.pt")
        self.env_min = torch.load(f"{folder}/env_min.pt")
        assert self.env_max.shape == self.env_min.shape == (ENV_DIM, )
        self.task_max = torch.load(f"{folder}/task_max.pt")
        self.task_min = torch.load(f"{folder}/task_min.pt")
        assert self.task_max.shape == self.task_min.shape == (SEQUENCE_LEN *
                                                              TASK_DIM, )
        self.kgc_trained = True
        print("skill_graph读取成功")

    def _pickle_save(self, path, obj):
        with open(path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _pickle_load(self, path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
