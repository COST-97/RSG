import random
from typing import (Any, Callable, Dict, Iterator, List, Literal, Optional,
                    OrderedDict, Set, Tuple, Union, cast)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import torch
from .base import SkillGraphBase, NNSkills, Env, Task, Skill
from .uid import SKILL_UUID
from neo4j import GraphDatabase
from torch import nn, optim
from pathlib import Path
from .kge import transH
from .vis import UMAP, TriMAP, TSNE, PacMAP
import colorcet as cc
from .common import DEVICE, SCORE_FACTOR
from tqdm import tqdm
from itertools import product
from .dataset import Dataset
from .utils.net import normalize_embeds, orth_to
from .tasks import SEQUENCE_LEN, RAW_TASK_DIM, process, TASK_DIM, CN_DESC as TASK_CN_DESC
from .envs import ENV_DIM, CN_DESC as ENV_CN_DESC
from os import path
import torch.nn.functional as F


class SkillGraph(SkillGraphBase):

    def __init__(self, config: Dict[str, Any]):
        super(SkillGraph, self).__init__(config)
        self.r_w_s = nn.Embedding(2, self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_w_s)
        self.r_w_optim = optim.AdamW(self.r_w_s.parameters(), 3e-4)

        self.r_d_s = nn.Embedding(2, self.skill_dim).to(DEVICE)
        normalize_embeds(self.r_d_s)
        with torch.no_grad():
            for i in range(self.r_d_s.weight.size(0)):
                self.r_d_s.weight[i] = orth_to(self.r_w_s.weight[i],
                                               self.r_d_s.weight[i])
            # assert torch.all(
            #     (self.r_w_s.weight * self.r_d_s.weight).sum(-1) < 1e-4)

        self.r_d_optim = optim.AdamW(self.r_d_s.parameters(), 3e-4)

        self.env_tensor_index = torch.tensor(0,
                                             dtype=torch.long,
                                             device=DEVICE)
        self.task_tensor_index = torch.tensor(1,
                                              dtype=torch.long,
                                              device=DEVICE)

    def calc_env_margins(self):
        assert not hasattr(self, "env_margins")
        env_margins = {}

        # _e = np.vstack([e.details for e in self.envs.values()])
        # e1 = np.repeat(_e, len(self.envs), axis=0)
        # e2 = np.vstack((_e, ) * len(self.envs))

        # assert len(e1.shape) == 2 == len(e2.shape)
        # deltas = np.linalg.norm(self.norm_env(e1) - self.norm_env(e2), axis=1)
        # print(f"max_env_delta的值是：{self.max_env_delta}")
        # del _e, e1, e2
        for e1, e2 in tqdm(product(self.envs.values(), self.envs.values()),
                           desc="calc env margins",
                           total=len(self.envs)**2):
            e1d = e1.details
            e2d = e2.details
            assert e1d.shape == (ENV_DIM, )

            slope_dist = np.abs(
                self.norm_env(e1d)[-1] - self.norm_env(e2d)[-1]) / 2

            other_dist = np.linalg.norm(
                self.norm_env(e1d)[:-1] - self.norm_env(e2d)[:-1],
                axis=0) / self.max_env_delta
            margin = max(slope_dist, other_dist)
            assert 0 <= margin <= 1

            if e1.uuid in env_margins:
                env_margins[e1.uuid][e2.uuid] = margin
            else:
                env_margins[e1.uuid] = {e2.uuid: margin}
        self.env_margins = env_margins

    def calc_task_margins(self):
        assert not hasattr(self, "task_margins")
        task_margins = {}

        for t1, t2 in tqdm(product(self.tasks.values(), self.tasks.values()),
                           desc="calc task margins",
                           total=len(self.tasks)**2):
            t1d = self.norm_task(t1.details).reshape((SEQUENCE_LEN, TASK_DIM))
            t2d = self.norm_task(t2.details).reshape((SEQUENCE_LEN, TASK_DIM))
            assert t1d.shape == (SEQUENCE_LEN, TASK_DIM) == t2d.shape

            # yaw 方向不同
            yaw_diff = np.sum(np.abs(t1d[:, -3] - t2d[:, -3])) / SEQUENCE_LEN

            # 速度方向不同
            cos_diff = np.sum(
                -np.sum(t1d[:, :3] * t2d[:, :3], axis=1) / 2 + 1 / 2,
                axis=0) / SEQUENCE_LEN
            # if t1d[-3] != t2d[-3]:
            #     margin = 1.
            # else:
            # margin是cosine相似度，如果内积为1，则margin为0；如果内积为-1，则margin为1
            # margin = -np.dot(t1d[:3], t2d[:3]) / 2 + 1 / 2
            margin = max(yaw_diff, cos_diff)

            if t1.uuid in task_margins:
                task_margins[t1.uuid][t2.uuid] = margin
            else:
                task_margins[t1.uuid] = {t2.uuid: margin}
        self.task_margins = task_margins

    def train_kgc(self, config: Dict[str, Any]):
        self.calc_env_margins()
        self.calc_task_margins()
        batch_size, train_iters = (
            config["batch_size"],
            config["train_iters"],
        )

        self.optims = [
            self.env_optim, self.task_optim, self.skill_optim, self.r_w_optim,
            self.r_d_optim
        ]

        def assemble_env_tensor(e: Env):
            return torch.as_tensor(self.norm_env(e.details),
                                   dtype=torch.float32,
                                   device=DEVICE)

        def assemble_task_tensor(t: Task):
            return torch.as_tensor(self.norm_task(t.details),
                                   dtype=torch.float32,
                                   device=DEVICE)

        env_tensors = {
            euid: assemble_env_tensor(env)
            for euid, env in self.envs.items()
        }
        task_tensors = {
            tuid: assemble_task_tensor(task)
            for tuid, task in self.tasks.items()
        }

        def assemble_dataset():
            p_e_d, p_t_d = Dataset(), Dataset()
            positive_envs = [(env_tensors[env.uuid], self.env_tensor_index,
                              self.env_tensor_index,
                              torch.tensor(self.skill_index_mapper[sid],
                                           dtype=torch.long,
                                           device=DEVICE), (env, sid))
                             for env in self.envs.values()
                             for sid in env.skill_uuids]

            p_e_d.add((torch.stack([e for (e, _, _, _, _) in positive_envs]),
                       torch.stack([w for (_, w, _, _, _) in positive_envs]),
                       torch.stack([d for (_, _, d, _, _) in positive_envs]),
                       torch.stack([s for (_, _, _, s, _) in positive_envs]), [
                           dict(kind='e_r_s', env=i[0], sid=i[1])
                           for (_, _, _, _, i) in positive_envs
                       ]))

            positive_tasks = [(task_tensors[task.uuid], self.task_tensor_index,
                               self.task_tensor_index,
                               torch.tensor(self.skill_index_mapper[sid],
                                            dtype=torch.long,
                                            device=DEVICE), (task, sid))
                              for task in self.tasks.values()
                              for sid in task.skill_uuids]

            p_t_d.add(
                (torch.stack([t for (t, _, _, _, _) in positive_tasks]),
                 torch.stack([w for (_, w, _, _, _) in positive_tasks]),
                 torch.stack([d for (_, _, d, _, _) in positive_tasks]),
                 torch.stack([s for (_, _, _, s, _) in positive_tasks]), [
                     dict(kind='t_r_s', task=i[0], sid=i[1])
                     for (_, _, _, _, i) in positive_tasks
                 ]))

            assert p_t_d.len() + p_e_d.len() == len(self.envs) * len(
                self.raw_task_desc_index) + len(self.tasks) * len(
                    self.raw_env_desc_index)

            n_e_wr_s, n_t_wr_s, n_e_r_e, n_t_r_t = Dataset(), Dataset(
            ), Dataset(), Dataset()

            e_wr_s = [(env_tensors[env.uuid], self.task_tensor_index,
                       self.task_tensor_index,
                       torch.tensor(self.skill_index_mapper[sid],
                                    dtype=torch.long,
                                    device=DEVICE))
                      for env in self.envs.values() for sid in env.skill_uuids]

            n_e_wr_s.add((torch.stack([e for (e, _, _, _) in e_wr_s]),
                          torch.stack([w for (_, w, _, _) in e_wr_s]),
                          torch.stack([d for (_, _, d, _) in e_wr_s]),
                          torch.stack([s for (_, _, _, s) in e_wr_s]),
                          [dict(kind='e_wr_s') for _ in range(len(e_wr_s))]))

            t_wr_s = [(task_tensors[task.uuid], self.env_tensor_index,
                       self.env_tensor_index,
                       torch.tensor(self.skill_index_mapper[sid],
                                    dtype=torch.long,
                                    device=DEVICE))
                      for task in self.tasks.values()
                      for sid in task.skill_uuids]

            n_t_wr_s.add((torch.stack([e for (e, _, _, _) in t_wr_s]),
                          torch.stack([w for (_, w, _, _) in t_wr_s]),
                          torch.stack([d for (_, _, d, _) in t_wr_s]),
                          torch.stack([s for (_, _, _, s) in t_wr_s]),
                          [dict(kind='t_wr_s') for _ in range(len(t_wr_s))]))

            e_r_e = [(env_tensors[env1.uuid], self.env_tensor_index,
                      self.env_tensor_index, env_tensors[env2.uuid])
                     for env1, env2 in tqdm(product(
                         random.choices(list(self.envs.values()),
                                        k=int(len(self.envs.values()) / 10)),
                         random.choices(list(self.envs.values()),
                                        k=int(len(self.envs.values()) / 10))),
                                            desc="ere1")]

            e_r_e.extend([
                (env_tensors[env1.uuid], self.task_tensor_index,
                 self.task_tensor_index, env_tensors[env2.uuid])
                for env1, env2 in tqdm(product(
                    random.choices(list(self.envs.values()),
                                   k=int(len(self.envs.values()) / 10)),
                    random.choices(list(self.envs.values()),
                                   k=int(len(self.envs.values()) / 10))),
                                       desc="ere1")
            ])

            n_e_r_e.add((torch.stack([e for (e, _, _, _) in e_r_e]),
                         torch.stack([w for (_, w, _, _) in e_r_e]),
                         torch.stack([d for (_, _, d, _) in e_r_e]),
                         torch.stack([e for (_, _, _, e) in e_r_e]),
                         [dict(kind='e_r_e') for _ in range(len(e_r_e))]))

            t_r_t = [(task_tensors[task1.uuid], self.env_tensor_index,
                      self.env_tensor_index, task_tensors[task2.uuid])
                     for task1, task2 in tqdm(product(
                         random.choices(list(self.tasks.values()),
                                        k=int(len(self.tasks.values()) / 20)),
                         random.choices(list(self.tasks.values()),
                                        k=int(len(self.tasks.values()) / 20))),
                                              desc="trt1")]
            t_r_t.extend([
                (task_tensors[task1.uuid], self.task_tensor_index,
                 self.task_tensor_index, task_tensors[task2.uuid])
                for task1, task2 in tqdm(product(
                    random.choices(list(self.tasks.values()),
                                   k=int(len(self.tasks.values()) / 20)),
                    random.choices(list(self.tasks.values()),
                                   k=int(len(self.tasks.values()) / 20))),
                                         desc="trt2")
            ])
            n_t_r_t.add((torch.stack([t for (t, _, _, _) in t_r_t]),
                         torch.stack([w for (_, w, _, _) in t_r_t]),
                         torch.stack([d for (_, _, d, _) in t_r_t]),
                         torch.stack([t for (_, _, _, t) in t_r_t]),
                         [dict(kind='t_r_t') for _ in range(len(t_r_t))]))

            return (p_e_d, p_t_d), (n_e_wr_s, n_t_wr_s, n_e_r_e, n_t_r_t)

        # rank_loss_fn = nn.MarginRankingLoss(margin=0.5)
        rank_loss_fn = nn.ReLU()
        score_loss_fn = nn.MSELoss()
        entity_loss_fn = nn.ReLU()
        orth_loss_fn = nn.ReLU()
        orth_epslion = 1e-4

        (p_e_d, p_t_d), (n_e_wr_s, n_t_wr_s, n_e_r_e,
                         n_t_r_t) = assemble_dataset()
        # (p_e_d, v_p_e_d) = p_e_d.split((0.9, 0.1))
        # (p_t_d, v_p_t_d) = p_t_d.split((0.9, 0.1))
        # (n_e_wr_s, v_n_e_wr_s) = n_e_wr_s.split((0.9, 0.1))
        # (n_t_wr_s, v_n_t_wr_s) = n_t_wr_s.split((0.9, 0.1))
        # (n_e_r_e, v_n_e_r_e) = n_e_r_e.split((0.9, 0.1))
        # (n_t_r_t, v_n_t_r_t) = n_t_r_t.split((0.9, 0.1))

        assert train_iters % 1000 == 0
        CORRECT_SAMPLE_MULTIPLY = 4
        cnt = 0
        for _ in tqdm(range(100)):
            for _ in range(10):
                for _ in range(int(train_iters / 100 / 10)):
                    self.env_encoder.train()
                    self.task_encoder.train()
                    p_e_r_s, p_t_r_s = p_e_d.sample(
                        batch_size, CORRECT_SAMPLE_MULTIPLY), p_t_d.sample(
                            batch_size, CORRECT_SAMPLE_MULTIPLY)

                    correct_scores = torch.concatenate(
                        (transH(
                            self.env_encoder(
                                p_e_r_s[0][::CORRECT_SAMPLE_MULTIPLY]),
                            self.r_w_s(p_e_r_s[1][::CORRECT_SAMPLE_MULTIPLY]),
                            self.r_d_s(p_e_r_s[2][::CORRECT_SAMPLE_MULTIPLY]),
                            self.skill_embeds(
                                p_e_r_s[3][::CORRECT_SAMPLE_MULTIPLY])),
                         transH(
                             self.task_encoder(
                                 p_t_r_s[0][::CORRECT_SAMPLE_MULTIPLY]),
                             self.r_w_s(p_t_r_s[1][::CORRECT_SAMPLE_MULTIPLY]),
                             self.r_d_s(p_t_r_s[2][::CORRECT_SAMPLE_MULTIPLY]),
                             self.skill_embeds(
                                 p_t_r_s[3][::CORRECT_SAMPLE_MULTIPLY]))))

                    _wrong_task_triplets, _wrong_env_triplets = [], []

                    for i in p_e_r_s[-1]:
                        # i = cinfos[id]
                        assert i['kind'] == 'e_r_s'
                        o_env = cast(Env, i['env'])
                        skill = cast(Skill, self.skills[i['sid']])

                        assert all(
                            map(lambda eid: eid in self.envs.keys(),
                                skill.env_uuids))

                        candidates = set(
                            self.envs.keys()).difference(*[skill.env_uuids])
                        n_env_id = random.choice(list(candidates))

                        _wrong_env_triplets.append(
                            (env_tensors[n_env_id], self.env_tensor_index,
                             self.env_tensor_index,
                             torch.tensor(self.skill_index_mapper[skill.uuid],
                                          dtype=torch.long,
                                          device=DEVICE),
                             self.env_margins[o_env.uuid][n_env_id]))

                    for i in p_t_r_s[-1]:
                        assert i['kind'] == 't_r_s'
                        o_task = cast(Task, i['task'])
                        skill = cast(Skill, self.skills[i['sid']])

                        assert all(
                            map(lambda tid: tid in self.tasks.keys(),
                                skill.task_uuids))

                        candidates = set(
                            self.tasks.keys()).difference(*[skill.task_uuids])
                        n_task_id = random.choice(list(candidates))

                        _wrong_task_triplets.append(
                            (task_tensors[n_task_id], self.task_tensor_index,
                             self.task_tensor_index,
                             torch.tensor(self.skill_index_mapper[skill.uuid],
                                          dtype=torch.long,
                                          device=DEVICE),
                             self.task_margins[o_task.uuid][n_task_id]))

                    wrong_env_triplets = [
                        torch.stack(
                            [t for (t, _, _, _, _) in _wrong_env_triplets]),
                        torch.stack(
                            [w for (_, w, _, _, _) in _wrong_env_triplets]),
                        torch.stack(
                            [d for (_, _, d, _, _) in _wrong_env_triplets]),
                        torch.stack(
                            [s for (_, _, _, s, _) in _wrong_env_triplets]),
                        torch.tensor(
                            [m for (_, _, _, _, m) in _wrong_env_triplets],
                            dtype=torch.float32,
                            device=DEVICE)
                    ]
                    wrong_task_triplets = [
                        torch.stack(
                            [t for (t, _, _, _, _) in _wrong_task_triplets]),
                        torch.stack(
                            [w for (_, w, _, _, _) in _wrong_task_triplets]),
                        torch.stack(
                            [d for (_, _, d, _, _) in _wrong_task_triplets]),
                        torch.stack(
                            [s for (_, _, _, s, _) in _wrong_task_triplets]),
                        torch.tensor(
                            [m for (_, _, _, _, m) in _wrong_task_triplets],
                            dtype=torch.float32,
                            device=DEVICE)
                    ]

                    (weh, wew, wed, wet, wem) = wrong_env_triplets
                    (wth, wtw, wtd, wtt, wtm) = wrong_task_triplets
                    wrong_scores = torch.concatenate(
                        (transH(self.env_encoder(weh), self.r_w_s(wew),
                                self.r_d_s(wed), self.skill_embeds(wet)),
                         transH(self.task_encoder(wth), self.r_w_s(wtw),
                                self.r_d_s(wtd), self.skill_embeds(wtt))))
                    margins = torch.concat((wem, wtm))
                    ranking_loss = rank_loss_fn(
                        -(torch.ones_like(wrong_scores, device=DEVICE) -
                          wrong_scores - margins.detach())).mean()

                    newrs, ntwrs, nere, ntrt = n_e_wr_s.sample(
                        batch_size), n_t_wr_s.sample(
                            batch_size), n_e_r_e.sample(
                                batch_size), n_t_r_t.sample(batch_size)

                    negative_scores = torch.concatenate(
                        (transH(self.env_encoder(newrs[0]),
                                self.r_w_s(newrs[1]), self.r_d_s(newrs[2]),
                                self.skill_embeds(newrs[3])),
                         transH(self.task_encoder(ntwrs[0]),
                                self.r_w_s(ntwrs[1]), self.r_d_s(ntwrs[2]),
                                self.skill_embeds(ntwrs[3])),
                         transH(self.env_encoder(nere[0]), self.r_w_s(nere[1]),
                                self.r_d_s(nere[2]),
                                self.env_encoder(nere[3])),
                         transH(self.task_encoder(ntrt[0]),
                                self.r_w_s(ntrt[1]), self.r_d_s(ntrt[2]),
                                self.task_encoder(ntrt[3]))))

                    score_loss = score_loss_fn(
                        negative_scores,
                        torch.zeros_like(negative_scores)) + score_loss_fn(
                            correct_scores, torch.ones_like(
                                correct_scores))  # + score_loss_fn(
                    # wrong_scores, torch.zeros_like(wrong_scores))

                    env_entity_loss = entity_loss_fn(
                        torch.linalg.norm(self.env_encoder(
                            torch.concatenate([
                                p_e_r_s[0], weh, newrs[0], nere[3], nere[3]
                            ])),
                                          dim=-1) - 1).mean()

                    task_entity_loss = entity_loss_fn(
                        torch.linalg.norm(self.task_encoder(
                            torch.concatenate([
                                p_t_r_s[0], wth, ntwrs[0], ntrt[0], ntrt[3]
                            ])),
                                          dim=-1) - 1).mean()

                    skill_entity_loss = entity_loss_fn(
                        torch.linalg.norm(self.skill_embeds(
                            torch.arange(len(self.skills),
                                         device=DEVICE,
                                         dtype=torch.long)),
                                          dim=-1) - 1).mean()

                    # with torch.no_grad():
                    _w = self.r_w_s.weight
                    _w = _w / torch.linalg.norm(_w, dim=-1, keepdim=True)

                    orths = (_w *
                             self.r_d_s.weight).sum(-1) / torch.linalg.norm(
                                 self.r_d_s.weight, dim=-1)

                    assert orths.shape == (self.r_w_s.weight.size(0), )
                    relation_orth_loss = orth_loss_fn(orths**2 -
                                                      orth_epslion**2).mean()

                    [o.zero_grad() for o in self.optims]
                    (ranking_loss + 1.5 * score_loss + 0.2 *
                     (task_entity_loss + env_entity_loss + skill_entity_loss +
                      relation_orth_loss)).backward()
                    [o.step() for o in self.optims]

                self.env_encoder.eval()
                self.task_encoder.eval()
                self.reporter.add_scalars(
                    dict(
                        ranking_loss=ranking_loss.item(),
                        score_loss=score_loss.item(),
                        # correct_scores=correct_scores.mean().item(),
                        # wrong_scores=wrong_scores.mean().item(),
                        # negative_scores=negative_scores.mean().item(),
                        task_entity_loss=task_entity_loss.item(),
                        relation_orth_loss=relation_orth_loss.item(),
                        skill_entity_loss=skill_entity_loss.item(),
                        env_entity_loss=env_entity_loss.item()),
                    'train')
                self.reporter.add_distributions(
                    dict(
                        correct_scores=correct_scores,
                        negative_scores=negative_scores,
                        wrong_scores=wrong_scores,
                    ), 'train')

                # v_p_e_r_s, v_p_t_r_s = v_p_e_d.sample(
                #     v_p_e_d.len(), all=True), v_p_t_d.sample(v_p_t_d.len(),
                #                                              all=True)

                # correct_scores = torch.concatenate(
                #     (transH(self.env_encoder(v_p_e_r_s[0]),
                #             self.r_w_s(v_p_e_r_s[1]), self.r_d_s(v_p_e_r_s[2]),
                #             self.skill_embeds(v_p_e_r_s[3])),
                #      transH(self.task_encoder(v_p_t_r_s[0]),
                #             self.r_w_s(v_p_t_r_s[1]), self.r_d_s(v_p_t_r_s[2]),
                #             self.skill_embeds(v_p_t_r_s[3]))))

                # v_newrs, v_ntwrs, v_nere, v_ntrt = v_n_e_wr_s.sample(
                #     v_n_e_wr_s.len(), all=True), v_n_t_wr_s.sample(
                #         v_n_t_wr_s.len(), all=True), v_n_e_r_e.sample(
                #             v_n_e_r_e.len(),
                #             all=True), v_n_t_r_t.sample(v_n_t_r_t.len(),
                #                                         all=True)

                # negative_scores = torch.concatenate(
                #     (transH(self.env_encoder(v_newrs[0]),
                #             self.r_w_s(v_newrs[1]), self.r_d_s(v_newrs[2]),
                #             self.skill_embeds(v_newrs[3])),
                #      transH(self.task_encoder(v_ntwrs[0]),
                #             self.r_w_s(v_ntwrs[1]), self.r_d_s(v_ntwrs[2]),
                #             self.skill_embeds(v_ntwrs[3])),
                #      transH(self.env_encoder(v_nere[0]), self.r_w_s(v_nere[1]),
                #             self.r_d_s(v_nere[2]),
                #             self.env_encoder(v_nere[3])),
                #      transH(self.task_encoder(v_ntrt[0]),
                #             self.r_w_s(v_ntrt[1]), self.r_d_s(v_ntrt[2]),
                #             self.task_encoder(v_ntrt[3]))))

                # self.reporter.add_distributions(
                #     dict(
                #         correct_scores=correct_scores,
                #         negative_scores=negative_scores,
                #         # wrong_scores=wrong_scores,
                #     ),
                #     'test')

            self.save(cnt)
            cnt += 1
        self.kgc_trained = True
        self.env_encoder.eval()
        self.task_encoder.eval()

    def kgc(
        self,
        # k: int,
        env_property: np.ndarray,
        task_property: np.ndarray,
        merge_fn: Optional[Callable[[torch.Tensor, torch.Tensor],
                                    torch.Tensor]] = None,
        reverse=False,
    ):  # -> NNSkills:
        assert hasattr(self, "kgc_trained") and self.kgc_trained
        skill_reverse_mapper = {
            i: suid
            for (suid, i) in self.skill_index_mapper.items()
        }

        merge_fn = merge_fn or (lambda es, ts: es * ts)

        q_e = torch.as_tensor(self.norm_env(env_property),
                              dtype=torch.float32,
                              device=DEVICE)  # -
        q_t = torch.as_tensor(self.norm_task(task_property),
                              dtype=torch.float32,
                              device=DEVICE)  #-
        l = len(self.skills)
        with torch.no_grad():
            all_skills = self.skill_embeds(
                torch.arange(len(self.skills), dtype=torch.long,
                             device=DEVICE))
            es = self.env_encoder(q_e.unsqueeze(0)).repeat_interleave(l, dim=0)
            e_w_s = self.r_w_s(torch.tensor(
                0, dtype=torch.long,
                device=DEVICE)).unsqueeze(0).repeat_interleave(l, dim=0)
            e_d_s = self.r_d_s(torch.tensor(
                0, dtype=torch.long,
                device=DEVICE)).unsqueeze(0).repeat_interleave(l, dim=0)

            e_scores = transH(es, e_w_s, e_d_s, all_skills)
            assert e_scores.shape == (l, )

            ts = self.task_encoder(q_t.unsqueeze(0)).repeat_interleave(l,
                                                                       dim=0)
            t_w_s = self.r_w_s(torch.tensor(
                1, dtype=torch.long,
                device=DEVICE)).unsqueeze(0).repeat_interleave(l, dim=0)
            t_d_s = self.r_d_s(torch.tensor(
                1, dtype=torch.long,
                device=DEVICE)).unsqueeze(0).repeat_interleave(l, dim=0)

            t_scores = transH(ts, t_w_s, t_d_s, all_skills)
            assert t_scores.shape == (l, )

        final_scores = merge_fn(e_scores, t_scores)
        assert final_scores.shape == (l, )

        idxs = final_scores.argsort(descending=True)

        return [self.skills[skill_reverse_mapper[i.item()]] for i in idxs
                ], final_scores[idxs], (e_scores[idxs], t_scores[idxs])

    def draw_in_neo4j(self):
        sg = self

        def add_env(tx, env: Env):
            query = "CREATE (e: Env {uuid: $uuid, skill_uuids: $skill_uuids, label: $label, desc: $desc})"
            tx.run(
                query,
                uuid=env.uuid,
                skill_uuids=env.skill_uuids,
                label=ENV_CN_DESC[env.desc],
                desc=env.desc,
            )

        def add_task(tx, task: Task):
            query = "CREATE (t: Task {uuid: $uuid,  skill_uuids: $skill_uuids, label: $label, desc: $desc})"
            tx.run(
                query,
                uuid=task.uuid,
                skill_uuids=task.skill_uuids,
                label=TASK_CN_DESC[task.desc],
                desc=task.desc,
            )

        def add_skill(tx, skill: Skill):
            query = f"CREATE (s: Skill {{uuid: $skill_uuid, label: $label, env_uuids: $env_uuids, task_uuids: $task_uuids, desc: $desc}})"

            tx.run(
                query,
                skill_uuid=skill.uuid,
                env_uuids=skill.env_uuids,
                desc=skill.desc,
                task_uuids=skill.task_uuids,
                label=skill.label,
            )

        def add_env_relations(tx, **kwargs):
            query = f"MATCH (e: Env {{uuid: $env_uuid}}), (s: Skill {{uuid: $skill_uuid}}) CREATE (e) -[:R] -> (s)"

            tx.run(query, **kwargs)

        def add_task_relations(tx, **kwargs):
            query = f"MATCH (t: Task {{uuid: $task_uuid}}), (s: Skill {{uuid: $skill_uuid}}) CREATE (t) -[:R] -> (s)"

            tx.run(query, **kwargs)

        with GraphDatabase.driver("neo4j://localhost:7687",
                                  auth=("neo4j", "neo4j123")) as driver:
            with driver.session() as session:
                session.execute_write(
                    lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

                for skill in list(sg.skills.values())[:250]:
                    session.execute_write(add_skill, skill)

                for env in list(sg.envs.values())[::100]:
                    session.execute_write(add_env, env)
                for task in list(sg.tasks.values())[::100]:
                    session.execute_write(add_task, task)


                for skill in tqdm(list(sg.skills.values())[:250], desc="add relations"):
                    for env_uuid in skill.env_uuids:
                        session.execute_write(
                            add_env_relations,
                            skill_uuid=skill.uuid, env_uuid=env_uuid)
                    for task_uuid in skill.task_uuids:
                        session.execute_write(
                            add_task_relations,
                            skill_uuid=skill.uuid, task_uuid=task_uuid)


    def tsne(self,
             on: Union[Literal["env"], Literal["task"], Literal['skill']],
             latent=False):
        assert on in ["env", "task", "skill"]
        if on == 'skill':
            embeds = self.skill_embeds(
                torch.tensor([
                    self.skill_index_mapper[s.uuid]
                    for s in self.skills.values()
                ],
                             dtype=torch.long,
                             device=DEVICE)).numpy(force=True)
            task_desc, env_desc = zip(
                *[s.desc.rsplit('_', 1) for s in self.skills.values()])
        else:

            data = list(self.envs.values() if on ==
                        "env" else self.tasks.values())
            if on == 'task':
                desc = np.array(
                    ["_".join(e.desc.split('_')[:-1]) for e in data],
                    dtype=object)
            else:
                desc = np.array([e.desc for e in data], dtype=object)

            embeds = np.vstack([d.details for d in data])

            if latent:
                embeds = ((self.env_encoder
                           if on == "env" else self.task_encoder)(
                               torch.as_tensor(embeds,
                                               dtype=torch.float32,
                                               device=DEVICE)))
                w_r = self.r_w_s(self.env_tensor_index if on ==
                                 'env' else self.task_tensor_index)
                d_r = self.r_d_s(self.env_tensor_index if on ==
                                 'env' else self.task_tensor_index)
                assert embeds.shape == (len(data), self.skill_dim)
                assert w_r.shape == (self.skill_dim, )
                embeds = (embeds - (w_r * embeds).sum(-1, keepdim=True) * w_r +
                          d_r).numpy(force=True)
                # embeds = (embeds).numpy(force=True)

        for vis in [TSNE(), TriMAP(), PacMAP()]:

            points = vis.forward(embeds).tolist()
            if on == 'skill':
                points = [[points[ri][ci] for ci in range(len(points[ri]))] +
                          [task_desc[ri], env_desc[ri]]
                          for ri in range(len(points))]

            elif on == 'task':
                points = [[points[ri][ci]
                           for ci in range(len(points[ri]))] + [desc[ri]]
                          for ri in range(len(points))]
            else:
                points = [[points[ri][ci]
                           for ci in range(len(points[ri]))] + [desc[ri]]
                          for ri in range(len(points))]

            # plt.figure(figsize=(19.2, 10.9), dpi=200)
            # sns.set(rc={'figure.figsize': (11.7,8.27)})
            palette = sns.color_palette(
                cc.glasbey_dark,
                n_colors=np.unique(
                    np.sort(desc if on != 'skill' else task_desc)).shape[0])

            # p =

            # p = p.get_figure()
            # Path(f"./output/{'' if not latent else 'latent_'}imgs/{on}").mkdir(
            #     parents=True, exist_ok=True)
            # p.savefig(

            # )
            def _draw_img():
                plt.figure(figsize=(2.2, 2.2), dpi=600)
                # plt.xticks([])
                # plt.yticks([])
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
                plt.tick_params(
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    right=False,  # ticks along the top edge are off
                    labelleft=False)  # labels along the bottom edge are off

                img = sns.scatterplot(
                    data=pd.DataFrame(
                        points,
                        columns=["x", "y", "task"]
                        if on == 'task' else ["x", 'y', 'env'] if on == 'env'
                        else ['x', 'y', 'task_desc', 'env_desc']),
                    x="x",
                    y="y",
                    s=7 if on != 'skill' else 15,
                    palette=palette,
                    **(dict(hue="task") if on == 'task' else dict(
                        hue="env") if on == 'env' else dict(hue='task_desc',
                                                            style='env_desc')),
                )

                # plt.legend(
                #     loc='center left',
                #     # bbox_to_anchor=(1, 0.5),
                #     #    scatterpoints=2,
                #     # markerscale=1.5,
                #     fontsize='x-small')
                plt.legend('', frameon=False)

                return img

            self._save_img(
                _draw_img,
                f"./output/{'' if not latent else 'latent_'}tsne/{on}/{vis.__class__.__name__}"
            )

    def save(self, counter: int, folder: Optional[str] = None):
        if folder is None:
            folder = f'{self.log_dir}/save/{counter}'
        path = folder
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.r_w_s.state_dict(), f"{path}/r_w_s.pt")
        torch.save(self.r_d_s.state_dict(), f"{path}/r_d_s.pt")

        return super().save(counter, folder)

    def load(self, folder):
        self.r_w_s.load_state_dict(torch.load(f"{folder}/r_w_s.pt"))
        self.r_d_s.load_state_dict(torch.load(f"{folder}/r_d_s.pt"))

        rlt = super().load(folder)
        self.env_encoder.eval()
        self.task_encoder.eval()
        return rlt

    @torch.no_grad()
    def inspect(self, skill_desc: str, on: Union[Literal['env'],
                                                 Literal['task']]):
        skill = list(
            filter(lambda skill: skill.desc == skill_desc,
                   self.skills.values()))
        assert len(
            skill) == 1, f"cannot find exact skill in {skill_desc} on {on}"
        skill = skill[0]
        all_envs = list(self.envs.values())
        all_tasks = list(self.tasks.values())
        env_desc = skill_desc.rsplit('_')[-1]
        task_desc = '_'.join(skill_desc.rsplit('_')[:-2])

        if on == 'env':
            hs = self.env_encoder(
                torch.as_tensor(self.norm_env(
                    np.vstack([e.details for e in all_envs])),
                                dtype=torch.float32,
                                device=DEVICE))  # - self.env_min) * 2 /
            # (self.env_max - self.env_min) - 1)
            desc = [e.desc for e in all_envs]
            r_w_s = self.r_w_s(torch.tensor(0, dtype=torch.long,
                                            device=DEVICE))
            r_d_s = self.r_d_s(torch.tensor(0, dtype=torch.long,
                                            device=DEVICE))
            desc = np.array([e.desc for e in all_envs], dtype=object)
        else:
            hs = self.task_encoder(
                torch.as_tensor(self.norm_task(
                    np.vstack([t.details for t in all_tasks])),
                                device=DEVICE,
                                dtype=torch.float32))  # - self.task_min) * 2 /
            # (self.task_max - self.task_min) - 1)
            desc = [t.desc for t in all_tasks]
            r_w_s = self.r_w_s(torch.tensor(1, dtype=torch.long,
                                            device=DEVICE))
            r_d_s = self.r_d_s(torch.tensor(1, dtype=torch.long,
                                            device=DEVICE))
            desc = np.array(
                ["_".join(e.desc.split('_')[:-1]) for e in all_tasks],
                dtype=object)

        scores = transH(
            hs,
            r_w_s.unsqueeze(0).repeat_interleave(hs.size(0), dim=0),
            r_d_s.unsqueeze(0).repeat_interleave(hs.size(0), dim=0),
            self.skill_embeds(
                torch.tensor(self.skill_index_mapper[skill.uuid],
                             dtype=torch.long,
                             device=DEVICE)).unsqueeze(0).repeat_interleave(
                                 hs.size(0), dim=0))
        palette = sns.color_palette(cc.glasbey_dark,
                                    n_colors=np.unique(np.sort(desc)).shape[0])

        def _draw_img():
            nonlocal palette
            plt.figure(figsize=(2.0, 2.2), dpi=600)
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            plt.xlabel('score', fontsize=6)
            plt.xlim((0., 1.))
            plt.xticks(fontsize=5, rotation=0)
            # plt.tick_params(axis='x', which='major', labelsize=3)
            # plt.tick_params(axis='x', which='minor', labelsize=3)
            img = sns.stripplot(data=pd.DataFrame(list(
                zip(scores.tolist(), desc)),
                                                  columns=['scores', 'desc']),
                                x="scores",
                                y='desc',
                                s=1.5,
                                palette=palette,
                                hue="desc")
            plt.legend('', frameon=False)
            # plt.title(f"{on.capitalize()} Cross Scores of {'forward walking' if on == 'env' else 'Grassland'} \n on {env_desc if on == 'env' else task_desc} by SG", fontsize=5, y=-0.35)
            if on == 'env':
                italic_env = ' '.join(
                    map(lambda e: f'$\it{{{e}}}$', env_desc.split(' ')))
                plt.title(
                    f"{on.capitalize()} Cross Scores by RSG of \n {' '.join(task_desc.split('_'))} on {italic_env}",
                    fontsize=5,
                    y=-0.35)
            else:
                italic_task = ' '.join(
                    map(lambda t: f'$\it{{{t}}}$', task_desc.split('_')))
                plt.title(
                    f"{on.capitalize()} Cross Scores by RSG of \n {italic_task} on {env_desc}",
                    fontsize=5,
                    y=-0.35)
            # plt.yticks([])
            # for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(),
            #                                 palette):
            #     ticklabel.set_color(tickcolor)
            return img

        self._save_img(_draw_img,
                       f'./output/inspect/kgc/{on}/{skill_desc}_{on}.png')

    def inspect_knn(self,
                    skill_desc: str,
                    on: Union[Literal['env'], Literal['task']],
                    same_metric: bool = False):
        skill = list(
            filter(lambda skill: skill.desc == skill_desc,
                   self.skills.values()))
        assert len(skill) == 1
        skill = skill[0]
        envs = skill.env_uuids
        all_envs = list(self.envs.values())
        tasks = skill.task_uuids
        all_tasks = list(self.tasks.values())
        env_desc = skill_desc.rsplit('_')[-1]
        task_desc = '_'.join(skill_desc.rsplit('_')[:-2])

        if on == 'env':
            center_env = np.mean(
                self.norm_env(
                    np.vstack([self.envs[euid].details for euid in envs])),  #-
                axis=0)
            env_points = self.norm_env(np.vstack([e.details
                                                  for e in all_envs]))  # -

            if same_metric:
                center_env_slope = center_env[[-1]]
                assert center_env_slope.shape == (1, )

                center_env_other = center_env[:-1]
                assert center_env_other.shape == (2, )

                env_points_slope = env_points[:, [-1]]
                assert env_points_slope.shape == (len(all_envs), 1)

                env_points_other = env_points[:, :-1]
                assert env_points_other.shape == (len(all_envs), 2)

                slope_dist = np.abs(env_points_slope - center_env_slope) / 2
                other_dist = np.linalg.norm(
                    env_points_other - center_env_other, axis=1,
                    keepdims=True) / self.max_env_delta

                assert slope_dist.shape == other_dist.shape == (len(all_envs),
                                                                1)
                margins = np.max(np.concatenate((slope_dist, other_dist),
                                                axis=1),
                                 axis=1)
                assert margins.shape == (len(all_envs), )

                sims = np.exp(-SCORE_FACTOR * margins)

            else:
                sims = np.exp(
                    -SCORE_FACTOR *
                    np.linalg.norm(env_points - center_env, axis=-1) /
                    self.max_env_delta_all)
            desc = [e.desc for e in all_envs]
        else:
            center_task = np.mean(self.norm_task(
                np.vstack([self.tasks[tuid].details for tuid in tasks])),
                                  axis=0)

            task_points = self.norm_task(
                np.vstack([t.details for t in all_tasks]))

            if same_metric:
                _center_task = center_task.reshape((SEQUENCE_LEN, TASK_DIM))
                _task_points = task_points.reshape(
                    (len(all_tasks), SEQUENCE_LEN, TASK_DIM))

                yaw_diff = np.sum(
                    np.abs(_task_points[:, :, -3] - _center_task[:, -3]),
                    axis=1,
                    keepdims=True) / SEQUENCE_LEN

                cos_diff = np.sum(
                    -np.sum(_task_points[:, :, :3] * _center_task[:, :3],
                            axis=2) / 2 + 1 / 2,
                    axis=1,
                    keepdims=True) / SEQUENCE_LEN

                assert yaw_diff.shape == (len(all_tasks), 1) == cos_diff.shape

                margins = np.max(np.concatenate((cos_diff, yaw_diff), axis=1),
                                 axis=1)
                assert margins.shape == (len(all_tasks), )

                sims = np.exp(-SCORE_FACTOR * margins)

            else:
                sims = np.exp(
                    -SCORE_FACTOR *
                    np.linalg.norm(task_points - center_task, axis=-1) /
                    self.max_task_delta)
            desc = [t.desc for t in all_tasks]

        palette = sns.color_palette(cc.glasbey_dark,
                                    n_colors=np.unique(np.sort(desc)).shape[0])

        def _draw_img():
            nonlocal palette
            plt.figure(figsize=(2.0, 2.2), dpi=600)
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            plt.xticks(fontsize=5, rotation=0)
            plt.xlim((0., 1.))
            plt.xlabel('score', fontsize=6)
            # plt.tick_params(axis='x', which='major', labelsize=3)
            # plt.tick_params(axis='x', which='minor', labelsize=3)
            img = sns.stripplot(data=pd.DataFrame(list(zip(
                sims.tolist(), desc)),
                                                  columns=['scores', 'desc']),
                                x="scores",
                                y='desc',
                                palette=palette,
                                s=1.5,
                                hue="desc")
            plt.legend('', frameon=False)
            # plt.title(f"{on.capitalize()} Cross Scores of {'forward walking' if on == 'env' else 'Grassland'}", fontsize=6, y=-0.3)
            # plt.title(f"{on.capitalize()} Cross Scores of {'forward walking' if on == 'env' else 'Grassland'} \n on {env_desc if on == 'env' else task_desc} by KNN", fontsize=5, y=-0.35)
            if on == 'env':
                italic_env = ' '.join(
                    map(lambda e: f'$\it{{{e}}}$', env_desc.split(' ')))
                plt.title(
                    f"{on.capitalize()} Cross Scores by KNN of \n {' '.join(task_desc.split('_'))} on {italic_env}",
                    fontsize=5,
                    y=-0.35)
            else:
                italic_task = ' '.join(
                    map(lambda t: f'$\it{{{t}}}$', task_desc.split('_')))
                plt.title(
                    f"{on.capitalize()} Cross Scores by KNN of \n {italic_task} on {env_desc}",
                    fontsize=5,
                    y=-0.35)
            # ax.tick_params(axis='x', which='minor', labelsize=3)
            # plt.yticks([])
            # for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(),
            #                                 palette):
            #     ticklabel.set_color(tickcolor)
            return img

        self._save_img(
            _draw_img,
            f'./output/inspect/knn/{on}/{skill_desc}_{on}{"_same_metric" if same_metric else ""}.png'
        )

    def _save_img(self, draw_img: Callable[[], Any], dir: str):
        p = (draw_img()).get_figure()

        Path(path.dirname(dir)).mkdir(parents=True, exist_ok=True)

        p.savefig(dir, bbox_inches='tight')
        plt.close()
