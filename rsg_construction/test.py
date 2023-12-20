from common import seed
from skill_graph import SkillGraph
from skill_graph.envs import ENV_DESC
from skill_graph.tasks import TASK_DESC, process as process_task
import matplotlib.pyplot as plt
import numpy as np
import argparse
from body_move import (
    forward_right_final,
    backward_left_final,
    forward_left_final,
    backward_right_final,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="models")

args, _ = parser.parse_known_args()

TRAINED_FOLDER = args.model

if __name__ == "__main__":
    seed(0)

    sg = SkillGraph({"hidden_dim": 256, "skill_dim": 48, "name": "test", "eval": True})

    sg.load(TRAINED_FOLDER)
    sg.draw_in_neo4j()
    exit(0)
    # for i, s in enumerate(
    #         # forward_left_final, forward_right_final, backward_left_final,
    #         # backward_right_final
    #         [up_backward_left_final, up_forward_right_final]
    #         # up_backward_left
    #         # up_forward_right
    # ):
    #     (skills, scores, (e_scores, t_scores)) = sg.kgc(
    #         env_property=np.array([0.9, 0.0, 0.0]),
    #         task_property=process_task(s.numpy(force=True)).reshape((-1, )))
    #     # _skills =
    #     # task_desc =
    #     # print(
    #     #     f'{i}: {list(dict.fromkeys([s.desc[:s.desc.rfind("_")] for s in skills]))[:5]}'
    #     # )
    #     print(f'{i}: {[s.desc for s in skills[:7]]}')
    #     # plt.scatter(list(range(scores.size(0))), scores.tolist())
    #     # plt.show()
    #     ...
    # exit(0)
    # for i,s in enumerate([ up_backward_left_tmp]):
    #     (skills, scores,
    #      (e_scores, t_scores)) = sg.kgc(env_property=process_env(np.array([0.8, 0.0, 0.0])),
    #                                     task_property=process_task(s.numpy(force=True)).reshape(
    #                                         (-1, )))
    # print(f'{i}: {[s.desc for s in skills[:5]]}')
    # plt.scatter(list(range(scores.size(0))), scores.tolist())
    # plt.show()
    # ...
    # exit(0)

    # sg.tsne('task')
    # sg.tsne('env')
    # sg.tsne('task', True)
    # sg.tsne('env', True)
    # sg.tsne('skill', True)
    # exit(0)

    # env representation
    for env in ENV_DESC.keys():
        sg.inspect(f"forward_walking_oe_{env}", "env")
        sg.inspect_knn(f"forward_walking_oe_{env}", "env")
        # sg.inspect_knn(f'forward_walking_oe_{env}', 'env', True)
    # sg.inspect('forward_walking_0.5_0.25_0.25_Crushed Stone', 'env')
    # sg.inspect('forward_left_0.25_0.5_0.25_Sand', 'env')
    # sg.inspect('backward_left_walking_0.25_0.5_0.25_Grass and Sand', 'env')
    # sg.inspect('backward_left_walking_0.25_0.5_0.25_Marble Slope Uphill', 'env')
    # sg.inspect_knn('forward_walking_0.5_0.25_0.25_Crushed Stone', 'env')
    # sg.inspect_knn('forward_left_0.25_0.5_0.25_Sand', 'env')
    # sg.inspect_knn('backward_left_walking_0.25_0.5_0.25_Grass and Sand', 'env')
    # sg.inspect_knn('backward_left_walking_0.25_0.5_0.25_Marble Slope Uphill', 'env')

    # task representation
    for task in TASK_DESC.keys():
        sg.inspect(f"{task}_Grassland", "task")
        sg.inspect_knn(f"{task}_Grassland", "task")
        # sg.inspect_knn(f'{task}_Grassland', 'task', True)
        # sg.inspect_knn('forward_left_0.25_0.5_0.25_Sand', 'task')
        # sg.inspect_knn('sidestep_right_0.25_0.25_0.5_Grass and Mud', 'task')
        # sg.inspect_knn('spin_clockwise_0.25_0.25_0.5_Grass and Sand', 'task')
        # sg.inspect_knn('gallop_0.25_0.5_0.25_Indoor Floor', 'task')
        # sg.inspect_knn('backward_left_walking_0.25_0.5_0.25_Indoor Floor', 'task')
