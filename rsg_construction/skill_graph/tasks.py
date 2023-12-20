import numpy as np

TASKS = [
    'sidestep_right_oe_R_23.1',
    'forward_mass_oe_R_29.2',
    'gallop_oe_R_-0.8',
    'sidestep_left_oe_R_24.2',
    'backward_left_walking_oe_R_23.5',
    'up_oe_R_89.4',
    'spin_clockwise_oe_R_17.2',
    'up_left_oe_R_152.2',
    'up_backward_oe_R_160.6',
    'forward_right_oe_R_20.6',
    'up_forward_oe_R_148.4',
    'backward_walking_oe_R_27.2',
    'forward_walking_fast_oe_R_25.0',
    # 'up1_oe_R_108.7',
    'up_right_oe_R_154.6',
    # 'gallop_fast_oe_R_-2.7',
    'spin_counterclockwise_oe_R_3.1',
    'forward_walking_oe_R_23.5',
    'backward_right_walking_oe_R_27.3',
    'forward_noise_oe_R_29.2',
    'forward_left_oe_R_12.9',  #"roll_oe_R_165.3",
    "standup_oe_R_40.5",
    'backward_slow_oe_R_12.2',
    'forward_slow_oe_R_11.8',
    'sidestep_left_slow_oe_R_11.8',
    "sidestep_right_slow_oe_R_11.9",
    "spin_clockwise_slow_oe_R_12.3",
    "spin_counterclockwise_slow_oe_R_12.4"
]

CN_DESC = {
    'sidestep_right_oe': "右平移",
    'forward_mass_oe': "前进1",
    'gallop_oe': "奔跑",
    'sidestep_left_oe': "左平移",
    'backward_left_walking_oe': "左后行走",
    'up_oe': "原地跳",
    'spin_clockwise_oe': "顺时针转圈",
    'up_left_oe': "左跳",
    'up_backward_oe': "后跳",
    'forward_right_oe': "右前行走",
    'up_forward_oe':"前跳",
    'backward_walking_oe':"向后走",
    'forward_walking_fast_oe': "快速前进",
    # 'up1_oe_R_108.7',
    'up_right_oe': "右跳",
    # 'gallop_fast_oe_R_-2.7',
    'spin_counterclockwise_oe': "逆时针转圈",
    'forward_walking_oe': "前进2",
    'backward_right_walking_oe': "右后行走",
    'forward_noise_oe': "前进3",
    'forward_left_oe': "左前行走",  #"roll_oe_R_165.3",
    "standup_oe": "原地站立",
    'backward_slow_oe': "后慢走",
    'forward_slow_oe': "前慢走",
    'sidestep_left_slow_oe': "左侧慢平移",
    "sidestep_right_slow_oe": "右侧慢平移",
    "spin_clockwise_slow_oe": "顺时针缓慢转圈",
    "spin_counterclockwise_slow_oe": "逆时针缓慢转圈"

}


def is_simple_task(task_name: str) -> bool:
    return "up" in task_name or "roll" in task_name or "slow" in task_name


SEQUENCE_LEN = 11
TASK_DESC = {
    "_".join(task.split('_')[:-2]): np.load(
        f'./new_data/task_desc/{task}/{"ActorCritic_EnvID_0_SkillID_3.npy" if not is_simple_task(task) else "ActorCritic.npy" }'
    )[:, :(3 * SEQUENCE_LEN), [0, 1, 2, 5]]
    for task in TASKS
}

assert len(TASK_DESC) == 26

SEQUENCE_INDEX = list(range(4, (2 * SEQUENCE_LEN) + 4, 2))
assert len(SEQUENCE_INDEX) == SEQUENCE_LEN
ROLLOUT_NUM = 100
TOTAL_ROLLOUT = 100
RAW_TASK_DIM = 4


def process(task_desc: np.ndarray):
    # return np.clip(np.concatenate((task_desc, -task_desc), axis=-1), 0., None)
    assert len(task_desc.shape) == 2
    dxdydz = task_desc[:, :3]
    velocity = np.linalg.norm(dxdydz, axis=1, keepdims=True)
    yaw = task_desc[:, [3]]
    return np.concatenate(
        (dxdydz / velocity, velocity,
         np.sign(np.clip(np.concatenate(
             (yaw, -yaw), axis=1), 0., None)), np.abs(yaw)),
        axis=1)


TASK_DIM = 7

if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    from itertools import product

    OUTPUT_FOLDER = './analysis'
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    Path(f'{OUTPUT_FOLDER}/imgs').mkdir(exist_ok=True)

    task_mean_desc = {
        k: np.round(v[:, :, :], decimals=3)
        for k, v in TASK_DESC.items()
    }

    with open(f'{OUTPUT_FOLDER}/./task_raw_desc.py', 'w') as f:
        f.write('import numpy as np\n')
        f.write('\n')

        for k, v in task_mean_desc.items():

            f.write(
                f'{k} = np.array(\n{np.array2string(v, separator=",")}\n)\n\n')

            _v = np.swapaxes(v, 0, 1)
            fig, axs = plt.subplots(TASK_DIM, 2)
            fig.set_size_inches(18.5, 10.5)

            for i in range(TASK_DIM):
                (ts, rs, ds, vs) = zip(*[(t, r, d, _v[t, r, d]) for (
                    t,
                    r, d) in product(range(3 * SEQUENCE_LEN),
                                     range(TOTAL_ROLLOUT), range(TASK_DIM))])

                db = pd.DataFrame({
                    "timestep": ts,
                    "rollout": rs,
                    "dim": ds,
                    "value": vs
                })
                raw_db_dimed = db.query(f'dim == {i}')
                axs[i][0].set(ylim=(np.min(raw_db_dimed["value"]),
                                    np.max(raw_db_dimed["value"])))
                axs[i][0].set_ylabel(
                    ylabel=["\u0394x", "\u0394y", "\u0394z", "\u0394yaw"][i],
                    rotation=0)
                axs[i][1].set(
                    ylim=(np.min(raw_db_dimed["value"]),
                          np.max(raw_db_dimed["value"])),
                    ylabel=["\u0394x", "\u0394y", "\u0394z", "\u0394yaw"][i])
                axs[i][1].set_ylabel(
                    ylabel=["\u0394x", "\u0394y", "\u0394z", "\u0394yaw"][i],
                    rotation=0)

                sns.lineplot(data=raw_db_dimed,
                             x="timestep",
                             y="value",
                             errorbar=("sd", 2),
                             ax=axs[i][0])
                steped_db_dimed = raw_db_dimed.query(
                    f"timestep in {SEQUENCE_INDEX}")
                sns.lineplot(data=steped_db_dimed,
                             x="timestep",
                             y="value",
                             errorbar=("sd", 2),
                             ax=axs[i][1])
            fig.savefig(f'{OUTPUT_FOLDER}/imgs/{k}.png')
            plt.close()

    with open(f'{OUTPUT_FOLDER}/./task_steps_desc.py', 'w') as f:
        f.write('import numpy as np\n')
        f.write('\n')
        for k, _v in task_mean_desc.items():
            v = _v[SEQUENCE_INDEX, :]
            f.write(
                f'{k} = np.array(\n{np.array2string(v, separator=",")}\n)\n\n')
            # for i in range(TASK_DIM):
            #     sns.lineplot(x=list(range(v.shape[0])), y=v[:, i].tolist(), ax=axs[i])
            # fig.savefig(f'{OUTPUT_FOLDER}/imgs/{k}.png')
            # plt.close()
