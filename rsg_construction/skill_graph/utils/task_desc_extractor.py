import csv
import numpy as np
import os
import pickle

RELATIVE_FOLDER = os.path.dirname(__file__)

ENV_START = 2
ENV_MASS_CENTER_LEN = 600 * 10 * 6
ENV_FEEDBACK_FORCE_LEN = 600 * 10 * 4
TASK_MASS_CENTER_LEN = 600 * 10 * 6
TASK_LEGS_MOVE_LEN = 600 * 10 * 4 * 3

desc_dict = {}

for (folder, dirs, files) in os.walk(RELATIVE_FOLDER):
    if len(dirs) == 0 and len(files) > 0:
        assert len(files) != 0
        fs = list(
            filter(lambda f: f.startswith('ActorCritic') and "EnvID_0" in f,
                   files))

        for f in fs:
            with open(f'{folder}/{f}', 'r') as f:
                rows = list(csv.reader(f))
                row = rows[0]
                assert len(row) == 168007

                env_all = np.array(
                    row[ENV_START:(ENV_START + ENV_MASS_CENTER_LEN +
                                   ENV_FEEDBACK_FORCE_LEN)],
                    dtype=np.float32,
                ).reshape((600, 10, 10))
                env_mass_center = env_all[:, :, :6]

                reward_weight = row[-3:]
                task_start = (ENV_START + ENV_MASS_CENTER_LEN +
                              ENV_FEEDBACK_FORCE_LEN + 1)

                task_all = np.array(
                    row[task_start:(task_start + TASK_MASS_CENTER_LEN +
                                    TASK_LEGS_MOVE_LEN)],
                    dtype=np.float32,
                ).reshape(600, 10, 18)

                task_desc = row[task_start - 1]
                task_mass_center_move = task_all[:, :, :6]

                assert np.all(env_mass_center == task_mass_center_move)

                desc_dict[
                    f'{task_desc}_{"_".join(map(str, reward_weight))}'] = np.transpose(
                        task_mass_center_move, (1, 0, 2))

with open(f'{RELATIVE_FOLDER}/task_desc.pkl', 'wb') as f:
    pickle.dump(desc_dict, f)
