import os
from itertools import product

env_id = list(range(7))
skill_id = [0, 3, 5]

trained_skills = {}

for (root, dirs, files) in os.walk('./'):
    if root == './':
        assert len(dirs) == 15
        assert all(map(lambda d: d.endswith('oe'), dirs))
    if root.endswith('oe'):
        # assert len(dirs) == 18
        for env, skill in product(env_id, skill_id):
            if not os.path.isdir(f'{root}/ActorCritic_EnvID_{env}_SkillID_{skill}'):
                trained_skills[f"{root}_{env}_{skill}"] = False
                continue

            contents = os.listdir(f'{root}/ActorCritic_EnvID_{env}_SkillID_{skill}')
            assert len(contents) <= 1, f'{root}/{env}_{skill}'
            if len(contents) == 0:
                trained_skills[f"{root}_{env}_{skill}"] = False
                continue

            content = contents[0]
            sks = os.listdir(f'{root}/ActorCritic_EnvID_{env}_SkillID_{skill}/{content}')
            if 'model_350.pt' in sks:
                trained_skills[f"{root}_{env}_{skill}"] = True
            else:
                trained_skills[f"{root}_{env}_{skill}"] = sks

print(trained_skills)