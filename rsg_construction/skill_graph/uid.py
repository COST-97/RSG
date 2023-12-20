from typing import Union

i = -1


def gen_uuid() -> int:
    global i
    i += 1
    return i


UUID = int

BODY_UUID = UUID
LEG_UUID = UUID
JOINT_UUID = UUID
AGENT_UUID = Union[BODY_UUID, LEG_UUID, JOINT_UUID]

TASK_UUID = UUID
ENV_UUID = UUID

SKILL_UUID = UUID
