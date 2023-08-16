import json

from prl_gym.exec_env import ExecEnv2

# Ref.: https://stackoverflow.com/a/34997118
class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)

def get_karel_task_config(karel_task: str):
    if karel_task == 'cleanHouse':
        from leaps.pretrain.leaps_cleanhouse import config
    elif karel_task == 'harvester':
        from leaps.pretrain.leaps_harvester import config
    elif karel_task == 'fourCorners':
        from leaps.pretrain.leaps_fourcorners import config
    elif karel_task == 'randomMaze':
        from leaps.pretrain.leaps_maze import config
    elif karel_task == 'stairClimber':
        from leaps.pretrain.leaps_stairclimber import config
    elif karel_task == 'topOff':
        from leaps.pretrain.leaps_topoff import config
    
    karel_task_config = dict2obj(config)
    karel_task_config.task_definition = 'custom_reward'
    karel_task_config.execution_guided = karel_task_config.rl.policy.execution_guided
    return karel_task_config