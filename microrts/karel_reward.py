import json
import sys

sys.path.insert(0, 'leaps') # hacky path manipulation to allow LEAPS code to be imported

from leaps.prl_gym.exec_env import ExecEnv2


# Ref.: https://stackoverflow.com/a/34997118
class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)

def karel_reward(program_text, mdp_config=None):
    program = program_text.replace('\\', '').replace('\'', '')

    config = dict2obj(mdp_config)
    config.task_definition = 'custom_reward'
    config.execution_guided = config.rl.policy.execution_guided
    
    karel_env = ExecEnv2(config)
    reward, pred_program = karel_env.reward(program)
    return reward