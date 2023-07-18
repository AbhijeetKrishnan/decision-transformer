import sys
sys.path.insert(0, 'leaps')

from leaps.prl_gym.exec_env import ExecEnv2

class Container(object):
    pass


def karel_reward(program_text):
    program = program_text.replace('\\', '').replace('\'', '')
    # print(program)

    config = Container()
    config.env_name = 'karel'
    config.env_task = 'cleanHouse' # ['cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff']
    config.width = 22
    config.height = 14
    config.wall_prob = 0.25
    config.num_demo_per_program = 1
    config.seed = 0
    config.task_definition = 'custom_reward'
    config.reward_diff = True
    config.final_reward_scale = False
    config.max_demo_length = 100
    
    karel_env = ExecEnv2(config)
    reward, pred_program = karel_env.reward(program)
    return reward