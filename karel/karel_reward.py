import sys

sys.path.insert(0, 'leaps')

from leaps.prl_gym.exec_env import ExecEnv2


def karel_reward(program_text, mdp_config=None):
    program = program_text.replace('\\', '').replace('\'', '')
    
    karel_env = ExecEnv2(mdp_config['args'])
    reward, pred_program = karel_env.reward(program, is_program_str=True)
    return reward