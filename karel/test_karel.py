import code
import sys
sys.path.insert(0, 'leaps') # hacky path manipulation to allow LEAPS code to be imported

from leaps.prl_gym.exec_env import ExecEnv2
from leaps.pretrain.get_karel_config import get_karel_task_config


if __name__ == '__main__':
    
    program_text = 'DEF run m( move move move WHILE c( noMarkersPresent c) w( move turnLeft w) m)'
    mdp_config = get_karel_task_config('stairClimber')

    program = program_text.replace('\\', '').replace('\'', '')
    
    karel_env = ExecEnv2(mdp_config)
    reward, pred_program = karel_env.reward(program, is_program_str=True)

    states_0 = pred_program['s_h'][0]
    actions_0 = pred_program['a_h'][0]
    rewards_0 = pred_program['reward_h'][0]

    actions_0_len = pred_program['a_h_len'][0]

    for i in range(actions_0_len):
        karel_env._world.print_state(states_0[i])
        print(actions_0[i], rewards_0[i])
    karel_env._world.print_state(states_0[actions_0_len])

    print(reward)