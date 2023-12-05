import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'leaps')) # hacky path manipulation to allow LEAPS code to be imported
from leaps.test_karel import get_reward


def karel_reward(program_text, mdp_config):
    return get_reward(program_text, mdp_config['seed'], mdp_config['env_task'])