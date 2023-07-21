import argparse
import pickle

import grammar_synthesis
import gymnasium
import numpy as np
import tables as tb
from karel_reward import karel_reward


def run_episode(env, agent, seed=None): # TODO: import random policy from grammar-synthesis package (refactor that repo to allow this too)
    "Run an episode with an agent policy and yield the timestep"

    obs, info, terminated, truncated = *env.reset(seed=seed), False, False
    while not terminated and not truncated:
        mask = info['action_mask']
        action = env.action_space.sample(mask=mask) # TODO: refactor to allow arbitrary agent policy
        obs, reward, terminated, truncated, info = env.step(action)
        yield obs, action, reward, terminated, truncated, np.array(mask, dtype=np.bool_)

def write_list_of_dicts_to_hdf5(filename, data_list, base_idx: int=0):
    with tb.File(filename, 'a') as f:
        for idx, data_dict in enumerate(data_list):
            group = f.create_group('/', f'dict_{base_idx + idx}')
            for key, value in data_dict.items():
                f.create_array(group, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grammar', choices=['microrts', 'karel'], default='microrts')
    parser.add_argument('-n', '--num_episodes', type=int, default=5000)
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Number of transitions in a batch to write to file')
    parser.add_argument('-f', '--format', choices=['h5', 'pkl'], default='h5')
    parser.add_argument('--agent', choices=['random'], default='random')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--karel_task', choices=['cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff'], default='cleanHouse')
    args = parser.parse_args()

    grammar = args.grammar

    if grammar == 'microrts':
        grammar_file = 'decision_transformer/envs/assets/microrts-dsl.lark'
        datapath = f'data/{grammar}-random.{args.format}'
        with open(grammar_file) as dsl_file: 
            env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=lambda program, _: len(program), parser='lalr')
    elif grammar == 'karel':
        grammar_file = 'decision_transformer/envs/assets/karel-leaps-dsl.lark'
        karel_task = args.karel_task
        datapath = f'data/{grammar}-{karel_task}-random.{args.format}'
        
        if karel_task == 'cleanHouse':
            from leaps.pretrain.leaps_cleanhouse import config
            karel_task_config = config
        elif karel_task == 'harvester':
            from leaps.pretrain.leaps_harvester import config
            karel_task_config = config
        elif karel_task == 'fourCorners':
            from leaps.pretrain.leaps_fourcorners import config
            karel_task_config = config
        elif karel_task == 'randomMaze':
            from leaps.pretrain.leaps_maze import config
            karel_task_config = config
        elif karel_task == 'stairClimber':
            from leaps.pretrain.leaps_stairclimber import config
            karel_task_config = config
        elif karel_task == 'topOff':
            from leaps.pretrain.leaps_topoff import config
            karel_task_config = config
        
        with open(grammar_file) as dsl_file: 
            env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=karel_reward, parser='lalr', mdp_config=karel_task_config)
    
    agent_name = args.agent
    if agent_name == 'random':
        agent = None # TODO: fix this after refactoring grammar-synthesis

    seed = args.seed
    
    # Generation statistics
    returns = []
    num_samples = 0 # number of transitions
    program_length = []
    traj_lens = []

    # Buffer
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    action_masks = []

    num_transitions = 0 # number of transitions currently in buffer
    episode_lens = [] # stores lengths of trajectories from previous completed episodes
    batch_idx = 0

    for _ in range(args.num_episodes):
        for obs, action, reward, terminated, truncated, action_mask in run_episode(env, agent, seed):
            if num_transitions <= args.batch_size:
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(terminated)
                timeouts.append(truncated)
                action_masks.append(action_mask)

                num_transitions += 1
            else:
                # make episode-wise batches of all current data and write to file
                paths = []
                for episode_len in episode_lens:
                    episode = {
                        "observations": np.array(observations[:episode_len]),
                        "actions": np.array(actions[:episode_len]),
                        "rewards": np.array(rewards[:episode_len]),
                        "terminals": np.array(terminals[:episode_len]),
                        "timeouts": np.array(timeouts[:episode_len]),
                        "action_masks": np.array(action_masks[:episode_len]),
                    }
                    paths.append(episode)
                    
                    # Remove data from buffers
                    observations = observations[episode_len:]
                    actions = actions[episode_len:]
                    rewards = rewards[episode_len:]
                    terminals = terminals[episode_len:]
                    timeouts = timeouts[episode_len:]
                    action_masks = action_masks[episode_len:]

                if len(paths) > 0:
                    batch_returns = [np.sum(p['rewards']) for p in paths]
                    batch_num_samples = np.sum([p['rewards'].shape[0] for p in paths])
                    batch_program_length = [np.nonzero(p['observations'][-1])[0][-1] + 1 for p in paths if p['terminals'][-1]]
                    batch_traj_lens = [len(p['observations']) for p in paths]
                    
                    returns += batch_returns
                    num_samples += batch_num_samples
                    program_length += batch_program_length
                    traj_lens += batch_traj_lens

                    print('-' * 50)
                    print(f'Number of samples collected in batch {batch_idx + 1}: {batch_num_samples}')
                    print(f'Trajectory returns in batch {batch_idx + 1}: mean = {np.mean(batch_returns)}, std = {np.std(batch_returns)}, max = {np.max(batch_returns)}, min = {np.min(returns)}')
                    if len(batch_program_length) > 0:
                        print(f'Complete program length in batch {batch_idx + 1}: mean = {np.mean(batch_program_length)}, std = {np.std(batch_program_length)}, max = {np.max(batch_program_length)}, min = {np.min(batch_program_length)}')
                    print(f'Episode lengths in batch {batch_idx + 1}: mean = {np.mean(batch_traj_lens)}, std = {np.std(batch_traj_lens)}, max = {np.max(batch_traj_lens)}, min = {np.min(batch_traj_lens)}')
                    print('-' * 50)

                if args.format == 'h5':
                    write_list_of_dicts_to_hdf5(datapath, paths, batch_idx * args.batch_size)
                elif args.format == 'pkl':
                    with open(datapath, 'ab') as f:
                        pickle.dump(paths, f)
                print(f'Wrote trajectories in batch {batch_idx + 1} to {datapath}')

                batch_idx += 1
                num_transitions -= sum(episode_lens)
                episode_lens = []
        
        prev_episode_length = num_transitions - sum(episode_lens)
        episode_lens.append(prev_episode_length)

    # Write final batch, if any
    paths = []
    for episode_len in episode_lens:
        episode = {
            "observations": np.array(observations[:episode_len]),
            "actions": np.array(actions[:episode_len]),
            "rewards": np.array(rewards[:episode_len]),
            "terminals": np.array(terminals[:episode_len]),
            "timeouts": np.array(timeouts[:episode_len]),
            "action_masks": np.array(action_masks[:episode_len]),
        }
        paths.append(episode)
        
        # Remove data from buffers
        observations = observations[episode_len:]
        actions = actions[episode_len:]
        rewards = rewards[episode_len:]
        terminals = terminals[episode_len:]
        timeouts = timeouts[episode_len:]
        action_masks = action_masks[episode_len:]

    if len(paths) > 0:
        batch_returns = [np.sum(p['rewards']) for p in paths]
        batch_num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        batch_program_length = [np.nonzero(p['observations'][-1])[0][-1] + 1 for p in paths if p['terminals'][-1]]
        batch_traj_lens = [len(p['observations']) for p in paths]
        
        returns += batch_returns
        num_samples += batch_num_samples
        # programs += batch_programs
        program_length += batch_program_length
        traj_lens += batch_traj_lens

        # Print batch summary statistics
        print('-' * 50)
        print(f'Number of samples collected in batch {batch_idx + 1}: {batch_num_samples}')
        print(f'Trajectory returns in batch {batch_idx + 1}: mean = {np.mean(batch_returns)}, std = {np.std(batch_returns)}, max = {np.max(batch_returns)}, min = {np.min(returns)}')
        if len(batch_program_length) > 0:
            print(f'Complete program length in batch {batch_idx + 1}: mean = {np.mean(batch_program_length)}, std = {np.std(batch_program_length)}, max = {np.max(batch_program_length)}, min = {np.min(batch_program_length)}')
        print(f'Episode lengths in batch {batch_idx + 1}: mean = {np.mean(batch_traj_lens)}, std = {np.std(batch_traj_lens)}, max = {np.max(batch_traj_lens)}, min = {np.min(batch_traj_lens)}')
        print('-' * 50)

    if args.format == 'h5':
        write_list_of_dicts_to_hdf5(datapath, paths, batch_idx * args.batch_size)
    elif args.format == 'pkl':
        with open(datapath, 'ab') as f:
            pickle.dump(paths, f)
    print(f'Wrote trajectories in batch {batch_idx + 1} to {datapath}')

    batch_idx += 1
    num_transitions -= sum(episode_lens)
    episode_lens = []

    # Print total summary statistics
    print('=' * 50)
    print(f'Total number of samples collected: {num_samples}')
    print(f'Total trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    if len(program_length) > 0:
        print(f'Total complete program length: mean = {np.mean(program_length)}, std = {np.std(program_length)}, max = {np.max(program_length)}, min = {np.min(program_length)}')
    print(f'Total episode lengths: mean = {np.mean(traj_lens)}, std = {np.std(traj_lens)}, max = {np.max(traj_lens)}, min = {np.min(traj_lens)}')

if __name__ == '__main__':
    main()