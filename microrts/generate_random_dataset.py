import argparse
import collections
import pickle

import grammar_synthesis
import gymnasium
import numpy as np
import tables as tb
from karel_reward import karel_reward


def generate_random_dataset(env, num_episodes: int=10, seed: int=None):
    
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "action_masks": [],
    }
    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(seed=seed), False, False
        while not terminated and not truncated:
            mask = info['action_mask']
            action = env.action_space.sample(mask=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            dataset['observations'].append(obs)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['action_masks'].append(np.array(mask, dtype=np.bool_))
    env.close()

    dataset['observations'] = np.array(dataset['observations']) # TODO: np.eye(env.vocabulary_size)[dataset['observations']] # one-hot encode tokens in current state
    dataset['actions'] = np.array(dataset['actions'])
    dataset['rewards'] = np.array(dataset['rewards'])
    dataset['terminals'] = np.array(dataset['terminals'])
    dataset['timeouts'] = np.array(dataset['timeouts'])
    dataset['action_masks'] = np.array(dataset['action_masks'], np.bool_)

    return dataset

def write_list_of_dicts_to_hdf5(filename, data_list, base_idx: int=0):
    with tb.File(filename, 'a') as f:
        for idx, data_dict in enumerate(data_list):
            group = f.create_group('/', f'dict_{base_idx + idx}')
            for key, value in data_dict.items():
                f.create_array(group, key, value)
        f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grammar', choices=['microrts', 'karel'], default='microrts')
    parser.add_argument('-n', '--num_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('-f', '--format', choices=['h5', 'pkl'], default='h5')
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
        datapath = f'data/{grammar}-{args.karel_task}-random.{args.format}'
        karel_config = None # TODO: add task-specific Karel config here from LEAPS codebase
        with open(grammar_file) as dsl_file: 
            env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=karel_reward, parser='lalr', mdp_config=karel_config)
    
    returns = []
    num_samples = 0 # number of transitions
    programs = []
    program_length = []

    batch_lens = [args.batch_size] * (args.num_episodes // args.batch_size) + ([args.num_episodes % args.batch_size] if args.num_episodes % args.batch_size != 0 else [])
    for batch_idx, batch_size in enumerate(batch_lens):
        dataset = generate_random_dataset(env, batch_size, args.seed + batch_idx)

        N = dataset['rewards'].shape[0] # number of episodes
        data_ = collections.defaultdict(list)

        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        paths = []
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'actions', 'rewards', 'terminals', 'action_masks']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

        batch_returns = [np.sum(p['rewards']) for p in paths]
        batch_num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        batch_programs = [dataset['observations'][i] for i in range(len(dataset['observations'])) if dataset['terminals'][i]]
        batch_program_length = [np.nonzero(dataset['observations'][i])[0][-1] + 1 for i in range(len(dataset['observations'])) if dataset['terminals'][i]]

        returns += batch_returns
        num_samples += batch_num_samples
        programs += batch_programs
        program_length += batch_program_length

        print(f'Number of samples collected in batch {batch_idx + 1}/{len(batch_lens)}: {batch_num_samples}')
        print(f'Trajectory returns in batch {batch_idx + 1}/{len(batch_lens)}: mean = {np.mean(batch_returns)}, std = {np.std(batch_returns)}, max = {np.max(batch_returns)}, min = {np.min(returns)}')
        print(f'Number of complete programs in batch {batch_idx + 1}/{len(batch_lens)}: {len(batch_programs)}')
        if len(batch_programs) > 0:
            print(f'Complete program length in batch {batch_idx + 1}/{len(batch_lens)}: mean = {np.mean(batch_program_length)}, std = {np.std(batch_program_length)}, max = {np.max(batch_program_length)}, min = {np.min(batch_program_length)}')

        if args.format == 'h5':
            write_list_of_dicts_to_hdf5(datapath, paths, batch_idx * args.batch_size)
        elif args.format == 'pkl':
            with open(datapath, 'ab') as f:
                pickle.dump(paths, f)
        print(f'Wrote trajectories in batch {batch_idx + 1}/{len(batch_lens)} to {datapath}')

    print('-' * 50)
    print(f'Total number of samples collected: {num_samples}')
    print(f'Total trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    print(f'Total number of complete programs: {len(programs)}')
    if len(program_length) > 0:
        print(f'Total complete program length: mean = {np.mean(program_length)}, std = {np.std(program_length)}, max = {np.max(program_length)}, min = {np.min(program_length)}')

if __name__ == '__main__':
    main()