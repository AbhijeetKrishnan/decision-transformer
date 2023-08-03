import argparse
import os
import pickle

import grammar_synthesis
import gymnasium
import numpy as np
import tables as tb
from grammar_synthesis.policy import RandomSampler
from karel_reward import karel_reward
from tqdm import tqdm


def run_episode(env, agent, seed=None):
    "Run an episode with an agent policy and yield the timestep"

    obs, info, terminated, truncated = *env.reset(seed=seed), False, False
    while not terminated and not truncated:
        mask = info['action_mask']
        action = agent.get_action(obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        yield obs, action, reward, terminated, truncated, np.array(mask, dtype=np.bool_)

def write_list_of_dicts_to_hdf5(filename, data_list, base_idx: int=0):
    with tb.File(filename, 'a') as f:
        for idx, data_dict in enumerate(data_list):
            group = f.create_group('/', f'dict_{base_idx}_{idx}')
            for key, value in data_dict.items():
                f.create_array(group, key, value)

def create_batch(episode_lens, observations, actions, rewards, terminals, timeouts, action_masks):
    batch = []

    for episode_len in episode_lens:
        episode = {
            "observations": np.array(observations[:episode_len]),
            "actions": np.array(actions[:episode_len]),
            "rewards": np.array(rewards[:episode_len]),
            "terminals": np.array(terminals[:episode_len]),
            "timeouts": np.array(timeouts[:episode_len]),
            "action_masks": np.array(action_masks[:episode_len]),
        }
        batch.append(episode)
        
        # Remove data from buffers
        del observations[:episode_len]
        del actions[:episode_len]
        del rewards[:episode_len]
        del terminals[:episode_len]
        del timeouts[:episode_len]
        del action_masks[:episode_len]
    return batch

def get_batch_stats(batch):
    batch_returns = [np.sum(p['rewards']) for p in batch]
    batch_program_length = [np.nonzero(p['observations'][-1])[0][-1] + 1 for p in batch if p['terminals'][-1]]
    batch_traj_lens = [p['observations'].shape[0] for p in batch]
    return batch_returns, batch_program_length, batch_traj_lens

def print_batch_stats(batch_idx, batch_returns, batch_program_length, batch_traj_lens):
    tqdm.write('-' * 50)
    tqdm.write(f'Number of samples collected in batch {batch_idx + 1}: {np.sum(batch_traj_lens)}')
    tqdm.write(f'Trajectory returns in batch {batch_idx + 1}: mean = {np.mean(batch_returns):.2f}, std = {np.std(batch_returns):.2f}, max = {np.max(batch_returns):.2f}, min = {np.min(batch_returns):.2f}')
    tqdm.write(f'Number of complete programs in batch {batch_idx + 1}: {len(batch_program_length)}')
    if len(batch_program_length) > 0:
        tqdm.write(f'Complete program length in batch {batch_idx + 1}: mean = {np.mean(batch_program_length):.2f}, std = {np.std(batch_program_length):.2f}, max = {np.max(batch_program_length):.2f}, min = {np.min(batch_program_length):.2f}')
    tqdm.write(f'Episode lengths in batch {batch_idx + 1}: mean = {np.mean(batch_traj_lens):.2f}, std = {np.std(batch_traj_lens):.2f}, max = {np.max(batch_traj_lens):.2f}, min = {np.min(batch_traj_lens):.2f}')
    tqdm.write('-' * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grammar', choices=['microrts', 'karel'], default='microrts')
    parser.add_argument('-n', '--num_episodes', type=int, default=7000)
    parser.add_argument('-b', '--batch_size', type=int, default=65536, help='Number of transitions in a batch to write to file')
    parser.add_argument('-f', '--format', choices=['h5', 'pkl'], default='h5')
    parser.add_argument('--agent', choices=['random'], default='random')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--karel_task', choices=['cleanHouse', 'harvester', 'fourCorners', 'randomMaze', 'stairClimber', 'topOff'], default='cleanHouse')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    batch_size = args.batch_size
    grammar = args.grammar
    seed = args.seed

    if grammar == 'microrts':
        grammar_file = 'decision_transformer/envs/assets/microrts-dsl.lark'
        datapath = f'data/{grammar}-{args.agent}.{args.format}'
        with open(grammar_file) as dsl_file: 
            env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=lambda program, _: len(program), parser='lalr')
    elif grammar == 'karel':
        grammar_file = 'decision_transformer/envs/assets/karel-leaps-dsl.lark'
        karel_task = args.karel_task
        datapath = f'data/{grammar}-{karel_task}-{args.agent}.{args.format}'
        
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
            env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', 
                                 reward_fn=karel_reward, truncation_reward=0.0, parser='lalr', mdp_config=karel_task_config)

    # Delete file if already present
    if args.overwrite and os.path.exists(datapath):
        os.remove(datapath)
        print(f'Deleting {datapath} before generation because it already exists')
    elif not args.overwrite and os.path.exists(datapath):
        print(f'File {datapath} already exists. Please rename it before generating a new dataset')
        return

    if args.agent == 'random':
        agent = RandomSampler(env)
    
    # Generation statistics
    returns = []
    program_length = []
    traj_lens = []

    # Buffer
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    action_masks = []

    episode_lens = [] # stores lengths of trajectories from previous completed episodes
    batch_idx = 0

    with tqdm(range(args.num_episodes), desc="Generating", unit="episode") as progress_bar:
        for _ in progress_bar:
            for obs, action, reward, terminated, truncated, action_mask in run_episode(env, agent, seed):
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(terminated)
                timeouts.append(truncated)
                action_masks.append(action_mask)

                if len(observations) >= batch_size: # buffer is full
                    if len(episode_lens) > 0: # buffer contains at least one full episode that can be written
                        # make episode-wise batches of all current data and write to file
                        paths = create_batch(episode_lens, observations, actions, rewards, terminals, timeouts, action_masks)

                        batch_returns, batch_program_length, batch_traj_lens = get_batch_stats(paths)
                        
                        returns += batch_returns
                        program_length += batch_program_length
                        traj_lens += batch_traj_lens

                        print_batch_stats(batch_idx, batch_returns, batch_program_length, batch_traj_lens)

                        if args.format == 'h5':
                            write_list_of_dicts_to_hdf5(datapath, paths, batch_idx)
                        elif args.format == 'pkl':
                            with open(datapath, 'ab') as f:
                                pickle.dump(paths, f)
                        tqdm.write(f'Wrote trajectories in batch {batch_idx + 1} to {datapath}')

                        batch_idx += 1
                        episode_lens = []
                    else:
                        # full buffer does not contain a complete episode that can be written to file -> double buffer size
                        batch_size *= 2
                        tqdm.write(f'Increased buffer size to {batch_size}')

            prev_episode_length = len(observations) - sum(episode_lens)
            episode_lens.append(prev_episode_length)

    # Write final batch, if necessary
    paths = create_batch(episode_lens, observations, actions, rewards, terminals, timeouts, action_masks)

    if len(paths) > 0:
        batch_returns, batch_program_length, batch_traj_lens = get_batch_stats(paths)
        
        returns += batch_returns
        program_length += batch_program_length
        traj_lens += batch_traj_lens

        print_batch_stats(batch_idx, batch_returns, batch_program_length, batch_traj_lens)

        if args.format == 'h5':
            write_list_of_dicts_to_hdf5(datapath, paths, batch_idx)
        elif args.format == 'pkl':
            with open(datapath, 'ab') as f:
                pickle.dump(paths, f)
        tqdm.write(f'Wrote trajectories in batch {batch_idx + 1} to {datapath}')
    elif len(paths) == 0 and len(observations) > 0:
        tqdm.write(f'Buffer still contains {len(observations)} transitions!')

    # Print total summary statistics
    tqdm.write('=' * 50)
    tqdm.write(f'Total number of samples collected: {np.sum(traj_lens)}')
    tqdm.write(f'Total trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}')
    tqdm.write(f'Total number of complete programs: {len(program_length)}')
    if len(program_length) > 0:
        tqdm.write(f'Total complete program length: mean = {np.mean(program_length):.2f}, std = {np.std(program_length):.2f}, max = {np.max(program_length):.2f}, min = {np.min(program_length):.2f}')
    tqdm.write(f'Total episode lengths: mean = {np.mean(traj_lens):.2f}, std = {np.std(traj_lens):.2f}, max = {np.max(traj_lens):.2f}, min = {np.min(traj_lens):.2f}')
    tqdm.write(f"File size (GB): {os.path.getsize(datapath) / (1024 ** 3):.2f}")

if __name__ == '__main__':
    main()