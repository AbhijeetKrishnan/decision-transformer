import argparse
import collections
import pickle

import grammar_synthesis
import gymnasium
import numpy as np

from karel_reward import karel_reward


def generate_random_dataset(grammar_file: str, num_episodes: int=10):
    with open(grammar_file) as dsl_file: 
        env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=karel_reward, parser='lalr')
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "action_masks": [],
    }
    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            mask = info["action_mask"]
            action = env.action_space.sample(mask=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            dataset['observations'].append(obs)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['action_masks'].append(mask)
    env.close()

    dataset['observations'] = np.array(dataset['observations']) # TODO: np.eye(env.vocabulary_size)[dataset['observations']] # one-hot encode tokens in current state
    dataset['actions'] = np.eye(env.action_space.n)[dataset['actions']] # one-hot encode actions
    dataset['rewards'] = np.array(dataset['rewards'])
    dataset['terminals'] = np.array(dataset['terminals'])
    dataset['timeouts'] = np.array(dataset['timeouts'])
    dataset['action_masks'] = np.array(dataset['action_masks'])

    print(f'Generated {dataset["rewards"].shape[0]} samples')
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grammar', choices=['microrts', 'karel'], default='microrts')
    parser.add_argument('-n', '--num_episodes', type=int, default=100)
    args = parser.parse_args()

    grammar = args.grammar

    if grammar == 'microrts':
        grammar_file = 'decision_transformer/envs/assets/microrts-dsl.lark'
    elif grammar == 'karel':
        grammar_file = 'decision_transformer/envs/assets/karel-leaps-dsl.lark'
    
    dataset = generate_random_dataset(grammar_file, args.num_episodes)

    N = dataset['rewards'].shape[0]
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

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    datapath = f'data/{grammar}-random.pkl'
    with open(datapath, 'wb') as f:
        pickle.dump(paths, f)
        print(f'Wrote trajectories to {datapath}')
