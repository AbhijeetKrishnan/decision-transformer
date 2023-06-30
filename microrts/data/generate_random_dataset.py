import gymnasium
import numpy as np

import collections
import pickle

import grammar_synthesis

def generate_random_dataset(num_episodes: int=10):
    with open('decision_transformer/envs/assets/microrts-dsl.lark') as dsl_file: 
        env = gymnasium.make('GrammarSynthesisEnv-v0', grammar=dsl_file.read(), start_symbol='program', reward_fn=lambda symbols: len(symbols), parser='lalr')
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "infos": [],
    }
    for _ in range(num_episodes):
        obs, info, terminated, truncated = *env.reset(), False, False
        while not terminated and not truncated:
            # env.render()
            mask = info["action_mask"]
            action = env.action_space.sample(mask=mask)
            obs, terminated, reward, truncated, info = env.step(action)
            # print(dataset['observations'].shape, obs.shape)
            dataset['observations'].append(obs)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)
        # env.render()
    env.close()

    dataset['observations'] = np.array(dataset['observations'])
    dataset['actions'] = np.array(dataset['actions'])
    dataset['rewards'] = np.array(dataset['rewards'])
    dataset['terminals'] = np.array(dataset['terminals'])
    dataset['timeouts'] = np.array(dataset['timeouts'])

    print(f'Generated {dataset["rewards"].shape[0]} samples')
    return dataset

if __name__ == '__main__':
    num_datasets = 100
    datasets = []

    dataset = generate_random_dataset(num_datasets)

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
        for k in ['observations', 'actions', 'rewards', 'terminals']:
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

    with open(f'data/microrts-random.pkl', 'wb') as f:
        pickle.dump(paths, f)
