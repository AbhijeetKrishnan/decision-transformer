import logging
import os
import sys

import optuna

from experiment import experiment, get_trajectories


def objective_factory(task):
    def objective(trial):

        variant = {
            'env': 'karel',
            'dataset': 'playback',
            'mode': 'delayed',
            'env_targets': '1',
            'scale': 1.0,
            'K': trial.suggest_int('K', 10, 30, step=20),
            'pct_traj': trial.suggest_categorical('pct_traj', [0.01, 0.1, 0.5, 1.0]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'model_type': 'dt',
            'embed_dim': trial.suggest_categorical('embed_dim', [32, 64, 128, 256]),
            'n_head': trial.suggest_categorical('n_head', [1, 2, 4]), # embed_dim % n_head == 0
            'n_layer': trial.suggest_int('n_layer', 1, 4),
            'activation_function': trial.suggest_categorical('activation_function', ['relu', 'silu', 'gelu', 'tanh', 'gelu_new']),
            'dropout': trial.suggest_float('dropout', 0.0, 0.2),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 1e2, 1e4, log=True),
            'num_eval_episodes': 64, # arbitrary
            'max_iters': trial.suggest_int('max_iters', 1, 10),
            'num_steps_per_iter': trial.suggest_int('num_steps_per_iter', 1e2, 1e4, log=True),
            'device': 'cuda',
            'log_to_wandb': False,
            'use_seq_state_embedding': trial.suggest_categorical('use_seq_state_embedding', [True, False]),
            'karel_task': task,
            'seed': 75092, # data generation seed
            'sample': trial.suggest_categorical('sample', ['length', 'reward']),
        }
        env, trajectories = get_trajectories(variant)
        result = experiment('karel-optuna', env, trajectories, variant)
        return result
    return objective

def main():
    task = sys.argv[1]
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f'{task}'
    storage_name = f'sqlite:///karel-optuna.sqlite3'
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    try:
        study.optimize(objective_factory(task))
    except:
        pass
    print(study.best_params)


if __name__ == '__main__':
    main()
