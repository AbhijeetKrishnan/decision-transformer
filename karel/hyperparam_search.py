import logging
import sys

import optuna

from experiment import experiment, get_trajectories


def objective_factory(task):
    def objective(trial):
        n_head = trial.suggest_int('n_head', 1, 3, step=1)
        variant = {
            'env': 'karel',
            'dataset': 'playback',
            'mode': 'delayed',
            'env_targets': '1',
            'scale': 1.0,
            'K': trial.suggest_int('K', 10, 50, step=20),
            'pct_traj': trial.suggest_float('pct_traj', 0.1, 1.0, step=0.3),
            'batch_size': trial.suggest_int('batch_size', 32, 128, step=32),
            'model_type': 'dt',
            'embed_dim': trial.suggest_int('embed_dim', 32, 128, step=32),
            'n_layer': trial.suggest_int('n_layer', n_head, n_head * 3, step=n_head), # must satisfy n_state % n_head == 0
            'n_head': n_head,
            'activation_function': trial.suggest_categorical('activation_function', ['relu', 'gelu_new']),
            'dropout': trial.suggest_float('dropout', 0.0, 0.2, step=0.1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 1e2, 1e4, log=True),
            'num_eval_episodes': 64,
            'max_iters': 1,
            'num_steps_per_iter': trial.suggest_int('num_steps_per_iter', 1e2, 1e4, log=True),
            'device': 'cuda',
            'log_to_wandb': False,
            'use_seq_state_embedding': trial.suggest_categorical('use_seq_state_embedding', [True, False]),
            'karel_task': task,
            'seed': 68997,
            'sample': trial.suggest_categorical('sample', ['length', 'reward']),
        }
        env, trajectories = get_trajectories(variant)
        result = experiment('karel-optuna', env, trajectories, variant)
        return result
    return objective

def main():
    task = sys.argv[1]
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f'karel-optuna-{task}'
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective_factory(task), n_trials=3)
    print(study.best_params)


if __name__ == '__main__':
    main()