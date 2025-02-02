import argparse
import pickle
import sys
from datetime import datetime

sys.path.insert(
    0, "leaps"
)  # hacky path manipulation to allow LEAPS code to be imported

import grammar_synthesis
import gymnasium
import numpy as np
import tables as tb
import torch
import wandb
import yaml
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from karel_reward import karel_reward
from leaps.pretrain.get_karel_config import get_karel_task_config


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def read_list_of_dicts_from_hdf5(filename):
    data_list = []
    with tb.File(filename, "r") as f:
        for node in f.iter_nodes(where="/"):
            data_dict = {}
            for array_key in node:
                data_dict[array_key.name] = array_key.read()
            data_list.append(data_dict)
    return data_list


def get_trajectories(variant):
    file_format = variant.get("format", "h5")
    env_name, dataset = variant["env"], variant["dataset"]

    if env_name == "karel":
        karel_task = variant["karel_task"]
        dataset_path = f"data/{env_name}-{karel_task}-{dataset}.{file_format}"
        karel_task_config = get_karel_task_config(karel_task, variant["seed"])

        with open("decision_transformer/envs/assets/karel-leaps-dsl.pg") as dsl_file:
            env = gymnasium.make(
                "GrammarSynthesisEnv-v0",
                grammar=dsl_file.read(),
                reward_fn=karel_reward,
                max_len=50,
                mdp_config=karel_task_config,
            )  # TODO: handle state max seq len better
        # env_targets = env_targets if env_targets is not None else [1]
    else:
        raise NotImplementedError

    # load dataset
    if file_format == "h5":
        trajectories = read_list_of_dicts_from_hdf5(dataset_path)
    elif file_format == "pkl":
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

    return env, trajectories


def experiment(
    exp_prefix,
    env,
    trajectories,
    variant,
):
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    rng = np.random.default_rng(variant["seed"])

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    group_name = f'{exp_prefix}-{env_name}-{variant["karel_task"]}-{dataset}'
    exp_prefix = f'{group_name}-{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}'
    env_targets = variant.get("env_targets", None)
    if env_targets is not None and isinstance(env_targets, str):
        env_targets = list(map(int, env_targets.split(",")))
    eval_seeds = variant.get("eval_seeds", None)
    if eval_seeds is None:
        eval_seeds = [variant["seed"]]
    elif isinstance(eval_seeds, str):
        eval_seeds = list(map(int, eval_seeds.split(",")))
    scale = variant.get("scale", 1000.0)

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # for discrete environments, assuming all actions are mapped to integers in a Discrete space
    vocab_size = env.observation_space.nvec[0]
    use_max_log_prob = variant.get("use_max_log_prob", False)
    use_seq_state_embedding = variant.get("use_seq_state_embedding", False)
    gru_hidden_size = variant.get("gru_hidden_size", None)
    gru_num_layers = variant.get("gru_num_layers", 1)
    gru_dropout = variant.get("gru_dropout", 0.0)

    # save all path information into separate lists
    mode = variant.get("mode", "normal")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f'Starting new experiment: {env_name} {variant["karel_task"]} {dataset}')
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print(f"Max episode length: {np.max(traj_lens)}, min: {np.min(traj_lens)}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)
    max_ep_len = traj_lens.max()

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    if variant["sample"] == "length":
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    elif variant["sample"] == "reward":
        p_sample = returns[sorted_inds]
        if np.all(p_sample):
            p_sample = np.ones_like(p_sample) / len(p_sample)
        else:
            p_sample /= sum(returns[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = rng.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to length/returns
        )

        s, a, r, d, rtg, action_masks, timesteps, mask = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = rng.integers(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(
                np.eye(act_dim)[traj["actions"][si : si + max_len]].reshape(
                    1, -1, act_dim
                )
            )
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            action_masks.append(
                traj["action_masks"][si : si + max_len].reshape(1, -1, act_dim)
            )

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            # s[-1] = (s[-1] - state_mean) / state_std # not needed since state is discrete
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            action_masks[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), action_masks[-1]], axis=1
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [
                        np.zeros((1, max_len - tlen), dtype=np.bool_),
                        np.ones((1, tlen), dtype=np.bool_),
                    ],
                    axis=1,
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.long, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        action_masks = torch.from_numpy(np.concatenate(action_masks, axis=0)).to(
            dtype=torch.bool, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, action_masks, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, programs = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length, prog = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                programs.append(prog)
            return {
                f"target_{target_rew}_return_max": np.max(returns),
                f"target_{target_rew}_best_prog": programs[np.argmax(returns)],
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    if model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            vocab_size=vocab_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            gru_dropout=gru_dropout,
            action_tanh=False,
            use_seq_state_embedding=use_seq_state_embedding,
            use_max_log_prob=use_max_log_prob,
            seed=variant["seed"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            use_max_log_prob=use_max_log_prob,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, a),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.CrossEntropyLoss()(a_hat, a),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        wandb.watch(model)

    all_outputs = []
    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        all_outputs.append(outputs)
        if log_to_wandb:
            wandb.log(outputs)
    objective = all_outputs[-1]["evaluation/target_2.2_return_max"]  # last iteration TODO: remove this hardcoding
    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument("--env", type=str, default="karel", help="Environment to use")
    parser.add_argument(
        "--dataset",
        type=str,
        default="playback",
        help="Dataset to use for training (identified by policy used for generation)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "delayed"],
        default="delayed",
        help='"normal" for standard setting, "delayed" for moving rewards to end of trajectory',
    )  # 'normal' for standard setting, 'delayed' for sparse
    parser.add_argument(
        "--env_targets",
        type=str,
        default="1",
        help="comma-separated list of target returns",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale for rewards/returns"
    )
    parser.add_argument("--K", type=int, default=30, help="Context length")
    parser.add_argument(
        "--pct_traj",
        type=float,
        default=1.0,
        help="Use top x% of trajectories (for %BC experiments)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of transitions to sample from a trajectory in a batch",
    )
    parser.add_argument(
        "--model_type",
        choices=["bc", "dt"],
        default="dt",
        help='"dt" for decision transformer, "bc" for behavior cloning',
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Size of (state, action, timestep) embedding layers",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=3,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=1,
        help="Number of attention heads for each attention layer in the Transformer encoder",
    )
    parser.add_argument(
        "--activation_function",
        choices=["relu", "silu", "gelu", "tanh", "gelu_new"],
        default="relu",
        help="Activation function",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout value to use for Transformer embeddings, encoder, pooler and attention",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Learning rate parameter for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=1e-4,
        help="Weight decay parameter for AdamW optimizer",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10000,
        help="Number of warmup steps for linear learning rate scheduling",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=64,
        help="Number of rollouts to sample to evaluate obtained return",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=10,
        help="Number of (train, eval) cycles to run",
    )
    parser.add_argument(
        "--num_steps_per_iter",
        type=int,
        default=10000,
        help="Number of (train, loss, backprop) steps to run per iteration",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--format",
        choices=["h5", "pkl"],
        default="h5",
        help="Format in which input dataset is stored",
    )
    parser.add_argument(
        "--use_max_log_prob",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use max log prob instead of sampling from distribution",
    )
    parser.add_argument(
        "--use_seq_state_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sequential state embedding instead of linear state embedding",
    )
    parser.add_argument(
        "--karel_task",
        choices=[
            "cleanHouse",
            "harvester",
            "fourCorners",
            "randomMaze",
            "stairClimber",
            "topOff",
        ],
        default="cleanHouse",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--sample",
        choices=["length", "reward"],
        default="reward",
        help="How to weight a trajectory when sampling from it",
    )
    parser.add_argument("--eval_seeds", type=str, default=None, help="Seeds on which to run the evaluation step")
    parser.add_argument("--gru_hidden_size", type=int, default=57, help="Hidden size of GRU in sequential state embedder")
    parser.add_argument("--gru_num_layers", type=int, default=1, help="Number of layers in GRU in sequential state embedder")
    parser.add_argument("--gru_dropout", type=float, default=0.0, help="Dropout in GRU in sequential state layer")

    args = parser.parse_args()
    variant = vars(args)
    if args.config is not None:
        with open(args.config, "r") as params:
            data = yaml.load(params, Loader=yaml.CLoader)
        yaml_config = {**data["task"], **data["training"], **data["hyperparams"]}

        # # if parameter exists in cmdline args, override config file
        # for key, value in variant.items():
        #     if key in yaml_config:
        #         yaml_config[key] = value
        variant = yaml_config
    print(variant)
    env, trajectories = get_trajectories(variant)
    experiment("karel-experiment", env, trajectories, variant)
