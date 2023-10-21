import numpy as np
import torch


def show_grads(model, tol=1e-2):
    # Ref.: https://blog.briankitano.com/llama-from-scratch/
    return sorted([(name, 100.0 * float(torch.sum(torch.abs(param) <= tol)) / float(param.nelement())) \
                   for name, param in model.named_parameters() if param.requires_grad], key=lambda t: t[1], reverse=True)

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):
    # for bc
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, info = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    action_masks = torch.from_numpy(info['action_mask']).reshape(1, act_dim).to(device=device, dtype=torch.bool)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32)), # - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            action_masks.to(dtype=torch.bool),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        
        state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            env.render()
            print(reward)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        cur_action_mask = torch.from_numpy(info['action_mask']).reshape(1, act_dim).to(device=device, dtype=torch.bool)
        states = torch.cat([states, cur_state], dim=0)
        action_masks = torch.cat([action_masks, cur_action_mask], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done or truncated:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    # for dt
    model.eval()
    model.to(device=device)

    # print('\n'.join([str(param) for param in show_grads(model)]))

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, info = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.long)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    action_masks = torch.from_numpy(info['action_mask']).reshape(1, act_dim).to(device=device, dtype=torch.bool)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.long)), # - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            action_masks.to(dtype=torch.bool),
            timesteps.to(dtype=torch.long),
            visualize_logits=None,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            env.render()
            print(reward)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        cur_action_mask = torch.from_numpy(info['action_mask']).reshape(1, act_dim).to(device=device, dtype=torch.bool)
        states = torch.cat([states, cur_state], dim=0)
        action_masks = torch.cat([action_masks, cur_action_mask], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done or truncated:
            break

    return episode_return, episode_length
