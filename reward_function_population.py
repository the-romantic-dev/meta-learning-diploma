from pathlib import Path

import imageio
import torch
import numpy as np
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import os
from gymnasium import wrappers as w
import torch.nn as nn
from typing import List

from hebbian_update import hebbian_update
from nn_models import CNN_heb, MLP_heb
from visual import spinner_and_time
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame
from gymnasium.vector import SyncVectorEnv
import time
from IPython import display
from PIL import Image


def _weights_init(m, init_weights):
    if isinstance(m, torch.nn.Linear):
        if init_weights == 'xa_uni':
            torch.nn.init.xavier_uniform(m.weight.data, 0.3)
        elif init_weights == 'sparse':
            torch.nn.init.sparse_(m.weight.data, 0.8)
        elif init_weights == 'uni':
            torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
        elif init_weights == 'normal':
            torch.nn.init.normal_(m.weight.data, 0, 0.024)
        elif init_weights == 'ka_uni':
            torch.nn.init.kaiming_uniform_(m.weight.data, 3)
        elif init_weights == 'uni_big':
            torch.nn.init.uniform_(m.weight.data, -1, 1)
        elif init_weights == 'xa_uni_big':
            torch.nn.init.xavier_uniform(m.weight.data)
        elif init_weights == 'ones':
            torch.nn.init.ones_(m.weight.data)
        elif init_weights == 'zeros':
            torch.nn.init.zeros_(m.weight.data)
        elif init_weights == 'default':
            pass


def make_envs(env_name: str, population_size: int) -> tuple[gym.Env, bool, int, int]:
    envs = [gym.make(env_name, verbose=0, render_mode='rgb_array') for _ in range(population_size)]
    # envs[0] = gym.make(env_name, verbose=0, render_mode='human')
    if hasattr(envs[0].unwrapped, 'get_action_meanings') and 'FIRE' in envs[0].unwrapped.get_action_meanings():
        envs = [FireEpisodicLifeEnv(env) for env in envs]
    shape = envs[0].observation_space.shape
    if len(shape) == 3:
        envs = [ScaledFloatFrame(w.ResizeObservation(env, (84, 84))) for env in envs]
        return envs, True, 3, get_action_dim(envs[0])
    else:
        return envs, False, shape[0] if shape else envs[0].observation_space.n, get_action_dim(envs[0])


def get_action_dim(env: gym.Env) -> int:
    if isinstance(env.action_space, Box):
        return env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        return env.action_space.n
    else:
        raise ValueError('Only Box and Discrete action spaces supported')


def make_policy(is_pixel_env, action_dim, input_dim):
    return CNN_heb(input_dim, action_dim) if is_pixel_env else MLP_heb(input_dim, action_dim)


def init_policy_weights(init_weights_type: str, is_pixel_env: bool, policy: CNN_heb | MLP_heb,
                        initial_weights_co: torch.Tensor):
    def weights_init(m):
        _weights_init(m, init_weights_type)

    if init_weights_type == 'coevolve':
        nn.utils.vector_to_parameters(initial_weights_co, policy.parameters())
    else:
        # Randomly sample initial weights from chosen distribution
        policy.apply(weights_init)

        # Load CNN paramters
        if is_pixel_env:
            cnn_weights1 = initial_weights_co[:162]
            cnn_weights2 = initial_weights_co[162:]
            list(policy.parameters())[0].data = torch.from_numpy(cnn_weights1.reshape((6, 3, 3, 3)))
            list(policy.parameters())[1].data = torch.from_numpy(cnn_weights2.reshape((8, 6, 5, 5)))
    return policy.float()


def get_action(env_name: str, model_out: torch.Tensor):
    if 'CarRacing' in env_name:
        model_out = model_out.t()
        actions = torch.stack([torch.tanh(model_out[0]), torch.sigmoid(model_out[1]), torch.sigmoid(model_out[2])])
        return actions.t().numpy()
    elif 'AntBulletEnv' in env_name:
        return model_out.numpy()
    else:
        raise ValueError(f"Incorrect env_name {env_name}")


def neg_count_add(curr_negs: np.ndarray, rewards: np.ndarray, environment, t):
    if 'AntBulletEnv' in environment and t <= 200:
        return np.zeros(len(rewards))
    adds = (rewards < 0.0).astype(int)
    return curr_negs * (rewards < 0.0).astype(int) + adds

def update_policies_weights(pixel_env: bool, population_policies, population_weights):
    for i, policy in enumerate(population_policies):
        # Собираем в список тензоры с новыми весами для текущей политики
        new_weights = [p[i] for p in population_weights]

        # Берём именно те параметры, которые хотим перезаписать
        # (здесь вы, судя по коду, пропускаете первые два параметра по задумке)
        params = list(policy.parameters())[2:]

        # Меняем значения параметров без отслеживания градиентов
        with torch.no_grad():
            for p_tensor, new_w in zip(params, new_weights):
                # Убеждаемся, что новый тензор на том же устройстве и имеет ту же форму
                new_w = new_w.to(p_tensor.device)
                # Классический приём: пишем прямо в .data
                p_tensor.data.copy_(new_w)


def adapt_observations(observations, envs: list[gym.Env], pixel_env: bool):
    if isinstance(envs[0].observation_space, Discrete):
        return [(obs == torch.arange(envs[0].observation_space.n)).float() for obs in observations]
    if pixel_env:
        return np.swapaxes(observations, 3, 1)
    return observations


def get_policies_outputs(population_policies, observations, environment):
    policies_outputs_func = lambda: [list(p([observations[i]])) for i, p in enumerate(population_policies)]
    policies_outputs = spinner_and_time(policies_outputs_func, 'Получение выходов нейронов моделей')
    if 'AntBulletEnv' in environment:
        policies_outputs[3] = torch.tanh(policies_outputs[3])
    return [torch.stack(grouped) for grouped in list(zip(*policies_outputs))]


def calc_actions(policies_outputs, environment):
    return spinner_and_time(lambda: get_action(environment, model_out=policies_outputs[3]), 'Рассчет действий')


def fitness_hebb(
        iteration: int,
        folder: Path,
        hebb_rule: str,
        environment: str,
        init_weights_type='uni',
        population_hebb_coeffs: list[torch.Tensor] = None,
        population_initial_weights_co: list[torch.Tensor] = None
) -> float:
    population_size = len(population_hebb_coeffs)
    batch_size = population_size
    rewards = np.zeros(population_size)
    for i in range(population_size // batch_size):
        start_index = i * batch_size
        end_index = i + 1 * batch_size
        if end_index > population_size:
            end_index = population_size
        rew = batch_fitness_hebb(
            iteration, folder,
            hebb_rule, environment, init_weights_type,
            population_hebb_coeffs[start_index:end_index],
            population_initial_weights_co[start_index:end_index]
        )
        rewards[start_index:end_index] = rew
    return rewards


def make_env_step(envs, actions, environment):
    results = spinner_and_time(lambda: [env.step(action) for env, action in zip(envs, actions)], 'Шаг симуляции')
    observations, rewards, terminateds, truncateds, _ = list([np.array(res) for res in zip(*results)])

    dones = terminateds | truncateds
    # expanded_dones = np.zeros_like(curr_envs_flags, dtype=bool)
    # expanded_dones[curr_envs_flags == 1] = dones

    if 'AntBulletEnv' in environment:
        rewards = [env.unwrapped.rewards[1] for env in envs]  # Distance walked

    return observations, rewards, dones


def batch_fitness_hebb(
        iteration: int,
        folder: Path,
        hebb_rule: str,
        environment: str,
        init_weights_type='uni',
        population_hebb_coeffs: list[torch.Tensor] = None,
        population_initial_weights_co: list[torch.Tensor] = None
) -> float:
    """
    Evaluate an agent 'evolved_parameters' controlled by a Hebbian network in an environment 'environment' during a lifetime.
    The initial weights are either co-evolved (if 'init_weights' == 'coevolve') along with the Hebbian coefficients or randomly sampled at each episode from the 'init_weights' distribution.
    Subsequently the weights are updated following the hebbian update mechanism 'hebb_rule'.
    Returns the episodic fitness of the agent.
    """

    population_size = len(population_hebb_coeffs)
    population_hebb_coeffs = torch.from_numpy(np.array(population_hebb_coeffs))
    print(f'Размер батча: {population_size}')
    with torch.no_grad():
        envs, pixel_env, input_dim, action_dim = spinner_and_time(lambda: make_envs(environment, population_size),
                                                                  'Инициализация среды для каждого члена популяции')
        population_policies_func = lambda: [
            init_policy_weights(
                init_weights_type,
                pixel_env,
                make_policy(pixel_env, action_dim, input_dim),
                population_initial_weights_co[i])
            for i in range(population_size)
        ]
        population_policies = spinner_and_time(population_policies_func, 'Инициализация весов моделей в популяции')
        population_weights_func = lambda: [
            [w.detach() for w in policy.parameters()]
            for policy in population_policies
        ]
        population_weights = spinner_and_time(population_weights_func, 'Выгрузка весов из моделей')
        population_weights = [torch.stack(elem) for elem in zip(*population_weights)]
        if pixel_env:
            population_weights = population_weights[2:]
        # vec_env = SyncVectorEnv([lambda: env for env in envs])
        # observations = spinner_and_time(lambda: [env.reset()[0] for env in envs], 'Получение первичных observations')
        observations = spinner_and_time(lambda: np.array([env.reset()[0] for env in envs]),
                                        'Получение первичных observations')
        # observations = observations if not pixel_env else np.swapaxes(observations, 1, 3)  # (3, 84, 84)

        # Burnout phase for the bullet quadruped so it starts off from the floor
        if 'Bullet' in environment:
            action = np.zeros(8)
            for _ in range(40):
                __ = [env.step(action) for env in envs]

        # Normalize weights flag for non-bullet envs
        normalised_weights = ('Bullet' not in environment)

        # Inner loop
        cumulative_rewards = np.zeros(population_size)
        population_indices = np.array(range(population_size))
        t = 0
        curr_envs_flags = np.ones(population_size)
        neg_count = np.zeros(population_size)
        step = 0
        neg_count_threshold = 20 if 'CarRacing' in environment else 30
        observations = adapt_observations(observations, envs, pixel_env)
        frames = [[] for _ in range(population_size)]
        while True:
            step += 1
            print(f'Шаг № {step}')

            # Выполнить шаг среды для активных сред
            policies_outputs = get_policies_outputs(population_policies, observations, environment)
            actions = calc_actions(policies_outputs, environment)

            observations, rewards, dones = make_env_step(envs, actions, environment)
            curr_frames = [env.render() for env in envs]
            for i, curr_i in zip(population_indices, range(len(curr_frames))):
                frames[i].append(curr_frames[curr_i
                                 ])
            observations = adapt_observations(observations, envs, pixel_env)

            # Добавить награды и посчитать негативные
            cumulative_rewards[population_indices] += rewards
            neg_count[population_indices] = neg_count_add(neg_count[population_indices], rewards, environment, t)

            # Обновляем флаги активных сред
            curr_envs_flags = (~(dones | (neg_count[population_indices] > neg_count_threshold))).astype(int)
            # curr_envs_flags[0] = 0
            population_indices = population_indices[curr_envs_flags.astype(bool)]
            if sum(curr_envs_flags) == 0:
                print(f'Количество активных сред: {sum(curr_envs_flags)}')
                break

            # Обновляем списки активных сред, политик этих сред и observations
            population_policies = [pp for i, pp in enumerate(population_policies) if curr_envs_flags[i]]
            observations = observations[curr_envs_flags.astype(bool)]
            envs = [env for env, flag in zip(envs, curr_envs_flags) if flag]
            policies_outputs = [p[curr_envs_flags.astype(bool)] for p in policies_outputs]
            population_hebb_coeffs = population_hebb_coeffs[curr_envs_flags.astype(bool)]
            population_weights = [p[curr_envs_flags.astype(bool)] for p in population_weights]
            get_pp = lambda: [p.detach().cpu().numpy().copy() for p in population_policies[0].parameters()]
            compare = lambda a, b: all(np.array_equal(ai, bi) for ai, bi in zip(a, b))
            pp = get_pp()
            print(f"params check eq: {compare(pp, get_pp())}")
            pp_ch = get_pp()
            pp_ch[0][0][0][0][0] = -111
            print(f"params check ineq: {compare(pp_ch, get_pp())}")
            # print(f"params check: {pp == get_pp()}")
            # Обновляем веса с помощью локальных правил Хебба
            update_func = lambda: hebbian_update(hebb_rule, population_hebb_coeffs, population_weights,
                                                 policies_outputs)
            population_weights = spinner_and_time(update_func, 'Обновление весов правил Хебба')
            print(f"Изменились ли веса после hebb_update: {not compare(pp, get_pp())}")
            pp = get_pp()
            spinner_and_time(
                lambda: update_policies_weights(pixel_env, population_policies, population_weights),
                'Обновление весов моделей'
            )
            print(f"Изменились ли веса после update_policies: {not compare(pp, get_pp())}")
            pp = get_pp()
            # for i, p in enumerate(population_weights):
            #     p[curr_envs_flags.astype(bool)] = curr_population_weights[i]
            # envs[0].render()

            print(f'Количество активных сред: {sum(curr_envs_flags)}')
            print('\n')
            # Normalise weights per layer
            if normalised_weights:
                for policy in population_policies:
                    params = list(policy.parameters())
                    indices = (0, 1, 2) if not pixel_env else (2, 3, 4)
                    for i in indices:
                        p = params[i]
                        p.data /= p.abs().max()
            print(f"Изменились ли веса после нормализации: {not compare(pp, get_pp())}")
            t += 1
        for env in envs:
            env.close()

    best_policy_index = np.argmax(cumulative_rewards)
    folder = Path(folder, 'videos')
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/best_gym_video_iter_{iteration}_rew_{cumulative_rewards[best_policy_index]:.2f}.mp4"
    writer = imageio.get_writer(path, fps=30)
    for frame in frames[best_policy_index]:
        writer.append_data(frame)
    writer.close()
    return cumulative_rewards
