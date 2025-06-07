from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
from gymnasium import wrappers as w, Env
import torch.nn as nn

from hebbian_update import hebbian_update
from nn_models import CNN_heb, MLP_heb
from parallels_envs import LimitedParallelEnv
from visual import spinner_and_time, sat
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame


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


def get_env_data(env_name: str):
    env = gym.make(env_name, verbose=0)
    action_dim = get_action_dim(env)
    shape = env.observation_space.shape
    is_pixel_env = len(shape) == 3
    input_dim = 3 if is_pixel_env else shape[0] if shape else env.observation_space.n
    is_fire = hasattr(env.unwrapped, 'get_action_meanings') and 'FIRE' in env.unwrapped.get_action_meanings()
    return is_pixel_env, is_fire, input_dim, action_dim


@sat('Инициализация среды для каждого члена популяции')
def make_envs(
        env_name: str, population_size: int, is_pixel_env: bool, is_fire: bool
) -> list[ScaledFloatFrame | FireEpisodicLifeEnv | Env]:
    envs = [make_env(env_name, is_pixel_env, is_fire, render_mode='rgb_array') for _ in range(population_size)]
    return envs


def make_env(
        env_name: str, is_pixel_env: bool, is_fire: bool, render_mode: str
) -> ScaledFloatFrame | FireEpisodicLifeEnv | Env:
    env = gym.make(env_name, verbose=0, render_mode=render_mode)
    if is_fire:
        env = FireEpisodicLifeEnv(env)
    if is_pixel_env:
        env = ScaledFloatFrame(w.ResizeObservation(env, (84, 84)))
    return env


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
        nn.utils.vector_to_parameters(torch.from_numpy(initial_weights_co), policy.parameters())
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


def get_population_actions(env_name: str, model_out: torch.Tensor):
    if 'CarRacing' in env_name:
        model_out = model_out.t()
        actions = torch.stack([torch.tanh(model_out[0]), torch.sigmoid(model_out[1]), torch.sigmoid(model_out[2])])
        return actions.t().numpy()
    elif 'AntBulletEnv' in env_name:
        return model_out.numpy()
    else:
        raise ValueError(f"Incorrect env_name {env_name}")


def get_action(env_name: str, model_out: torch.Tensor):
    if 'CarRacing' in env_name:
        torch.stack([torch.tanh(model_out[0]), torch.sigmoid(model_out[1]), torch.sigmoid(model_out[2])])
    elif 'AntBulletEnv' in env_name:
        return model_out.numpy()
    else:
        raise ValueError(f"Incorrect env_name {env_name}")


def neg_count_add(curr_negs: np.ndarray, rewards: np.ndarray, environment, t):
    if 'AntBulletEnv' in environment and t <= 200:
        return np.zeros(len(rewards))
    adds = (rewards < 0.0).astype(int)
    return curr_negs * (rewards < 0.0).astype(int) + adds


# @sat('Обновление весов')
def update_population_policies_weights(population_policies, population_weights):
    for i, policy in enumerate(population_policies):
        # Собираем в список тензоры с новыми весами для текущей политики
        new_weights = [p[i] for p in population_weights]
        update_policy_weights(policy, new_weights)
        # # Берём именно те параметры, которые хотим перезаписать
        # # (здесь вы, судя по коду, пропускаете первые два параметра по задумке)
        # params = list(policy.parameters())[2:]
        #
        # # Меняем значения параметров без отслеживания градиентов
        # with torch.no_grad():
        #     for p_tensor, new_w in zip(params, new_weights):
        #         # Убеждаемся, что новый тензор на том же устройстве и имеет ту же форму
        #         new_w = new_w.to(p_tensor.device)
        #         # Классический приём: пишем прямо в .data
        #         p_tensor.data.copy_(new_w)


def update_policy_weights(policy, weights):
    params = list(policy.parameters())[2:]

    # Меняем значения параметров без отслеживания градиентов
    with torch.no_grad():
        for p_tensor, new_w in zip(params, weights):
            # Убеждаемся, что новый тензор на том же устройстве и имеет ту же форму
            new_w = new_w.to(p_tensor.device)
            # Классический приём: пишем прямо в .data
            p_tensor.data.copy_(new_w)


def adapt_poppulation_observations(observations, pixel_env: bool):
    # if isinstance(envs[0].observation_space, Discrete):
    #     return [(obs == torch.arange(envs[0].observation_space.n)).float() for obs in observations]
    if pixel_env:
        return np.swapaxes(observations, 3, 1)
    return observations


def adapt_observations(observations, pixel_env: bool):
    if pixel_env:
        return np.swapaxes(observations, 2, 0)
    return observations


def get_population_policies_outputs(population_policies, observations, environment):
    # policies_outputs = [list(p([observations[i]])) for i, p in enumerate(population_policies)]
    # if 'AntBulletEnv' in environment:
    #     policies_outputs[3] = torch.tanh(policies_outputs[3])
    policies_outputs = [get_policy_outputs(p, o, environment) for p, o in zip(population_policies, observations)]
    return [torch.stack(grouped) for grouped in list(zip(*policies_outputs))]


def get_policy_outputs(policy, observations, environment, make_blur: bool = False):
    def show_tensor(tensor):
        img = TF.to_pil_image(tensor.clamp(0, 1))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    if make_blur:
        observations = blur_tensor(torch.from_numpy(observations), kernel_size=10).numpy()
    # show_tensor(ttt)
    policy_outputs = list(policy([observations]))
    if 'AntBulletEnv' in environment:
        policy_outputs[3] = torch.tanh(policy_outputs[3])
    return policy_outputs


def gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    # 1D Gaussian
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()

    # 2D Gaussian
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
    return gauss_2d


def blur_tensor(img_tensor, kernel_size=5, sigma=1.0):
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
    kernel = gaussian_kernel(kernel_size, sigma, channels=img_tensor.shape[1])
    padding = kernel_size // 2
    blurred = F.conv2d(img_tensor, kernel, groups=img_tensor.shape[1], padding=padding)
    return blurred.squeeze(0)  # back to (3, H, W)

# def calc_actions(policies_outputs, environment):
#     return get_population_actions(environment, model_out=policies_outputs[3])


# @sat('Шаг среды')
# def make_env_step(envs, actions, environment):
#     results = [env.step(action) for env, action in zip(envs, actions)]
#     observations, rewards, terminateds, truncateds, _ = list([np.array(res) for res in zip(*results)])
#     dones = terminateds | truncateds
#
#     if 'AntBulletEnv' in environment:
#         rewards = [env.unwrapped.rewards[1] for env in envs]  # Distance walked
#
#     return observations, rewards, dones


# @sat('Нормализация весов')
def normalize_population_weights(population_weights, popsize: int):
    for weight in population_weights:
        for i in range(popsize):
            weight[i] /= weight[i].abs().max()
    return population_weights


def normalize_weights(weights):
    for i in range(len(weights)):
        weights[i] /= weights[i].abs().max()
    return weights


def fitness_hebb(
        iteration: int,
        folder: Path,
        hebb_rule: str,
        environment: str,
        save_videos: bool,
        init_weights_type='uni',
        population_hebb_coeffs: list[torch.Tensor] = None,
        population_initial_weights_co: list[torch.Tensor] = None,
        is_coevolved: bool = False
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
            hebb_rule, environment, save_videos, init_weights_type,
            population_hebb_coeffs[start_index:end_index],
            population_initial_weights_co[start_index:end_index],
            is_coevolved
        )
        rewards[start_index:end_index] = rew
    return rewards


def batch_fitness_hebb(
        iteration: int,
        folder: Path,
        hebb_rule: str,
        environment: str,
        save_videos: bool,
        init_weights_type='uni',
        population_hebb_coeffs: list[torch.Tensor] = None,
        population_initial_weights_co: list[torch.Tensor] = None,
        is_coevolved: bool = False
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
        pixel_env, is_fire, input_dim, action_dim = get_env_data(environment)
        # envs = make_envs(environment, population_size, pixel_env, is_fire)
        envs = LimitedParallelEnv(environment, population_size, 20, pixel_env, is_fire)
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
        # observations = spinner_and_time(lambda: np.array([env.reset()[0] for env in envs]),
        #                                 'Получение первичных observations')
        observations = envs.reset()
        # Burnout phase for the bullet quadruped so it starts off from the floor
        if 'Bullet' in environment:
            action = np.zeros(8)
            for _ in range(40):
                envs.step(action)

        # Normalize weights flag for non-bullet envs
        normalise_weights = ('Bullet' not in environment)

        # Inner loop
        cumulative_rewards = np.zeros(population_size)
        population_indices = np.array(range(population_size))
        t = 0
        neg_count = np.zeros(population_size)
        step = 0
        neg_count_threshold = 20 if 'CarRacing' in environment else 30
        observations = adapt_poppulation_observations(observations, pixel_env)
        pbar = tqdm(desc=f"Рассчет популяции поколения {iteration + 1}")
        while True:
            step += 1
            # Выполнить шаг среды для активных сред
            policies_outputs = get_population_policies_outputs(population_policies, observations, environment)
            actions = get_population_actions(environment, model_out=policies_outputs[3])
            # calc_actions(policies_outputs, environment)
            observations, rewards, dones = envs.step(actions)
            observations = adapt_poppulation_observations(observations, pixel_env)

            # Добавить награды и посчитать негативные
            cumulative_rewards[population_indices] += rewards
            neg_count[population_indices] = neg_count_add(neg_count[population_indices], rewards, environment, t)

            # Обновляем флаги активных сред
            curr_envs_flags = (~(dones | (neg_count[population_indices] > neg_count_threshold))).astype(int)
            population_indices = population_indices[curr_envs_flags.astype(bool)]
            if sum(curr_envs_flags) == 0:
                pbar.set_postfix({'Количество активных сред': sum(curr_envs_flags)})
                pbar.update(1)
                break

            # Обновляем списки активных сред, политик этих сред и observations
            population_policies = [pp for i, pp in enumerate(population_policies) if curr_envs_flags[i]]
            observations = observations[curr_envs_flags.astype(bool)]
            envs.remove_envs(curr_envs_flags)
            policies_outputs = [p[curr_envs_flags.astype(bool)] for p in policies_outputs]
            population_hebb_coeffs = population_hebb_coeffs[curr_envs_flags.astype(bool)]
            population_weights = [p[curr_envs_flags.astype(bool)] for p in population_weights]
            if not is_coevolved:
                population_weights = hebbian_update(hebb_rule, population_hebb_coeffs, population_weights, policies_outputs)
                if normalise_weights:
                    population_weights = normalize_population_weights(population_weights, envs.n_envs)
                update_population_policies_weights(population_policies, population_weights)
            pbar.set_postfix({'Количество активных сред': sum(curr_envs_flags)})
            pbar.update(1)
            t += 1
        envs.close()
    return cumulative_rewards


def evaluate_rewards(
        hebb_rule: str,
        environment: str,
        init_weights_type='uni',
        hebb_coeffs: torch.Tensor = None,
        initial_weights_co: torch.Tensor = None,
        make_blur: bool = False
) -> float:
    """
    Evaluate an agent 'evolved_parameters' controlled by a Hebbian network in an environment 'environment' during a lifetime.
    The initial weights are either co-evolved (if 'init_weights' == 'coevolve') along with the Hebbian coefficients or randomly sampled at each episode from the 'init_weights' distribution.
    Subsequently the weights are updated following the hebbian update mechanism 'hebb_rule'.
    Returns the episodic fitness of the agent.
    """
    with torch.no_grad():
        pixel_env, is_fire, input_dim, action_dim = get_env_data(environment)
        env = make_env(environment, pixel_env, is_fire, render_mode='human')
        policy = init_policy_weights(init_weights_type, pixel_env, make_policy(pixel_env, action_dim, input_dim),
                                     initial_weights_co)
        weights = [w.detach() for w in policy.parameters()]
        # population_weights = [torch.stack(elem) for elem in zip(*population_weights)]
        if pixel_env:
            weights = weights[2:]
        observations, _ = env.reset()

        cumulative_reward = 0
        neg_count = 0
        neg_count_threshold = 20 if 'CarRacing' in environment else 30
        observations = adapt_observations(observations, pixel_env)
        while True:
            # Выполнить шаг среды для активных сред
            policy_outputs = get_policy_outputs(policy, observations, environment, make_blur)
            model_out = policy_outputs[3]
            actions = torch.stack([torch.tanh(model_out[0]), torch.sigmoid(model_out[1]), torch.sigmoid(model_out[2])])

            observations, rewards, term, trunc, info = env.step(actions.numpy())
            env.render()
            done = trunc or term
            observations = adapt_observations(observations, pixel_env)

            # Добавить награды и посчитать негативные
            cumulative_reward += rewards
            neg_count = neg_count + 1 if float(rewards) < 0.0 else 0

            # Обновляем флаги активных сред

            if done or neg_count > neg_count_threshold:
                print(f'Итоговая награда: {cumulative_reward}')
                break

            weights = hebbian_update(hebb_rule, hebb_coeffs, weights, policy_outputs)
            weights = normalize_weights(weights)
            update_policy_weights(policy, weights)
        env.close()
    return cumulative_reward
