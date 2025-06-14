import json
import random
from datetime import timedelta, datetime
from pathlib import Path
import torch
import numpy as np
import multiprocessing as mp
import time
from os.path import exists
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import os
from tqdm import tqdm

from reward_function_population import fitness_hebb
from visual import spinner_and_time, sat


def compute_ranks(x):
    """
    Returns rank as a vector of len(x) with integers from 0 to len(x)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    Maps x to [-0.5, 0.5] and returns the rank
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def worker_process_hebb(arg):
    get_reward_func, hebb_rule, eng, init_weights, coeffs = arg

    wp = np.array(coeffs)
    decay = - 0.01 * np.mean(wp ** 2)
    r = get_reward_func(hebb_rule, eng, init_weights, coeffs) + decay

    return r


def worker_process_hebb_coevo(arg):
    get_reward_func, hebb_rule, eng, init_weights, coeffs, coevolved_parameters = arg

    wp = np.array(coeffs)
    decay = - 0.01 * np.mean(wp ** 2)
    r = get_reward_func(hebb_rule, eng, init_weights, coeffs, coevolved_parameters) + decay

    return r


def coefs_per_synapse_for_rule(rule: str):
    mapping = {
        'A': 1,
        'AD': 2,
        'AD_lr': 3,
        'ABC': 3,
        'ABC_lr': 4,
        'ABCD': 4,
        'ABCD_lr': 5,
        'ABCD_lr_D_out': 5,
        'ABCD_lr_D_in_and_out': 6
    }
    if rule not in mapping:
        raise ValueError('The provided Hebbian rule is not valid')
    return mapping[rule]


def get_action_dim(env):
    if isinstance(env.action_space, Box):
        return env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        return env.action_space.n
    else:
        raise ValueError('Action space not supported')


def is_pixel_env(env):
    return len(env.observation_space.shape) == 3


def check_observation_state(env):
    if (not len(env.observation_space.shape) == 3 and
            not len(env.observation_space.shape) == 1 and
            not isinstance(env.observation_space, Discrete)):
        raise ValueError('Observation space not supported')


def get_input_dim(env):
    if len(env.observation_space.shape) == 1:  # State-based environment
        return env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        return env.observation_space.n
    else:
        raise ValueError('Observation space not supported')


def get_init_coeffs(distribution: str, plastic_weights, coefficients_per_synapse):
    if distribution == 'uniform':
        return np.random.uniform(-1, 1, (plastic_weights, coefficients_per_synapse))
    elif distribution == 'normal':
        return torch.randn(plastic_weights, coefficients_per_synapse).detach().numpy().squeeze()
    else:
        raise ValueError('Unsupported destribution')


def get_init_weights(distribution: str, plastic_weights, cnn_weights):
    if distribution == 'uniform':
        return np.random.uniform(-1, 1, (cnn_weights + plastic_weights, 1))
    elif distribution == 'normal':
        return torch.randn(cnn_weights + plastic_weights, 1).detach().numpy().squeeze()
    else:
        raise ValueError('Unsupported destribution')


class EvolutionStrategyHebb(object):
    def __init__(self, config, start_coeffs=None, start_init_weights_co=None, start_iteration=None, start_metadata=None,
                 start_folder=None):
        self.config = config
        self.num_threads = mp.cpu_count() if config.threads == -1 else config.threads
        self.update_factor = config.lr / (config.popsize * config.sigma)
        self.coefficients_per_synapse = coefs_per_synapse_for_rule(config.hebb_rule)
        self.learning_rate = config.lr
        self.SIGMA = config.sigma
        self.POPULATION_SIZE = config.popsize
        self.decay = config.decay
        self.hebb_rule = config.hebb_rule
        self.init_weights = config.init_weights
        self.environment = config.environment
        # Look up observation and action space dimension
        env = gym.make(config.environment)
        check_observation_state(env)
        self.pixel_env = is_pixel_env(env)
        action_dim = get_action_dim(env)
        self.coevolve_init = (config.init_weights == 'coevolve')
        if self.coevolve_init:
            print('\nCo-evolving initial weights of the network')
        self.start_iteration = start_iteration
        self.start_metadata = start_metadata
        self.start_folder = start_folder
        cnn_weights = 1362 if self.pixel_env else 0
        inp_dim = 648 if self.pixel_env else get_input_dim(env)
        plastic_weights = ((128 * inp_dim) + (64 * 128) + (action_dim * 64))
        if start_init_weights_co is not None:
            self.initial_weights_co = start_init_weights_co
        else:
            if self.pixel_env or self.coevolve_init:
                self.initial_weights_co = get_init_weights(
                    config.distribution,
                    plastic_weights if self.coevolve_init else 0,
                    cnn_weights)

        if start_coeffs is not None:
            self.coeffs = start_coeffs
        else:
            self.coeffs = get_init_coeffs(config.distribution, plastic_weights, self.coefficients_per_synapse)
        self.get_reward = fitness_hebb

    def get_coeffs(self):
        return self.coeffs.astype(np.float32)

    def get_coevolved_parameters(self):
        return self.initial_weights_co.astype(np.float32)

    @sat('Генерация популяции')
    def _get_population(self, coevolved=False):
        weights = self.initial_weights_co if coevolved else self.coeffs
        pop_size = self.POPULATION_SIZE
        half = pop_size // 2
        pops = [
            np.concatenate([
                np.random.randn(half, *w.shape),
                -np.random.randn(half, *w.shape)
            ], axis=0)
            for w in weights
        ]
        return np.stack(pops, axis=1).astype(np.float32)

    def _get_rewards(self, population):
        def get_heb_coeffs_try(p):
            return np.array([self.coeffs[index] + self.SIGMA * i for index, i in enumerate(p)]).astype(np.float32)

        rewards = [
            self.get_reward(
                self.config.hebb_rule,
                self.config.environment,
                self.config.init_weights,
                get_heb_coeffs_try(p)
            )
            for p in population
        ]
        return np.array(rewards).astype(np.float32)

    def _get_rewards_coevolved(self, iteration: int, folder: Path, population: np.ndarray,
                               population_coevolved: np.ndarray):
        def _get_params_try(w: np.ndarray, p: np.ndarray) -> np.ndarray:
            return w + self.SIGMA * p

        heb_coeffs_tries = spinner_and_time(lambda: [_get_params_try(self.coeffs, p) for p in population],
                                            'Генерация heb_coeffs')
        coevolved_parameters_tries = spinner_and_time(
            lambda: [_get_params_try(self.initial_weights_co, p) for p in population_coevolved],
            'Генерация coevolved params')
        rewards = self.get_reward(iteration, folder,
                                  self.config.hebb_rule, self.config.environment, self.config.save_videos,
                                  self.config.init_weights,
                                  heb_coeffs_tries, coevolved_parameters_tries, is_coevolved=self.coevolve_init)

        rewards = np.array(rewards).astype(np.float32)
        return rewards

    def _update_coeffs(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')

        rewards = (rewards - rewards.mean()) / std
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.coeffs += self.update_factor * np.tensordot(rewards, population, axes=(0, 0))

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        if self.SIGMA > 0.01:
            self.SIGMA *= 0.999

    def _update_coevolved_param(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')

        rewards = (rewards - rewards.mean()) / std
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.initial_weights_co += self.update_factor * np.tensordot(rewards, population, axes=(0, 0))

    def run(self, iterations, print_step=10, path='/content/hebb_coeffs'):
        curr_datetime = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
        experiment_folder_name = f'{self.environment.split("-")[0]}_{self.hebb_rule}_{self.init_weights}_{self.POPULATION_SIZE}_{iterations}_{self.config.distribution} {curr_datetime}'

        if self.start_folder is not None:
            folder = self.start_folder
        else:
            folder = Path(path, experiment_folder_name)
            if not exists(folder):
                os.makedirs(folder, exist_ok=True)
        print(f'Папка: {folder.name}')
        print("CUDA доступна:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Имя устройства:", torch.cuda.get_device_name(0))
            print("Количество доступных устройств:", torch.cuda.device_count())
        print(f'Запуск от {curr_datetime}\n\n{"." * 72}\n')
        if self.start_iteration is not None:
            print(f"Продолжение эволюции с итерации {self.start_iteration}")
        generations_rewards = []
        if self.start_metadata is not None:
            meta_data = self.start_metadata
        else:
            meta_data = {
                'params': self.config.__dict__,
                'min_rewards': [],
                'mean_rewards': [],
                'max_rewards': [],
                'std_rewards': [],
                'calc_rewards_times': []
            }
        start_iteration = 0 if self.start_iteration is None else self.start_iteration
        for iteration in range(start_iteration,
                               iterations):  # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864
            start = time.time()
            population = self._get_population()  # Sample normal noise:         Step 5
            # Evolution of Hebbian coefficients & coevolution of cnn parameters and/or initial weights

            if self.pixel_env or self.coevolve_init:
                population_coevolved = self._get_population(coevolved=True)  # Sample normal noise:         Step 5
                rew_start = time.time()
                rewards = self._get_rewards_coevolved(iteration, folder, population,
                                                      population_coevolved)  # Compute population fitness:  Step 6
                rew_end = time.time()
                self._update_coeffs(rewards, population)  # Update coefficients:         Steps 8->12
                self._update_coevolved_param(rewards, population_coevolved)  # Update coevolved parameters: Steps 8->12
            else:
                rew_start = time.time()
                rewards = self._get_rewards(population)  # Compute population fitness:  Step 6
                rew_end = time.time()
                self._update_coeffs(rewards, population)

            meta_data['min_rewards'].append(float(rewards.min()))
            meta_data['max_rewards'].append(float(rewards.max()))
            meta_data['mean_rewards'].append(float(rewards.mean()))
            meta_data['std_rewards'].append(float(rewards.std()))
            meta_data['calc_rewards_times'].append(float(rew_end - rew_start))
            # Update coefficients:         Steps 8->12

            with open(Path(folder, 'meta_data.json'), 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)

            # Print fitness and save Hebbian coefficients and/or Coevolved / CNNs parameters
            rew_ = rewards.mean()
            end = time.time()
            diff = end - start
            print(
                f'Итерация {iteration + 1} | Награда (средняя): {rew_:.2f} | Награда (макс.): {rewards.max():.2f} | Время: {int(diff) // 3600:02}:{int(diff) % 3600 // 60:02}:{int(diff) % 60}:02')
            print(
                f'update_factor: {self.update_factor}  lr: {self.learning_rate} | sum_coeffs: {int(np.sum(self.coeffs))} sum_abs_coeffs: {int(np.sum(abs(self.coeffs)))}')
            print('--------------------------------')
            self.save(iteration + 1, folder, rew_)
            generations_rewards.append(rew_)

    def save(self, iteration: int, folder: Path, reward):
        postfix = f'iter_{iteration}_rew_{int(reward)}'
        torch.save(self.get_coeffs(), Path(folder, f'hebb_coeffs_{postfix}'))
        if self.coevolve_init:
            torch.save(self.get_coevolved_parameters(), Path(folder, f'coevolved_initial_weights_{postfix}'))
        elif self.pixel_env:
            torch.save(self.get_coevolved_parameters(), Path(folder, f'CNN_weights_{postfix}'))
