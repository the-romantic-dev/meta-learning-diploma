from datetime import datetime
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

from reward_function import fitness_hebb


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
    def __init__(self, config):
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

        if self.pixel_env:
            cnn_weights = 1362  # CNN: (6, 3, 3, 3) + (8, 6, 5, 5) = 162+1200 = 1362
            # Hebbian coefficients: MLP x coefficients_per_synapse : plastic_weights x coefficients_per_synapse
            plastic_weights = (128 * 648) + (64 * 128) + (action_dim * 64)
            self.coeffs = get_init_coeffs(config.distribution, plastic_weights, self.coefficients_per_synapse)
            self.initial_weights_co = get_init_weights(config.distribution,
                                                       plastic_weights if self.coevolve_init else 0, cnn_weights)
        else:

            plastic_weights = (128 * get_input_dim(env)) + (64 * 128) + (
                    action_dim * 64)  # Hebbian coefficients:  MLP x coefficients_per_synapse :plastic_weights x coefficients_per_synapse
            self.coeffs = get_init_coeffs(config.distribution, plastic_weights, self.coefficients_per_synapse)
            if self.coevolve_init:
                self.initial_weights_co = get_init_weights(config.distribution, plastic_weights, 0)

        # Load fitness function for the selected environment
        self.get_reward = fitness_hebb

    def _get_params_try(self, w, p):
        param_try = [w[index] + self.SIGMA * i for index, i in enumerate(p)]
        res = np.array(param_try).astype(np.float32)
        return res

    def get_coeffs(self):
        return self.coeffs.astype(np.float32)

    def get_coevolved_parameters(self):
        return self.initial_weights_co.astype(np.float32)

    def _get_population(self, coevolved_param=False):
        population = []
        weights = self.initial_weights_co if coevolved_param else self.coeffs
        for i in range(int(self.POPULATION_SIZE / 2)):
            x = [np.random.randn(*w.shape) for w in weights]  # j: (coefficients_per_synapse, 1) eg. (5,1)
            x2 = [-np.random.randn(*w.shape) for w in
                  weights]  # x: (coefficients_per_synapse, number of synapses) eg. (92690, 5)
            population.append(
                x)  # population : (population size, coefficients_per_synapse, number of synapses), eg. (10, 92690, 5)
            population.append(x2)
        return np.array(population).astype(np.float32)

    def _get_rewards(self, pool, population):
        def get_heb_coeffs_try(p):
            return np.array([self.coeffs[index] + self.SIGMA * i for index, i in enumerate(p)]).astype(np.float32)

        if pool is not None:
            worker_args = [
                (self.get_reward,
                 self.config.hebb_rule,
                 self.config.environment,
                 self.config.init_weights,
                 get_heb_coeffs_try(p)
                 )
                for p in population]
            rewards = pool.map(worker_process_hebb, worker_args)
        else:
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

    def _get_rewards_coevolved(self, pool, population, population_coevolved):
        if pool is not None:
            worker_args = []
            for z in tqdm(
                    range(len(population)),
                    desc=f'Обработка популяции'):
                heb_coeffs_try = np.array(
                    [self.coeffs[index] + self.SIGMA * i for index, i in enumerate(population[z])]
                ).astype(np.float32)

                coevolved_parameters_try = np.array(
                    [self.initial_weights_co[index] + self.SIGMA * i for index, i in enumerate(population_coevolved[z])]
                ).astype(np.float32)

                worker_args.append((
                    self.get_reward, self.config.hebb_rule, self.config.environment, self.config.init_weights,
                    heb_coeffs_try,
                    coevolved_parameters_try))
            print('Начало параллельной обработки')
            start_time = time.time()
            rewards = pool.map(worker_process_hebb_coevo, worker_args)
            end_time = time.time()
            print(f'Время обработки: {(end_time - start_time):.2f} секунд')

        else:
            rewards = []
            for z in tqdm(
                    range(len(population)),
                    desc=f'Обработка популяции'):
                # print(f'Обработка элемента популяции {z + 1} из {len(population)}')
                heb_coeffs_try = np.array(self._get_params_try(self.coeffs, population[z])).astype(np.float32)
                coevolved_parameters_try = np.array(
                    self._get_params_try(self.initial_weights_co, population_coevolved[z])).astype(np.float32)
                rewards.append(self.get_reward(self.config.hebb_rule, self.config.environment, self.config.init_weights,
                                               heb_coeffs_try,
                                               coevolved_parameters_try))

        rewards = np.array(rewards).astype(np.float32)
        return rewards

    def _update_coeffs(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')

        rewards = (rewards - rewards.mean()) / std
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        for index, c in enumerate(self.coeffs):
            layer_population = np.array([p[index] for p in population])

            self.coeffs[index] = c + self.update_factor * np.dot(layer_population.T, rewards).T

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        # Decay sigma
        if self.SIGMA > 0.01:
            self.SIGMA *= 0.999

    def _update_coevolved_param(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')

        rewards = (rewards - rewards.mean()) / std

        for index, w in enumerate(self.initial_weights_co):
            layer_population = np.array([p[index] for p in population])

            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.initial_weights_co[index] = w + self.update_factor * np.dot(layer_population.T, rewards).T

    def run(self, iterations, print_step=10, path='/content/hebb_coeffs'):

        timestamp = time.time()
        dt_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H-%M-%S')
        experiment_folder_name = f'{self.environment.split("-")[0]}_{self.hebb_rule}_{self.init_weights}_{self.POPULATION_SIZE}_{iterations}_{self.config.distribution} {dt_str}'
        folder = Path(path, experiment_folder_name)
        if not exists(folder):
            os.makedirs(folder, exist_ok=True)

        print("CUDA доступна:", torch.cuda.is_available())
        # Если CUDA доступна, вывод информации о видеокарте
        if torch.cuda.is_available():
            print("Имя устройства:", torch.cuda.get_device_name(0))
            print("Количество доступных устройств:", torch.cuda.device_count())
        print(f'Запуск от {dt_str}\n\n{"." * 72}\n')

        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None

        generations_rewards = []

        for iteration in range(iterations):  # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864
            population = self._get_population()  # Sample normal noise:         Step 5
            # Evolution of Hebbian coefficients & coevolution of cnn parameters and/or initial weights
            if self.pixel_env or self.coevolve_init:
                population_coevolved = self._get_population(coevolved_param=True)  # Sample normal noise:         Step 5
                rewards = self._get_rewards_coevolved(pool, population,
                                                      population_coevolved)  # Compute population fitness:  Step 6
                self._update_coeffs(rewards, population)  # Update coefficients:         Steps 8->12
                self._update_coevolved_param(rewards, population_coevolved)  # Update coevolved parameters: Steps 8->12

            # Evolution of Hebbian coefficients
            else:
                rewards = self._get_rewards(pool, population)  # Compute population fitness:  Step 6
                self._update_coeffs(rewards, population)  # Update coefficients:         Steps 8->12

            # Print fitness and save Hebbian coefficients and/or Coevolved / CNNs parameters
            if (iteration + 1) % print_step == 0:
                rew_ = rewards.mean()
                print('iter %4i | reward: %3i |  update_factor: %f  lr: %f | sum_coeffs: %i sum_abs_coeffs: %4i' % (
                    iteration + 1, rew_, self.update_factor, self.learning_rate, int(np.sum(self.coeffs)),
                    int(np.sum(abs(self.coeffs)))), flush=True)

                if rew_ > 0:
                    self.save(folder, rew_)
                generations_rewards.append(rew_)
                # np.save(path + "/" + dt_str + '/Fitness_values_' + dt_str + '_' + self.environment + '.npy',
                #         np.array(generations_rewards).astype(np.float32))

        if pool is not None:
            pool.close()
            pool.join()

    def save(self, folder: Path, reward):
        torch.save(self.get_coeffs(), Path(folder, f'hebb_coeffs_rew_{int(reward)}'))
        if self.coevolve_init:
            torch.save(self.get_coevolved_parameters(), Path(folder, f'coevolved_initial_weights_rew_{int(reward)}'))
        elif self.pixel_env:
            torch.save(self.get_coevolved_parameters(), Path(folder, f'CNN_weights_rew_{int(reward)}'))
