import json
import os
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from reward_function_population import show_model, evaluate


@dataclass
class ExperimentMetadata:
    params: Config
    step_times: list[float]
    min_rewards: list[float]
    max_rewards: list[float]
    mean_rewards: list[float]
    std_rewards: list[float]


def get_metadata(experiment_folder: Path):
    with open(Path(experiment_folder, 'meta_data.json'), 'r') as f:
        metadata_dict = json.load(f)
    metadata = ExperimentMetadata(
        params=Config(**metadata_dict['params']),
        step_times=metadata_dict['calc_rewards_times'],
        min_rewards=metadata_dict['min_rewards'],
        max_rewards=metadata_dict['max_rewards'],
        mean_rewards=metadata_dict['mean_rewards'],
        std_rewards=metadata_dict['std_rewards']
    )
    return metadata


def max_element_index(data: list):
    curr_max = 0
    curr_max_i = -1
    for i in range(len(data)):
        if data[i] > curr_max:
            curr_max_i = i
            curr_max = data[i]
    return curr_max_i


def get_best_iter(experiment_folder: Path):
    metadata = get_metadata(experiment_folder)
    max_rew_iter = max_element_index(metadata.mean_rewards)
    return max_rew_iter + 1


def get_best_iter_weights(experiment_folder: Path):
    best_iter = get_best_iter(experiment_folder)
    hebb_coeffs_filename = None
    cnn_weights_filename = None
    coevolve_weights_filename = None
    for filename in os.listdir(experiment_folder):
        if f'iter_{best_iter}' in filename:
            if 'hebb_coeffs' in filename:
                hebb_coeffs_filename = filename
            if 'CNN_weights' in filename:
                cnn_weights_filename = filename
            if 'coevolved_initial_weights' in filename:
                coevolve_weights_filename = filename

    hebb_coeffs = torch.load(Path(experiment_folder, hebb_coeffs_filename), weights_only=False)
    init_weights_coevolve_filename = cnn_weights_filename if cnn_weights_filename is not None else coevolve_weights_filename
    init_weights_coevolve = torch.load(Path(experiment_folder, init_weights_coevolve_filename), weights_only=False)
    return torch.from_numpy(hebb_coeffs), np.array(init_weights_coevolve)


if __name__ == '__main__':
    # CNN_weights_path = Path('experiments/CNN_best_ka_uni_200_normal')
    # hebb_coeffs_path = Path('experiments/best_ka_uni_200_normal')
    experiment_folder = Path(r"D:\Убежище\Университет\Диплом\Эксперименты\_законченные_")
    exp = 'CarRacing_ABCD_lr_uni_200_300_uniform 2025-06-07 12-34-59'
    experiment_folder = Path(experiment_folder, exp)
    print(
        f'best_iter: {get_best_iter(experiment_folder)}, best_rew: {get_metadata(experiment_folder).mean_rewards[get_best_iter(experiment_folder) - 1]}')
    hebb_coeffs, init_weights = get_best_iter_weights(experiment_folder)
    # cnn_weights = torch.load(CNN_weights_path, weights_only=False)
    # hebb_coeffs = torch.Tensor(torch.load(hebb_coeffs_path, weights_only=False))
    # show_model(
    #     'ABCD_lr', 'CarRacing-v3', init_weights_type='coevolve',
    #     hebb_coeffs=hebb_coeffs, initial_weights_co=init_weights, make_blur=True, is_coevolved=True)

    rew_for_clear = evaluate(
        hebb_rule='ABCD_lr',
        environment='CarRacing-v3',
        is_coevolved=False,
        population_hebb_coeffs=[hebb_coeffs.clone() for _ in range(100)],
        population_initial_weights_co=[init_weights.copy() for _ in range(100)]
    )

    rew_for_blur = evaluate(
        hebb_rule='ABCD_lr',
        environment='CarRacing-v3',
        is_coevolved=False,
        is_blur=True,
        population_hebb_coeffs=[hebb_coeffs.clone() for _ in range(100)],
        population_initial_weights_co=[init_weights.copy() for _ in range(100)]
    )

    print(f'clear: {sum(rew_for_clear) / 100}')
    print(f'blur: {sum(rew_for_blur) / 100}')

    # folder = Path('D:\Убежище\Университет\Диплом\Эксперименты\_законченные_')
    #
    # experiment = Path(folder, 'CarRacing_ABCD_lr_ka_uni_200_300_normal 2025-06-05 22-43-30')
    #
    # with open(Path(experiment, 'meta_data.json'), 'r') as f:
    #     metadata_dict = json.load(f)
    # metadata = ExperimentMetadata(
    #     step_times=metadata_dict['calc_rewards_times'],
    #     min_rewards=metadata_dict['min_rewards'],
    #     max_rewards=metadata_dict['max_rewards'],
    #     mean_rewards=metadata_dict['mean_rewards'],
    #     std_rewards=metadata_dict['std_rewards']
    # )
    # plt.plot(metadata.min_rewards)
    # plt.plot(metadata.max_rewards)
    # plt.plot(metadata.mean_rewards)
    #
    # plt.show()

    # print(metadata)
