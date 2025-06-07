import json
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from reward_function_population import evaluate_rewards


@dataclass
class ExperimentMetadata:
    step_times: list[float]
    min_rewards: list[float]
    max_rewards: list[float]
    mean_rewards: list[float]
    std_rewards: list[float]


def get_metadata(experiment_folder):
    with open(Path(experiment_folder, 'meta_data.json'), 'r') as f:
        metadata_dict = json.load(f)
    metadata = ExperimentMetadata(
        step_times=metadata_dict['calc_rewards_times'],
        min_rewards=metadata_dict['min_rewards'],
        max_rewards=metadata_dict['max_rewards'],
        mean_rewards=metadata_dict['mean_rewards'],
        std_rewards=metadata_dict['std_rewards']
    )
    return metadata


if __name__ == '__main__':
    CNN_weights_path = Path('experiments/CNN_best_ka_uni_200_normal')
    hebb_coeffs_path = Path('experiments/best_ka_uni_200_normal')
    cnn_weights = torch.load(CNN_weights_path, weights_only=False)
    hebb_coeffs = torch.Tensor(torch.load(hebb_coeffs_path, weights_only=False))
    evaluate_rewards('ABCD_lr', 'CarRacing-v3', hebb_coeffs=hebb_coeffs, initial_weights_co=cnn_weights)
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
