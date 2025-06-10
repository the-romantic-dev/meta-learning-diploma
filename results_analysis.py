import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

from config import Config


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

if __name__ == '__main__':
    exp_folder = Path('D:\Убежище\Университет\Диплом\Эксперименты\_законченные_')
    # experiments = {
    #     "uni": 'CarRacing_ABCD_lr_uni_200_300_uniform 2025-06-07 12-34-59',
    #     "ka_uni": 'CarRacing_ABCD_lr_ka_uni_200_300_normal 2025-06-05 22-43-30',
    #     "xa_uni": 'CarRacing_ABCD_lr_xa_uni_200_300_normal 2025-06-05 22-40-03',
    #     "sparse": 'CarRacing_ABCD_lr_sparse_200_300_normal 2025-06-05 22-40-41'
    # }

    # experiments = {
    #     "popsize 100": 'CarRacing_ABCD_lr_uni_100_300_normal 2025-06-07 08-05-19',
    #     "popsize 150": 'CarRacing_ABCD_lr_uni_150_300_normal 2025-06-07 08-03-25',
    #         "popsize 200": 'CarRacing_ABCD_lr_uni_200_300_uniform 2025-06-07 12-34-59',
    #     "popsize 250": 'CarRacing_ABCD_lr_uni_250_300_normal 2025-06-05 21-37-59',
    #     "popsize 300": 'CarRacing_ABCD_lr_uni_300_300_normal 2025-06-05 22-20-59',
    #     # "popsize 300_2": 'CarRacing_ABCD_lr_uni_300_300_normal 2025-06-06 22-08-13',
    #     "popsize 350": 'CarRacing_ABCD_lr_uni_350_300_normal 2025-06-05 22-56-11',
    #     "popsize 400": 'CarRacing_ABCD_lr_uni_400_300_normal 2025-06-05 22-58-28'
    #
    # }

    experiments = {
        "A": 'CarRacing_A_uni_200_300_normal 2025-06-09 22-34-14',
        "AD": 'CarRacing_AD_uni_200_300_normal 2025-06-08 23-32-38',
        "ABC": 'CarRacing_ABC_uni_200_300_normal 2025-06-09 16-34-06',
        "ABCD_lr": 'CarRacing_ABCD_lr_uni_200_300_uniform 2025-06-07 12-34-59'
    }

    import pandas as pd

    for exp in experiments:
        metadata = get_metadata(Path(exp_folder, experiments[exp]))
        total_train_time = sum(metadata.step_times)
        hour = total_train_time // 3600
        minutes = total_train_time % 3600 // 60
        seconds = total_train_time % 60
        print(f"{exp}: {int(hour)}:{int(minutes)}:{int(seconds)} rew = {max(metadata.mean_rewards)}")

    window = 20  # размер окна сглаживания
    for exp in experiments:
        metadata = get_metadata(Path(exp_folder, experiments[exp]))
        rewards = pd.Series([mean / iter_time for mean, iter_time in zip(metadata.mean_rewards, metadata.step_times)])
        ma = rewards.rolling(window, min_periods=1).mean()
        # plt.plot(ma, label=f'{exp} mean = {max(metadata.mean_rewards):.2f}', marker='.')
        plt.plot(ma, label=f'{exp}', marker='.')
    plt.legend()
    plt.show()

    for exp in experiments:
        metadata = get_metadata(Path(exp_folder, experiments[exp]))
        rewards = pd.Series(metadata.step_times)
        ma = rewards.rolling(window, min_periods=1).mean()
        # plt.plot(ma, label=f'{exp} mean = {max(metadata.mean_rewards):.2f}', marker='.')
        plt.plot(ma, label=f'{exp}', marker='.')
    plt.legend()
    plt.show()


    for exp in experiments:
        metadata = get_metadata(Path(exp_folder, experiments[exp]))
        rewards = pd.Series(metadata.mean_rewards)
        ma = rewards.rolling(window, min_periods=1).mean()
        # plt.plot(ma, label=f'{exp} mean = {max(metadata.mean_rewards):.2f}', marker='.')
        plt.plot(ma, label=f'{exp}', marker='.')
    plt.legend()
    plt.show()


