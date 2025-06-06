import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from evoultion_stratetgy import EvolutionStrategyHebb
from visual import spinner_and_time


@dataclass
class Config:
    environment: str
    hebb_rule: str
    popsize: int
    lr: float
    decay: float
    sigma: float
    init_weights: str
    print_every: int
    generations: int
    threads: int
    folder: str
    distribution: str
    save_videos: bool

def get_last_iter_files(folder: Path):
    max_files = {
        'CNN_weights': {'i': -1, 'path': None},
        'hebb_coeffs': {'i': -1, 'path': None}
    }
    pattern = re.compile(r'iter_(\d+)')
    for filename in os.listdir(folder):
        match = pattern.search(filename)
        if match:
            i = int(match.group(1))
            for key in max_files:
                if key in filename and i > max_files[key]['i']:
                    max_files[key] = {'i': i, 'path': Path(folder, filename)}
    return max_files
def parse_args():
    parser = argparse.ArgumentParser(description='Evolution Strategy with Hebbian Learning')

    parser.add_argument('--folder', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    folder = Path(args.folder)
    with open(Path(folder, 'meta_data.json'), 'r') as f:
        metadata = json.load(f)
    files = get_last_iter_files(folder)
    config = Config(**metadata['params'])

    cnn_weights = torch.load(files['CNN_weights']['path'], weights_only=False)
    hebb_coeffs = torch.load(files['hebb_coeffs']['path'], weights_only=False)
    start_iteration = files['hebb_coeffs']['i']
    print(f'Среда: {config.environment}')
    print(f'Правило Хебба: {config.hebb_rule}')
    print(f'Размер популяции: {config.popsize}')
    print(f'Количество поколений: {config.generations}')

    es = spinner_and_time(
        lambda: EvolutionStrategyHebb(config, start_coeffs=hebb_coeffs, start_init_weights_co=cnn_weights, start_metadata=metadata, start_iteration=start_iteration, start_folder=folder), 'Инициализация эволюционной стратегии')

    print('\n ♪┏(°.°)┛┗(°.°)┓ Запуск эволюционной стратегии ┗(°.°)┛┏(°.°)┓ ♪ \n')
    start = time.time()
    es.run(config.generations, print_step=config.print_every, path=config.folder)
    end = time.time()
    print(f'\nЭволюция заняла: {end - start:.2f} сек\n')