import argparse
import random
import time
from dataclasses import dataclass
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evolution Strategy with Hebbian Learning')

    parser.add_argument('--environment', type=str, default='CarRacing-v3')
    parser.add_argument('--hebb_rule', type=str, default='ABCD_lr')
    parser.add_argument('--popsize', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--decay', type=float, default=0.995)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--init_weights', type=str, choices=['uni', 'normal'], default='uni')
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--generations', type=int, default=300)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--folder', type=str, default='hebb_coeffs')
    parser.add_argument('--distribution', type=str, choices=['normal', 'cauchy'], default='normal')
    parser.add_argument('--save_videos', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config(**vars(args))

    print(f'Среда: {config.environment}')
    print(f'Правило Хебба: {config.hebb_rule}')
    print(f'Размер популяции: {config.popsize}')
    print(f'Количество поколений: {config.generations}')

    random.seed(42)
    es = spinner_and_time(lambda: EvolutionStrategyHebb(config), 'Инициализация эволюционной стратегии')

    print('\n ♪┏(°.°)┛┗(°.°)┓ Запуск эволюционной стратегии ┗(°.°)┛┏(°.°)┓ ♪ \n')
    start = time.time()
    es.run(config.generations, print_step=config.print_every, path=config.folder)
    end = time.time()
    print(f'\nЭволюция заняла: {end - start:.2f} сек\n')