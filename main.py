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


if __name__ == '__main__':
    config = Config(
        environment='CarRacing-v3',
        hebb_rule='ABCD_lr',
        popsize=200,
        lr=0.2,
        decay=0.995,
        sigma=0.1,
        init_weights='uni',
        print_every=1,
        generations=10,
        threads=1,
        folder='hebb_coeffs',
        distribution='normal'
    )
    print(f'Среда: {config.environment}')
    print(f'Правило Хебба: {config.hebb_rule}')
    print(f'Размер популяции: {config.popsize}')
    print(f'Количество поколений: {config.generations}')

    es = spinner_and_time(lambda: EvolutionStrategyHebb(config), 'Инициализация эволюционной стратегии')
    # Start the evolution
    print('\n ♪┏(°.°)┛┗(°.°)┓ Запуск эволюционной стратегии ┗(°.°)┛┏(°.°)┓ ♪ \n')
    start = time.time()
    es.run(config.generations, print_step=config.print_every, path=config.folder)
    end = time.time()
    print(f'\nЭволюция заняла: {end - start:.2f} сек\n')
