import time
from dataclasses import dataclass

from evoultion_stratetgy import EvolutionStrategyHebb


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
    print('\n\n........................................................................')
    print('\nInitilisating Hebbian ES for ' + config.environment + ' with ' + config.hebb_rule + ' Hebbian rule\n')
    es = EvolutionStrategyHebb(config)

    # Start the evolution
    print('\n........................................................................')
    print('\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n')
    tic = time.time()
    es.run(config.generations, print_step=config.print_every, path=config.folder)
    toc = time.time()
    print('\nEvolution took: ', int(toc - tic), ' seconds\n')
