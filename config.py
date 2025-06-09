from dataclasses import dataclass


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