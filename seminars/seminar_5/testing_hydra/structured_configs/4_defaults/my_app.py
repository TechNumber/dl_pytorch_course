from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class MySQLConfig:
    driver: str = 'mysql'
    host: str = 'localhost'
    port: int = 3306
    user: str = 'omry'
    password: str = 'secret'


@dataclass
class PostGreSQLConfig:
    driver: str = 'postgresql'
    host: str = 'localhost'
    port: int = 5432
    timeout: int = 10
    user: str = 'postgres_user'
    password: str = 'drowssap'


defaults = [
    {'db': 'mysql'}
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    db: Any = MISSING


cs = ConfigStore.instance()
cs.store(group='db', name='mysql', node=MySQLConfig)
cs.store(group='db', name='postgresql', node=PostGreSQLConfig)
cs.store(name='config', node=Config)


@hydra.main(version_base=None, config_name='config')
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    my_app()
