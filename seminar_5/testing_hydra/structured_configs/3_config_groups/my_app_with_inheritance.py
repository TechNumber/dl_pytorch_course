from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING


@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = MISSING
    driver: str = MISSING


@dataclass
class MySQLConfig(DBConfig):
    driver: str = 'mysql'
    port: int = 3306


@dataclass
class PostGreSQLConfig(DBConfig):
    driver: str = 'postgresql'
    port: int = 5432
    timeout: int = 10


@dataclass
class Config:
    db: DBConfig


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(group='db', name='mysql', node=MySQLConfig)
cs.store(group='db', name='postgresql', node=PostGreSQLConfig)


@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
