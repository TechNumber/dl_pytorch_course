from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig


@dataclass
class MySQLConfig:
    host: str = 'localhost'
    pork: int = 3306


cs = ConfigStore.instance()
cs.store(name='config', node=MySQLConfig)


@hydra.main(version_base=None, config_name='config')
def my_app(cfg: MySQLConfig) -> None:
    print(isinstance(cfg, DictConfig))

    if cfg.port == 80:
        print('Is this a webserver?!')


if __name__ == '__main__':
    my_app()
