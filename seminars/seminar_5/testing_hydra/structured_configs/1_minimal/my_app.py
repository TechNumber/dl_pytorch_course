from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig


@dataclass
class PostgresSQLConfig:
    driver: str = 'postgresql'
    user: str = 'jieru'
    password: str = 'secret'


cs = ConfigStore.instance()
cs.store(name='postgresql', group='db', node=PostgresSQLConfig)



@hydra.main(version_base=None, config_path='conf')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    my_app()
