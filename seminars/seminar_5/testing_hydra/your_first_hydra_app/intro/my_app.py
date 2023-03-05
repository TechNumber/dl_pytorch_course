import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(type(cfg))
    print(cfg)
    print(type(OmegaConf.to_yaml(cfg)))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
