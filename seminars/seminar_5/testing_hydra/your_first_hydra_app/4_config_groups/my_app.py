from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path='conf')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    my_app()
