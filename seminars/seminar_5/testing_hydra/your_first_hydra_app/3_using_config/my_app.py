from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path='', config_name='config')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.node.loompa == 10
    assert cfg['node']['loompa'] == 10

    assert cfg.node.zippity == 10
    assert isinstance(cfg.node.zippity, int)
    assert cfg.node.do == "oompa 10"

    print(cfg.node.waldo)


if __name__ == '__main__':
    my_app()
