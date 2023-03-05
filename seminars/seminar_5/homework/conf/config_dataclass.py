from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class BaseDataConfig:
    num_classes: int = MISSING
    data_dir: str = './data'
    subset_size: Optional[int] = None
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 16


@dataclass
class DTDConfig(BaseDataConfig):
    _target_: str = 'datasets.textures_dataset.DTDDataModule'
    name: str = 'Describable Textures Dataset'
    data_dir: str = './datasets/downloaded/DTD'
    num_classes: int = 47
    img_size: int = 256  # TODO: поэкспериментировать


@dataclass
class GTSRBConfig(BaseDataConfig):
    _target_: str = 'datasets.road_signs_dataset.GTSRBDataModule'
    name: str = 'German Traffic Sign Recognition Benchmark'
    data_dir: str = './datasets/downloaded/GTSRB'
    num_classes: int = 43


@dataclass
class Flowers102Config(BaseDataConfig):
    _target_: str = 'datasets.flowers_datasets.Flowers102DataModule'
    name: str = '102 Category Flower Dataset'
    data_dir: str = './datasets/downloaded/Flowers102'
    num_classes: int = 102


@dataclass
class ViTConfig:
    _target_: str = 'models.vit.ViT'
    name: str = 'Visual Transformer'
    num_classes: int = '${dataset.num_classes}'
    img_size: int = '${dataset.img_size}'
    in_chans: int = 3
    patch_size: int = 16
    embedding_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    drop_rate: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.


@dataclass
class LitClassifierConfig:
    _target_: str = 'models.lightning_classifier.LitClassifier'
    num_classes: int = '${dataset.num_classes}'
    lr: float = 1e-5
    max_lr: float = 5e-4


@dataclass
class WandBConfig:
    _target_: str = 'pytorch_lightning.loggers.WandbLogger'
    project: str = '${dataset.name} ${model.name} classification'


@dataclass
class TrainerConfig:
    max_epochs: int = MISSING
    accelerator: str = 'gpu'


defaults = [
    {'dataset': 'dtd'},
    {'model': 'vit'},
    {'module': 'lit_classifier'},
    {'logger': 'wandb'},
    {'trainer': 'default_trainer'}
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    dataset: Any = MISSING
    model: Any = MISSING
    module: Any = MISSING
    logger: Any = MISSING
    trainer: Any = MISSING
    seed: int = 17


cs = ConfigStore.instance()
cs.store(group='dataset', name='dtd', node=DTDConfig)
cs.store(group='dataset', name='gtsrb', node=GTSRBConfig)
cs.store(group='dataset', name='flowers102', node=Flowers102Config)
cs.store(group='model', name='vit', node=ViTConfig)
cs.store(group='module', name='lit_classifier', node=LitClassifierConfig)
cs.store(group='logger', name='wandb', node=WandBConfig)
cs.store(group='trainer', name='default_trainer', node=TrainerConfig)
cs.store(name='base_config', node=Config)


@hydra.main(version_base=None, config_path='./', config_name='config')
def read_config(cfg: Config):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    read_config()

# Параметры: SEED, IMG_SIZE, NUM_CLASSES (в датасет), EMBED_DIM, DEPTH, N_HEADS, DROP_RATE, QKV_BIAS,
# LR, MAX_LR, EPOCHS, BATCH_SIZE, project, name, checkpoint (path, save_top_k, monitor, mode), accelerator#

# img_size - в модель, в датамодуль или в аугментации?
# num_workers: int = 2 - перезаписать в yaml конфиге
# аугментации - проверка на None, отдельные конфиги, конфигами задаются параметры и т.д.
