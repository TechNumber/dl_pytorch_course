import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import wandb
from datasets.textures_dataset import DTDDataModule
from models.lightning_classifier import LitClassifier
from models.vit import ViT
from hydra.utils import instantiate
from conf.config_dataclass import Config


def train(cfg: Config):
    SEED = 17

    IMG_SIZE = 160
    NUM_CLASSES = 47
    EMBED_DIM = 768
    DEPTH = 2
    N_HEADS = 8
    DROP_RATE = 0.3
    QKV_BIAS = False

    LR = 1e-4
    MAX_LR = 5e-4
    EPOCHS = 200
    BATCH_SIZE = 10

    seed_everything(cfg.seed, workers=True)

    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    # model = LitClassifier(vit, num_classes=NUM_CLASSES, lr=LR, max_lr=MAX_LR)
    module = instantiate(cfg.module, model)
    logger = instantiate(cfg.logger)
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/", save_top_k=2, monitor='val/acc', mode='max')
    lr_monitor_callback = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback]
        # deterministic=True
    )
    trainer.fit(module, dataset)
    wandb.finish()


@hydra.main(version_base=None, config_path='conf', config_name='config')
def train_model(cfg: Config) -> None:
    # config preprocessing
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


# Параметры: SEED, IMG_SIZE, NUM_CLASSES (в датасет), EMBED_DIM, DEPTH, N_HEADS, DROP_RATE, QKV_BIAS,
# LR, MAX_LR, EPOCHS, BATCH_SIZE, project, name, checkpoint (path, save_top_k, monitor, mode), accelerator#

if __name__ == '__main__':
    train_model()
