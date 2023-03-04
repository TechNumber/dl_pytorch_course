import pytorch_lightning as pl
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets.textures_dataset import DTDDataModule
from models.lightning_classifier import LitClassifier
from models.vit import ViT


if __name__ == '__main__':
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

    seed_everything(SEED, workers=True)

    dtd = DTDDataModule(img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=16)
    vit = ViT(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=N_HEADS,
        qkv_bias=QKV_BIAS,
        drop_rate=DROP_RATE
    )
    model = LitClassifier(vit, num_classes=NUM_CLASSES, lr=LR, max_lr=MAX_LR)
    wandb_logger = WandbLogger(project="dtd_classification", name='vit_pos_emb_trainable_weights')
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/", save_top_k=2, monitor='val/acc', mode='max')
    lr_monitor_callback = LearningRateMonitor()
    trainer = pl.Trainer(max_epochs=EPOCHS,
                         accelerator='gpu',
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, lr_monitor_callback]
                         #                      deterministic=True
                         )
    trainer.fit(model, dtd)
    wandb.finish()
