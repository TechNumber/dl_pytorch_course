import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional.classification import multiclass_accuracy


class LitClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, lr=1e-5, max_lr=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.max_lr = max_lr

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, acc = self._calc_loss_acc(batch, batch_idx)

        self.log_dict({f'train/loss': loss, f'train/acc': acc}, on_step=False, on_epoch=True, prog_bar=True,
                      logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._calc_loss_acc(batch, batch_idx)

        self.log_dict({f'val/loss': loss, f'val/acc': acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._calc_loss_acc(batch, batch_idx)

        self.log_dict({f'test/loss': loss, f'test/acc': acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = batch
        y_pred = self.model(x)
        print(y_pred, y_true)
        return {'pred': torch.argmax(y_pred, dim=-1), 'true': y_true}

    def _calc_loss_acc(self, batch, batch_idx):
        x, y_true = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y_true)
        y_pred = torch.argmax(logits, dim=-1)
        acc = multiclass_accuracy(y_pred, y_true, num_classes=self.num_classes)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.max_lr:
            scheduler = OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        else:
            return optimizer
