import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB


class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self,
                 img_size: int,
                 data_dir: str = './datasets/downloaded/GTSRB',
                 batch_size: int = 8,
                 num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            #             transforms.RandomRotation(degrees=(-30, 30))
        ])
        self.min_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size))
        ])

    def prepare_data(self):
        GTSRB(root=self.data_dir,
              split='train',
              download=True)
        GTSRB(root=self.data_dir,
              split='test',
              download=True)

    def setup(self, stage: str):
        if stage == 'fit':
            self.gtsrb_train = Subset(GTSRB(
                root=self.data_dir,
                transform=self.transform,
                split='train'
            ), torch.arange(1500))
            self.gtsrb_val = Subset(GTSRB(
                root=self.data_dir,
                transform=self.transform,
                split='test'
            ), torch.arange(1500))

        if stage == 'test':
            self.gtsrb_test = Subset(GTSRB(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            ), torch.arange(1500))

        if stage == 'predict':
            self.gtsrb_predict = Subset(GTSRB(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            ), torch.arange(1500))

    def train_dataloader(self):
        return DataLoader(
            self.gtsrb_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.gtsrb_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.gtsrb_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.gtsrb_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
