from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import DTD


class DTDDataModule(pl.LightningDataModule):
    def __init__(self,
                 img_size: int,
                 data_dir: str = './datasets/downloaded/DTD',
                 subset_size: Optional[int] = None,
                 batch_size: int = 8,
                 num_workers: int = 2,
                 *args, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subset_size = subset_size
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
        DTD(root=self.data_dir,
            split='train',
            download=True)
        DTD(root=self.data_dir,
            split='val',
            download=True)
        DTD(root=self.data_dir,
            split='test',
            download=True)

    def setup(self, stage: str):
        if stage == 'fit':
            self.dtd_train = DTD(
                root=self.data_dir,
                transform=self.transform,
                split='train'
            )
            if self.subset_size and self.subset_size < len(self.dtd_train):
                self.dtd_train = Subset(self.dtd_train, torch.arange(self.subset_size))
            self.dtd_val = DTD(
                root=self.data_dir,
                transform=self.transform,
                split='val'
            )
            if self.subset_size and self.subset_size < len(self.dtd_val):
                self.dtd_val = Subset(self.dtd_val, torch.arange(self.subset_size))

        if stage == 'test':
            self.dtd_test = DTD(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            )
            if self.subset_size and self.subset_size < len(self.dtd_test):
                self.dtd_test = Subset(self.dtd_test, torch.arange(self.subset_size))

        if stage == 'predict':
            self.dtd_predict = DTD(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            )
            if self.subset_size and self.subset_size < len(self.dtd_predict):
                self.dtd_predict = Subset(self.dtd_predict, torch.arange(self.subset_size))

    def train_dataloader(self):
        return DataLoader(
            self.dtd_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dtd_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dtd_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dtd_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
