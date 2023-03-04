import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import DTD


class DTDDataModule(pl.LightningDataModule):
    def __init__(self,
                 img_size: int,
                 data_dir: str = './datasets/downloaded/DTD',
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
            self.dtd_train = Subset(DTD(
                root=self.data_dir,
                transform=self.transform,
                split='train'
            ), torch.arange(1500))
            self.dtd_val = Subset(DTD(
                root=self.data_dir,
                transform=self.transform,
                split='val'
            ), torch.arange(1500))

        if stage == 'test':
            self.dtd_test = Subset(DTD(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            ), torch.arange(1500))

        if stage == 'predict':
            self.dtd_predict = Subset(DTD(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            ), torch.arange(1500))

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
