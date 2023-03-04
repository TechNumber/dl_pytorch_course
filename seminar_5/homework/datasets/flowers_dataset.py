import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102


class Flowers102DataModule(pl.LightningDataModule):
    def __init__(self,
                 img_size: int,
                 data_dir: str = './datasets/downloaded/Flowers102',
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
        Flowers102(root=self.data_dir,
                   split='train',
                   download=True)
        Flowers102(root=self.data_dir,
                   split='val',
                   download=True)
        Flowers102(root=self.data_dir,
                   split='test',
                   download=True)

    def setup(self, stage: str):
        if stage == 'fit':
            self.flowers_train = Flowers102(
                root=self.data_dir,
                transform=self.transform,
                split='train'
            )
            self.flowers_val = Flowers102(
                root=self.data_dir,
                transform=self.transform,
                split='val'
            )

        if stage == 'test':
            self.flowers_test = Flowers102(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            )

        if stage == 'predict':
            self.flowers_predict = Flowers102(
                root=self.data_dir,
                transform=self.min_transform,
                split='test'
            )

    def train_dataloader(self):
        return DataLoader(
            self.flowers_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.flowers_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.flowers_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.flowers_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
