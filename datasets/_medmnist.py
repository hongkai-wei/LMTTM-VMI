from torch import nn
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.data import DataLoader
from config import Config
cfg = Config.getInstance()["dataset"]
# rewrite the medmnist class


class MedMNISTDataset:
    # root文件下包含*.npz就行
    def __init__(self, dataset_name="organmnist3d", batch_size=cfg["batch_size"], spilt="train", transform=None, target_transform=None, download=False, root=cfg["root"]):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        info = INFO[self.dataset_name]
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        detal_dataset = info["python_class"]
        self.dataset = getattr(medmnist, detal_dataset)(
            root=root, split=spilt, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.__len__()
