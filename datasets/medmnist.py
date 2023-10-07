import medmnist
from torch import nn
from torch.utils.data import DataLoader

# rewrite the medmnist class
class MedMNISTDataset:
    def __init__(self, dataset_name, root, split, transform=None, target_transform=None, download=False):
        self.dataset_name = dataset_name
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = getattr(medmnist, dataset_name)(root=root, split=split, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)