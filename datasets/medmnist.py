from torch import nn
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.data import DataLoader

# rewrite the medmnist class
class MedMNISTDataset:
    def __init__(self, dataset_name, data_flag, root, batch_size, transform=None, target_transform=None, download=True):
        self.dataset_name = dataset_name
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.getDatasets()
        # self.dataset = getattr(medmnist, dataset_name)(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def getDatasets(self):
        data_flag = self.data_flag
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        train_dataset = DataClass(split='train', root=self.root, download=self.download)
        val_dataset = DataClass(split='val', root=self.root, download=self.download)
        test_dataset = DataClass(split='test', root=self.root, download=self.download)

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dataset = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader, val_dataloader, test_dataset, task, n_channels, n_classes
