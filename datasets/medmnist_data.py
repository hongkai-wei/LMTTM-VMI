from torch import nn
import torch.utils.data as data
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.data import DataLoader
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
config = Config.getInstance()
# rewrite the medmnist class

class MedMNISTDataset:
    # root文件下包含*.npz就行
    def __init__(self, dataset_name=config["dataset_name"], batch_size=config["batch_size"], split="train", transform=None, target_transform=None, download=False, root=config["root"]):
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
            root=root, split=split, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.__len__()

if __name__ == "__main__":
    from torchshow import torchshow
    dataset = MedMNISTDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (data, label) in enumerate(dataloader):
        # print(data[0].permute(1, 0, 2, 3))
        
        torchshow.show_video(data[0].permute(1, 0, 2, 3))