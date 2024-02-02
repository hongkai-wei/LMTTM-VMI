import torch.utils.data as data
import datasets
from config import Config
import torch

def get_dataloader(split,config, download=False, transform=None):
    basic_data = datasets.get_dataset(split=split, download=download, transform=transform,config=config)
    dataloader = data.DataLoader(
        basic_data, batch_size=config["batch_size"], num_workers=0, drop_last=True, shuffle=True)
    return dataloader

