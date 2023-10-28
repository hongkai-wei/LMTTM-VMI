import torch.utils.data as data
import datasets
from config import Config
import torch

def get_dataloader(split,config, download=False, transform=None):
    basic_data = datasets.get_dataset(split=split, download=download, transform=transform)
    seed = 42
    torch.manual_seed(seed)
    dataloader = data.DataLoader(
        basic_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    return dataloader