import torch.utils.data as data
import datasets
from config import Config
config = Config.getInstance()


def get_dataloader(split, download=False, transform=None):
    basic_data = datasets.get_dataset(split=split, download=download, transform=transform)
    dataloader = data.DataLoader(
        basic_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    return dataloader