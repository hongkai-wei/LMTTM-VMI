import torch.utils.data as data
import datasets
from config import Config
config = Config.getInstance()


def get_iter(split):
    basic_data = datasets.get_dataset(split=split)
    dataloader = data.DataLoader(
        basic_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    return dataloader