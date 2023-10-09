import torch.utils.data as data
import datasets
from config import Config
cfg = Config.getInstance()["dataset"]


def get_iter(spilt):
    basic_data = datasets.get_dataset(spilt=spilt)
    dataloader = data.DataLoader(
        basic_data, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    return dataloader
