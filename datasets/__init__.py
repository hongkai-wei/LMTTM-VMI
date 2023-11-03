from config import Config
import os

def get_dataset(split, download=False, transform=None,config=Config.getInstance()):
    config = config
    if config["dataset_name"] == "organmnist3d":
        from .medmnist_data import MedMNISTDataset
        # if cannot find the root directory, then create it.
        if not os.path.exists(Config.getInstance()["root"]):
            os.makedirs(Config.getInstance()["root"])
        return MedMNISTDataset(split=split, download=download, transform=transform)
    elif config["dataset_name"] == "HMDB":
        from .hmdb51 import HMDB51Dataset
        if split == "train":
            pass
        else:
            config["root"] = os.path.join(os.path.dirname(config["root"]),"_test")
        return HMDB51Dataset(config["root"],transform)
