from config import Config
import os

def get_dataset(split, download=False, transform=None):
    config = Config.getInstance()
    if config["dataset_name"] == "organmnist3d":
        from .medmnist import MedMNISTDataset
        # if cannot find the root directory, then create it.
        if not os.path.exists(Config.getInstance()["root"]):
            os.makedirs(Config.getInstance()["root"])
        return MedMNISTDataset(split=split, download=download, transform=transform)
    # elif config["dataset_name"]
