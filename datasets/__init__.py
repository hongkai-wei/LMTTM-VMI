from config import Config
import os

def get_dataset(split, download=False, transform=None,config=Config.getInstance("base.json")):

    config = config

    if config["dataset_name"] == "organmnist3d":
        from .medmnist_data import MedMNISTDataset
        # if cannot find the root directory, then create it.
        if not os.path.exists(Config.getInstance()["root"]):
            os.makedirs(Config.getInstance()["root"])
        return MedMNISTDataset(split=split, download=download, transform=transform)
    
    elif config["dataset_name"] == "hmdb_dataset0" or config["dataset_name"] == "hmdb_dataset1" or config["dataset_name"] == "hmdb_dataset2":
        from .hmdb_data import HMDBDataset, HMDBDataset_download
        if not os.path.exists(Config.getInstance()["root"]):
            os.makedirs(Config.getInstance()["root"])
        # HMDBDataset_download(split=split, download=download, transform=transform)
        if split == "train":
            return HMDBDataset(config["root"] + "/" + config["dataset_name"] + "_train.h5")
        if split == "val":
            return HMDBDataset(config["root"] + "/" + config["dataset_name"] + "_val.h5")
        if split == "test":
            return HMDBDataset(config["root"] + "/" + config["dataset_name"] + "_test.h5")