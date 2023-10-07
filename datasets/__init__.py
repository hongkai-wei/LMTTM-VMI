from config import Config

def get_dataset(download=False):
    config = Config.getInstance()["dataset"]
    if config["dataset_name"] == "medmnist":
        from .medmnist import MedMNISTDataset
        return MedMNISTDataset(config["dataset_name"], config["data_root"], config["split"], download=download)
    elif config["dataset_name"]