from config import Config


def get_dataset(split):

    config = Config.getInstance()
    if config["dataset_name"] == "organmnist3d":
        from .medmnist import MedMNISTDataset
        return MedMNISTDataset(split=split)
    # elif config["dataset_name"]