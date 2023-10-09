from config import Config


def get_dataset(spilt):

    config = Config.getInstance()["dataset"]
    if config["dataset_name"] == "organmnist3d":
        from ._medmnist import MedMNISTDataset
        return MedMNISTDataset(spilt=spilt)
    # elif config["dataset_name"]
