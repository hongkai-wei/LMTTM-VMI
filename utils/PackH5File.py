import h5py
import tqdm
from torch.utils.data import Dataset
import torch

class H5Data(Dataset):
    """
    A PyTorch Dataset class for loading data from an H5 file.

    Args:
        h5pth (str): The path to the H5 file.

    Attributes:
        h5file (h5py.File): The H5 file object.
        group (h5py.Group): The group containing the data in the H5 file.

    Methods:
        __getitem__(self, index): Returns the data at the given index.
        __len__(self): Returns the number of data points in the dataset.
    """
    def __init__(self,h5pth) -> None:
        super().__init__()
        self.h5file = h5py.File(h5pth,"r")
        self.group = self.h5file["gourp_all"]
        
    def __getitem__(self, index):
        """
        Returns the data at the given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the input tensor and the target tensor.
        """
        x = self.group[f"data_{index}"]["x"][:]
        y = self.group[f"data_{index}"]["y"][()]
        x=torch.tensor(x,device="cuda",dtype=torch.float32)
        y=torch.tensor(y,device="cuda",dtype=torch.long)
        return x,y
    
    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self.group)
    
class H5Package(object):
    """
notice:
    if u have a dataset,(isnt the dataloader dataset,it is the subclass of Dataset)
    then,u can use this py to package the dataset to a h5 file,and also use this py to load the h5 file to a dataset,for acclearting the data loading

uasge:
    train_data=datasets.get_dataset(split="train", download=False, transform=None,config=config)  #get the dataset
    h5package = H5Package("dataset_train.h5")  #init the h5package,set your h5 file path
    h5package.package(train_data)  #package the dataset to h5 file
    then, u get  a H5file in your speicail path
    if u want to use this h5:
    datset = h5package.load()  #load the h5 file to a dataset





    A class for packaging data into an H5 file or unpack H5 file.

    Args:
        save_path (str): The path to save the H5 file.

    Attributes:
        path (str): The path to save the H5 file.
        data (Dataset): The dataset to package.

    Methods:
        package(self, dataset): Packages the given dataset into an H5 file.
        load(self): Loads the data from the H5 file into a PyTorch Dataset object.
    """
    def __init__(self,save_path) -> None:
        self.path=save_path 
        
    def package(self,dataset):
        """
        Packages the given dataset into an H5 file.

        Args:
            dataset (Dataset): The dataset to package.
        """
        self.data=dataset
        print("begin package")
        hdf5_file = h5py.File(self.path, 'w')
        group = hdf5_file.create_group('gourp_all')
        for i in tqdm.tqdm(range(len(dataset))):
            x=self.data[i][0]
            y=self.data[i][1]
            subgroup = group.create_group(f'data_{i}')
            subgroup.create_dataset(name="x",data=x)
            subgroup.create_dataset(name="y",data=y)
        hdf5_file.close()
        print("package done")
    
    def load(self):
        """
        Loads the data from the H5 file into a PyTorch Dataset object.

        Returns:
            H5Data: A PyTorch Dataset object containing the data from the H5 file.
        """
        return H5Data(self.path)


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])





if __name__ == "__main__":
    from config import Config
    import datasets
    from torch.utils.data import Dataset,DataLoader
    # config = Config.getInstance("base_otehr_dataset.json")
    # train_data=datasets.get_dataset(split="train", download=False, transform=transform,config=config)
    # test_data = datasets.get_dataset(split="test", download=False, transform=transform,config=config)
    # val_data = datasets.get_dataset(split="val", download=False, transform=transform,config=config)
    # h5package = H5Package("dataset2_train.h5").package(train_data)
    # h5package = H5Package("dataset2_test.h5").package(test_data)
    # h5package = H5Package("dataset2_val.h5").package(val_data)
    data = H5Package("dataset2_train.h5").load()
    for i in range(len(data)):
        print(data[i][0].shape,data[i][1])


