import os
import requests
from tqdm import tqdm
import torch
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
config = Config.getInstance()

class HMDBDataset_download:
    def __init__(self, split, download=False, transform=None):
        self.split = split
        self.download = download
        self.transform = transform
        self.config = config
        self.dataset = self.download_data(split)

    def download_data(self, split):
        def download_from_yun(url, save_path):
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(save_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            progress_bar.close()

        hmdb_data_train = ["dataset0_train", "dataset1_train", "dataset2_train"]
        hmdb_data_val = ["dataset0_val", "dataset1_val", "dataset2_val"]
        hmdb_data_test = ["dataset0_test", "dataset1_test", "dataset2_test"]
        hmdb_data_train_url = ["0"]
        hmdb_data_val_url = ["1"]
        hmdb_data_test_url = ["2"]
        if "HMDB_dataset0":
            if split == "train":
                name = hmdb_data_train[0]
                file_url = hmdb_data_train_url[0]
            elif split == "val":
                name = hmdb_data_val[0]    
                file_url = hmdb_data_val_url[0]
            elif split == "test":
                name = hmdb_data_test[0]
                file_url = hmdb_data_test_url[0]

            save_directory = './datasets_data/HMDB/dataset0'  # 替换为您希望保存文件的目录

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name = name + f".h5"
            save_path = os.path.join(save_directory, file_name)

            if not os.path.exists(save_path):
                download_from_yun(file_url, save_path)
                print(f"文件已保存到：{save_path}")
            else:
                print(f"文件已存在：{save_path}")

        elif "HMDB_dataset1":
            if split == "train":
                name = hmdb_data_train[1]
                file_url = hmdb_data_train_url[1]
            elif split == "val":
                name = hmdb_data_val[1]    
                file_url = hmdb_data_val_url[1]
            elif split == "test":
                name = hmdb_data_test[1]
                file_url = hmdb_data_test_url[1]

            save_directory = './datasets_data/HMDB51/dataset0'  # 替换为您希望保存文件的目录

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name = name + f".h5"
            save_path = os.path.join(save_directory, file_name)

            if not os.path.exists(save_path):
                download_from_yun(file_url, save_path)
                print(f"文件已保存到：{save_path}")
            else:
                print(f"文件已存在：{save_path}")

        if "HMDB_dataset2":
            if split == "train":
                name = hmdb_data_train[2]
                file_url = hmdb_data_train_url[2]
            elif split == "val":
                name = hmdb_data_val[2]    
                file_url = hmdb_data_val_url[2]
            elif split == "test":
                name = hmdb_data_test[2]
                file_url = hmdb_data_test_url[2]

            save_directory = './datasets_data/HMDB51/dataset0'  # 替换为您希望保存文件的目录

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name = name + f".h5"
            save_path = os.path.join(save_directory, file_name)

            if not os.path.exists(save_path):
                download_from_yun(file_url, save_path)
                print(f"文件已保存到：{save_path}")
            else:
                print(f"文件已存在：{save_path}")


class HMDBDataset:
    def __init__(self,h5pth) -> None:
        super().__init__()
        self.h5file = h5py.File(h5pth,"r")
        self.group = self.h5file["gourp_all"]
        
    def __getitem__(self, index):
        x = self.group[f"data_{index}"]["x"][:]
        y = self.group[f"data_{index}"]["y"][()]
        x=torch.tensor(x,device="cuda",dtype=torch.float32)
        y=torch.tensor(y,device="cuda",dtype=torch.long)
        return x,y
    
    def __len__(self):
        return len(self.group)
    