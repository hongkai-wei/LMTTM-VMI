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
        hmdb_data_train_url = ["https://59-47-225-69.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-563/674b1d2c/1821686999-0/674b1d2cfe3ce816066cc22deaafea33/c-m19?v=5&t=1699953395&s=16999533950f379a5ed4726805e4e589eef1446213&r=Y5PKNW&bzc=2&bzs=313832313638363939393a35313135383232303a333139363530343132383a31383231363836393939&filename=dataset1_train.h5&x-mf-biz-cid=3a4327cf-8c73-421a-801b-710179917f6c-3dab77&auto_redirect=0&xmfcid=fe990b06-64e3-4740-bcf9-3ccc3f300427-0-9eed82220",
                            "https://59-47-225-70.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-878/dfbe920a/1821686999-0/dfbe920a084c14f53467a3fd43b7cb6e/c-m6?v=5&t=1699458639&s=1699458639281dba023a5983fda878608c3d3cdcb8&r=HU1KOF&bzc=2&bzs=313832313638363939393a35313135383035383a343138383831383031363a31383231363836393939&filename=dataset2_train.h5&x-mf-biz-cid=5baf468b-5121-43be-9170-15da49febe2a-c4937c&auto_redirect=0&xmfcid=8bd2471f-ece6-47be-891b-135b3dbd2574-0-abf611255",
                            "https://59-47-225-68.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-972/1a1e8b92/1821686999-0/1a1e8b927adb3eaa5207e1d295e7abcc/c-m12?v=5&t=1699458764&s=1699458764d933c88ece7f2a0dc4a2763b0a86bd1c&r=2GW8N8&bzc=2&bzs=313832313638363939393a35313135383036313a333333343737323830303a31383231363836393939&filename=dataset3_train.h5&x-mf-biz-cid=f38926ad-4bfe-4379-861f-2c6f01b6a6ee-6eaa77&auto_redirect=0&xmfcid=e4f4242d-67ba-456f-a80e-5ca8c4d0c47b-0-abf611255"]
        hmdb_data_val_url = ["https://59-47-225-69.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-563/674b1d2c/1821686999-0/674b1d2cfe3ce816066cc22deaafea33/c-m19?v=5&t=1699953395&s=16999533950f379a5ed4726805e4e589eef1446213&r=Y5PKNW&bzc=2&bzs=313832313638363939393a35313135383232303a333139363530343132383a31383231363836393939&filename=dataset1_train.h5&x-mf-biz-cid=3a4327cf-8c73-421a-801b-710179917f6c-3dab77&auto_redirect=0&xmfcid=fe990b06-64e3-4740-bcf9-3ccc3f300427-0-9eed82220",
                            "https://59-47-225-69.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-178/e4cea262/1821686999-0/e4cea262d467672f9d350bbe11ac60b2/c-m6?v=5&t=1699458726&s=1699458726480ec7ccd0c07d23c8a69bf81b9df6ba&r=W1BY0Z&bzc=2&bzs=313832313638363939393a35313135383036303a313339383938343430383a31383231363836393939&filename=dataset2_val.h5&x-mf-biz-cid=44b77359-fbb2-43ac-864c-5095da78a6bf-c4937c&auto_redirect=0&xmfcid=49ed4a5a-d0b7-4b4d-bba7-dfca9ff29e79-0-abf611255",
                            "https://59-47-225-70.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-94/5bcc6856/1821686999-0/5bcc685636cd9779ca8ec1517bd0e45e/c-m12?v=5&t=1699458882&s=1699458882af59316e9f91ce51a376456b68dc35ec&r=ERDTY0&bzc=2&bzs=313832313638363939393a35313135383036343a313132323434313139323a31383231363836393939&filename=dataset3_val.h5&x-mf-biz-cid=25839995-5eeb-47ec-ae43-b6686f0daf25-6eaa77&auto_redirect=0&xmfcid=7bccb3b1-d72f-43a7-a5b3-3b50202b6d55-1-abf611255"]
        hmdb_data_test_url = ["https://59-47-225-69.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-315/4ee78156/1821686999-0/4ee78156c284fdc4cb739a9cae4af93b/c-m12?v=5&t=1699448141&s=16994481418aef3942ee013ed84b1326e11061c7e7&r=E9317G&bzc=2&bzs=313832313638363939393a35313135383035343a313038373230303938343a31383231363836393939&filename=dataset1_test.h5&x-mf-biz-cid=37aff562-9d8f-4db0-8cb6-e54ccea710a8-3dab77&auto_redirect=0&xmfcid=08282669-7b90-40b0-8e28-3a9c8deeb746-0-abf611255",
                            "https://59-47-225-68.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-961/d92f004a/1821686999-0/d92f004a919827ae06429dc2355f73c1/c-m6?v=5&t=1699458695&s=16994586957d698e8fdfb04aed54950f9dfca01656&r=2S7Q7B&bzc=2&bzs=313832313638363939393a35313135383035393a313431373936313434383a31383231363836393939&filename=dataset2_test.h5&x-mf-biz-cid=d78e76b7-5b0d-4b2a-9965-ad288b61e538-584000&auto_redirect=0&xmfcid=b3c5551c-6c79-4d3f-99c7-6ae2f5e375f6-0-abf611255",
                            "https://59-47-225-68.pd1.123pan.cn:30443/download-cdn.123pan.cn/123-252/45815d8a/1821686999-0/45815d8a82c80a5490ec7e63ecb600fc/c-m12?v=5&t=1699458854&s=16994588547c02936c59220eb5489f1bf75724f203&r=1F55CA&bzc=2&bzs=313832313638363939393a35313135383036333a313133383730383435363a31383231363836393939&filename=dataset3_test.h5&x-mf-biz-cid=c4a4c7aa-9808-4571-82bf-f4e1edf31075-c4937c&auto_redirect=0&xmfcid=1021d9a8-a058-4ba3-951b-ec827ce15ad1-1-abf611255"]

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
    