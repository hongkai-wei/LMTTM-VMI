from torch.utils.data import Dataset
import os
import torch

from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class img_data(Dataset):
    '''
    A PyTorch dataset for loading image data.

    Args:
    - imgs_path (str): The path to the directory containing the image data.
    - transforme (transformer) : The transformer to apply to the image data.

    '''
    def __init__(self,imgs_path:str,transforme) -> None:
        super().__init__()
        self.labels = sorted(os.listdir(imgs_path))
        self.imgs_path =[os.path.join(imgs_path,cls,single_img)  for cls in (os.listdir(imgs_path)) for single_img in os.listdir(os.path.join(imgs_path,cls))]
        self.transforme = transforme
        self.transforme2 = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224))])
    def __getitem__(self, index):

        get_img_path = self.imgs_path[index]
        label = self.labels.index(get_img_path.split("\\")[-2])
        imgs_file = os.listdir(get_img_path)
        tensor = torch.empty((len(imgs_file),3,224,224))
        for i, image_file in enumerate(imgs_file):
            image_path = os.path.join(get_img_path, image_file)
            image = Image.open(image_path)
            image = self.transforme2(image)#for cat the tensor
            tensor[i] = image
        tensor = self.transforme(tensor)
        tensor = tensor.transpose(0, 1)
        return tensor,label

    def __len__(self):
        return len(self.imgs_path)





__all__ = ['img_data']

# if __name__ == "__main__":
#     data = img_data(r"F:\imgs")
#     print("test")
