import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .general_video_process import *
from .general_videoimgs_dataset import *
from utils.spilt import spilt
import os
class HMDB51Dataset(GeneralImgsDataset):
    """
    A class representing the UCF101 dataset.

    NOTES:
    please download the ucf dataset and unzip it to the speical path ,like   f:/ucf101 ,for the reason of the ucf101 dataset is too large ,so we do not download it in the code.   
    About file structure:
    if u dont have the jpgs file,please run thr function in the general_video_process.py or u can run general_videoimgs_dataset.py
    ucf101
    ├── ApplyEyeMakeup
    │   ├── 01├──001.jpg 002.jpg 003.jpg ...
    │   ├── 02├──001.jpg 002.jpg 003.jpg ..
    ├── ApplyLipstick
    │   ├── 01├──001.jpg 002.jpg 003.jpg ...
    │   ├── 02├──001.jpg 002.jpg 003.jpg ..

    for other datset:
    your dataset  path
    ├── classA
    │   ├── 01   include├──001.jpg 002.jpg 003.jpg ...
    │   ├── 02   include├──001.jpg 002.jpg 003.jpg ..
    ├── classB
    │   ├── 01   include├──001.jpg 002.jpg 003.jpg ...
    │   ├── 02   include├──001.jpg 002.jpg 003.jpg ..

    Args:
    imgs_path (str): The path to the directory containing the dataset.
    transforme (optional): The transformation to apply to the dataset. Defaults to transform.
    """
    def __init__(self, imgs_path: str, transforme) -> None:
        super().__init__(imgs_path, transforme)
        if not DetectResult(path=imgs_path):
            print("the dir isnt the HMDB51 Imgs Dir,start to convert video to imgs")
            PreProcess(path=imgs_path, speicial_frames_num=32)
            PreProcessVideos2Imgs(path=imgs_path, speicial_frames_num=32)
            spilt(os.path.join(os.path.dirname(imgs_path),"imgs"),os.path.join(os.path.dirname(imgs_path),"_test"),0.3)
            print("your test imgs files dir is {}".format(os.path.join(os.path.dirname(imgs_path),"_test")))
        print("check for datset finish")


if __name__ == "__main__":
    data = HMDB51Dataset(r"F:\imgs", None)
    print("test")
