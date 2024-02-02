import os
import random
import shutil
def spilt(source_dir:str, target_dir1:str, percentage1, target_dir2:str, percentage2,traget_dir3:str,percentage3):
    '''
    source_dir: the source dir, like f:/imgs
    target_dir1: the target dir1, like f:/imgs/train
    percentage1: the percentage of the target dir1, like 0.7
    target_dir2: the target dir2, like f:/imgs/val
    percentage2: the percentage of the target dir2, like 0.2
    traget_dir3: the target dir3, like f:/imgs/test
    percentage3: the percentage of the target dir3, like 0.1
    '''
    if os.path.exists(os.path.dirname(target_dir1)) == False:
        os.mkdir(os.path.dirname(target_dir1))
    if os.path.exists(target_dir1) == False:
        os.mkdir(target_dir1)
    if os.path.exists(target_dir2) == False:
        os.mkdir(target_dir2)
    if os.path.exists(traget_dir3) == False:
        os.mkdir(traget_dir3)
    sub_dirs = os.listdir(source_dir)#imgs/classes
    for _ in range(len(sub_dirs)):
        subdir = sub_dirs[_]#img/classs1
        imgs = os.listdir(os.path.join(source_dir, subdir))
        random.shuffle(imgs)
        train_imgs = imgs[0:int(len(imgs) * percentage1)]
        val_imgs = imgs[int(len(imgs) * percentage1):int(len(imgs) * (percentage1 + percentage2))]
        test_imgs = imgs[int(len(imgs) * (percentage1 + percentage2)):len(imgs)]
        for img in train_imgs:
            shutil.move(os.path.join(source_dir, subdir, img), os.path.join(target_dir1, subdir, img))
        for img in val_imgs:
            shutil.move(os.path.join(source_dir, subdir, img), os.path.join(target_dir2, subdir, img))
        for img in test_imgs:
            shutil.move(os.path.join(source_dir, subdir, img), os.path.join(traget_dir3, subdir, img)) 


if __name__ == "__main__":
    pass
    # spilt(r"F:\imgs",r"F:\imgs1\train",0.6,r"F:\imgs1\val",0.2,r"F:\imgs1\test",0.2)