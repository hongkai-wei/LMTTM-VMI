import os
from torchvision.io import read_video
from tqdm import tqdm
import torch
import PIL
import torchvision.transforms as transforms
def PreProcess(path:str,speicial_frames_num:int=16):
    del_list = []
    sub_dirs = os.listdir(path)# video classes dir
    for _ in tqdm(range(len(sub_dirs))):
        subdir = sub_dirs[_] #a single class videos dir, include *.avi
        video_list = os.listdir(os.path.join(path, subdir))
        for video in video_list:
            video_path = os.path.join(path, subdir, video)#sing video slip
            vfarme,_t,_td = read_video(video_path, pts_unit='sec',output_format="TCHW")
            if len(vfarme) <= speicial_frames_num:
                os.remove(video_path)
                del_list.append(video_path)
    
    print("remove the videos that havet enough frames:",del_list)
                

# prepreprocessvideos(r"F:\YANGYANG\etst2video",60)
def PreProcessVideos2Imgs(path:str,speicial_frames_num:int=16):
    save_path_abovbe = os.path.dirname(path)
    save_path_abovbe = os.path.join(save_path_abovbe,"imgs")# if u path is f:/data   ,then your img dir will be f:/imgs
    print("your imgs dir is {},!!!!!!!pls set this path to your base.json!!!!!!!!!!".format(save_path_abovbe))
    sub_dirs = os.listdir(path)# video classes dir
    labels = sorted(sub_dirs)
    for _ in tqdm(range(len(sub_dirs))):
        # if _ < 2:
            subdir = sub_dirs[_] #a single class videos dir, include *.avi
            video_list = os.listdir(os.path.join(path, subdir))
            nums_id = 0
            for video in video_list:
                nums_id += 1
                video_path = os.path.join(path, subdir, video)
                vfarme,_t,_td = read_video(video_path, pts_unit='sec',output_format="TCHW")
                v_count = len(vfarme)
                skip = int(v_count // speicial_frames_num)
                lable = torch.tensor(labels.index(video_path.split("\\")[-2]))
                save_dir = f'{lable}'
                abs_save_dir = os.path.join(save_path_abovbe,labels[int(lable)] ,str(nums_id))
                if not os.path.exists(abs_save_dir):
                    os.makedirs(abs_save_dir)
                for i in range(speicial_frames_num):
                    img = transforms.ToPILImage()(vfarme[i*skip,:,:,:])
                    img.save(os.path.join(abs_save_dir,f"{i:03d}.jpg"))


def DetectResult(path:str):
    sub_dirs = os.listdir(path) 
    try:
        for _ in tqdm(range(len(sub_dirs))):
            subdir = sub_dirs[_]
            imgss = os.listdir(os.path.join(path, subdir))
            for img_dir in imgss:
                img_path = os.path.join(path, subdir, img_dir)
                wait_detect_imgs = os.listdir(img_path)
               
            result = [s.endswith("jpg") for s in wait_detect_imgs]

        if False  in result:
            print("the dir have the files unbelong *.jpg file \n{img_path}")
            return False#need prepross
                # raise Exception(f"the dir have the files unbelong *.jpg\n{img_path}")
        else:
            return True
    except Exception as e:
                    print("error,it isn't the imgs dir,pls set spilt = train firstly , and init HMDB class again, error is \n {}".format(e))
                











__all__ = ["PreProcess","PreProcessVideos2Imgs","DetectResult"]

# DetectResult(r"F:\imgs")