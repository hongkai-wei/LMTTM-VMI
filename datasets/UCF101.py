import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import random



# 处理RGB（如减去均值归一化）
class ClipNormal(object):
  def __init__(self, b=104, g=117, r=123):
    self.means = np.array((r, g, b))
    
  def __call__(self, sample):
    #__call__使类可直接调用
    video_con ,video_label = sample['video_con'],sample['video_label']
    new_video_con = video_con-self.means
    return {'video_con': new_video_con, 'video_label': video_label}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): tuple则为输出size，int则为较小的的图像边，保持纵横比不变
    """
    def __init__(self, output_size=(182,242)):
        assert isinstance(output_size, (int, tuple)),"size must be int or tuple"
        self.output_size = output_size
        
    def __call__(self, sample):
        video_con,video_label = sample['video_con'],sample['video_label']
        
        h,w = video_con.shape[1],video_con.shape[2]
        if isinstance(self.output_size,int):
            if h<w:
                o_h,o_w = self.output_size, int(self.output_size*w/h)
            else:
                o_h,o_w = int(self.output_size*h/w), self.output_size
        else:
            o_h,o_w = self.output_size
        
        rescale_video = []
        for i in range(16):
            image = video_con[i,:,:,:]
            ## transform.resize必须接收的是uint8，否则会出错
            img = transforms.resize(image.astype("uint8"),(o_h,o_w))
            rescale_video.append(img)
        
        rescale_video = np.array(rescale_video)
                
        return {"video_con": rescale_video, "video_label":video_label}

class RandCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): 如果为int，则裁剪为方形
    """
    def __init__(self, output_size=(160,160)):
        assert isinstance(output_size, (int, tuple)),"size must be int or tuple"
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        
    def __call__(self, sample):
        video_con,video_label = sample['video_con'],sample['video_label']
        
        h,w = video_con.shape[1],video_con.shape[2]
        o_h,o_w = self.output_size
        
        top = np.random.randint(0, h-o_h)
        left = np.random.randint(0, w-o_w)
        
        crop_video = []
        for i in range(16):
            image = video_con[i,:,:,:]
            img = video_con[top:top+o_h,left:left+o_w]
            crop_video.append(img)
        
        crop_video = np.array(crop_video)
                
        return {"video_con": crop_video, "video_label":video_label}  

class ToTensor(object):
    def __call__(self,sample):
        video_con,video_label = sample['video_con'],sample['video_label']
        # numpy image: batch_size x FPS x H x W x C
        # torch image: batch_size x FPS x C X H X W
        video_con = video_con.transpose((0,3,1,2))
        return {"video_con": torch.from_numpy(video_con), "video_label":torch.FloatTensor([video_label])}

class UCF11(Dataset):
    def __init__(self,dir_name,transform=None):
        self.dir_name = dir_name
        self.transform = transform
        self.video_class = {}
        self.base_dir = os.path.join(os.path.dirname('__file__'),dir_name)
        self.label_dir = {k:v for v,k in enumerate(os.listdir(self.base_dir))}
        for each in os.listdir(self.base_dir):
            for i_class in os.listdir(os.path.join(self.base_dir,each)):
                if i_class != "Annotation":
                    for i_video in os.listdir(os.path.join(os.path.join(self.base_dir,each),i_class)):
                        self.video_class[i_video] = each
        # 由于只能按索引getitem，所以要将这里处理成列表或dataframe
        self.video_list = list(self.video_class.keys())
    
    def __len__(self):
        return len(self.video_class)
    
    ## 获取帧
    def __getitem__(self, video_idx):
        video_name = self.video_list[video_idx]
        video_path = os.path.join(os.path.join(os.path.join(self.base_dir,self.video_class[video_name]),video_name[:-7]),video_name)
        video_label = self.video_class[video_name]
        video_con = self.get_video_i(video_path)
        sample = {'video_con':video_con, 'video_label':self.label_dir[video_label]}
        if self.transform:
            sample = self.transform(sample)
        return sample

        
    def get_video_i(self, video_path):
        vd = cv2.VideoCapture(video_path)
        # 7 CV_CAP_PROP_FRAME_COUNT  #视频总帧数
        frames_num=vd.get(7)
        image_start=random.randint(1,frames_num-17)
#         print(frames_num)
        success, frame = vd.read()
        i = 1
        video_cut = []
#         videoWriter =cv2.VideoWriter('video4.avi',cv2.CAP_FFMPEG,fourcc=cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'),fps=4,frameSize=(240,320))

        while success:
            if i>=image_start and i<image_start+16:
                video_cut.append(frame)
#                 plt.subplot(4,4,i-image_start+1)
#                 plt.imshow(frame)
#                 videoWriter.write(frame)
            elif i>=image_start+16:
                break
            i += 1
            success, frame = vd.read()
#         plt.show()
#         videoWriter.release()
        return np.array(video_cut)
                            
if __name__=='__main__':
    myUCF11 = UCF11("UCF11_updated_mpg",transform=transforms.Compose([ClipNormal(),Rescale(),RandCrop(),ToTensor()]))
    dataloader=DataLoader(myUCF11,batch_size=8,shuffle=True,num_workers=0)
    ## 封装为dataset后，需要构建__len__和__getitem__方法，getitem方法回按照0-len(self)遍历
    for i_batch,sample_batched in enumerate(dataloader):
        print (i_batch,sample_batched['video_con'].size(),sample_batched['video_label'].size())
