import torch
import torch.nn as nn
from ttm_model import TokenTuringMachineEncoder
from einops.layers.torch import Rearrange
import torch
import medmnist
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
import torch.utils.data as data
import tqdm
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE=100
data_flag="organmnist3d"
download = False
root=r'E:\\CHDLearning\\TTM\\TTM-Pytorch\\data'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

loger=SummaryWriter(".\\TTM-Pytorch\\ttm_test26mixer007")

train_dataset = DataClass(split='train',root=root, download=False)
test_dataset = DataClass(split='test',root=root, download=False)
valid_dataset = DataClass(split='val',root=root, download=False)
train_loader=data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
val_dataloader=data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

pth = "E:\\CHDLearning\\TTM\\TTM-Pytorch\\ttm_Loss_6mixer007\\"
pth_files = [f"{pth}_complex_TTM_{i * 100}_loss.pth" for i in range(1, 45)] 

for i in range(len(pth_files)):
    checkpoint = torch.load(pth_files[i])
    load_state = checkpoint["model"]
    load_mem = checkpoint["memory"]
    model = TokenTuringMachineEncoder().cuda()
    model.load_state_dict(load_state)
    model.eval()

    all_y = 0
    all_real = 0

    for x,y in tqdm.tqdm(val_dataloader):
        x = Rearrange('b c t h w -> b t c h w')(x)     
        x = x.to("cuda", dtype = torch.float32)
        y = y.to("cuda", dtype = torch.long)
        out, mem = model(x)
        # out,mem=model(x,load_mem)     #acc is 38.229166666666664%
        out = torch.argmax(out, dim=1)
        y = y.squeeze(1)
        all = y.size(0)
        result = (out == y).sum().item()

        all_y += all

        all_real += result
         ###   B,C,STEP,H,W
        print("总样本数：",all_y,"预测对的数目:",all_real)
        print("acc is {}%".format((all_real/all_y)*100))
        acc = (all_real/all_y)*100
        loger.add_scalar("acc per 100 step ", acc, i)

        
raise KeyboardInterrupt
    
# #############################eval 部分结束





