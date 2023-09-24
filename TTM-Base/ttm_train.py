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
data_flag="organmnist3d" # organmnist3d是3d的数据集，organmnist是2d的数据集
download = False
root=r'E:\\CHDLearning\\TTM\\TTM-Pytorch\\data'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', root=root, download=False)
test_dataset = DataClass(split='test', root=root, download=False)
valid_dataset = DataClass(split='val', root=root, download=False)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



loger=SummaryWriter(".\\TTM-Pytorch\\ttm26_mixer007")
model=TokenTuringMachineEncoder().cuda()
optim=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4)
cit=nn.CrossEntropyLoss()
losses=[]
flag=1


for i in range(500):
    for x,y in tqdm.tqdm(train_loader):
        # x = x.type(torch.float32)
        x = Rearrange('b c t h w -> b t c h w')(x)
        x = x.to("cuda", dtype = torch.float32)
        y = y.to("cuda", dtype = torch.long)
        # y=y.type(torch.long)
        optim.zero_grad()
        out, save_memory=model(x)
        loss = cit(out, y.squeeze(1))
        losses.append(loss.item())
        loss.backward()
        optim.step()
        loger.add_scalar("loss per step ", loss.item(), flag)
        if flag % 100 == 0:
            save_loss = sum(losses) / len(losses)
            loger.add_scalar("loss per 100 step ", save_loss, flag)
            print(save_loss)
            losses=[]
            # print("*"*10)
            torch.save({"model":model.state_dict(),"memory":save_memory},f"E:\\CHDLearning\\TTM\\TTM-Pytorch\\ttm_Loss_6mixer007\\_complex_TTM_{flag}_loss.pth")
        flag += 1 