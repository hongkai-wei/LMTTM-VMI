
import torch
from modify import ttm  #input b c t h w ,out : out [b,clss] mem[b,size,dim]
import torch
model=ttm().cuda()
pth=r"F:\TTM_CODE\modify_ttm\pth\_decay_noise-change_0.07016966515220702.pth"
a=torch.load(pth)
mem=a["mem"]
load_state=a["mdoel"]
from einops.layers.torch import Rearrange
import tqdm
import medmnist
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
import torch.utils.data as data
BATCH_SIZE=21
data_flag="organmnist3d"
download = False
root=r'C:\Users\BYounng\Documents\vscode\vivit\data'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train',root=root, download=download)
test_dataset = DataClass(split='test',root=root, download=download)
valid_dataset = DataClass(split='val',root=root, download=download)
train_loader=data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
model.eval()
al_sam=0
real=0
from modify import mem_process
mem=mem_process(mem)
bar=tqdm.tqdm(train_loader,leave=False)
model.load_state_dict(load_state)




# for x,y in bar:
#         model.train()  
#       # x=Rearrange('b c t h w -> b t c h w')(x)
#         y=y.squeeze(1)
#         x=x.to("cuda",dtype=torch.float32)
#         y=y.to("cuda",dtype=torch.long)

#         out,mem=model(x,mem)
       



for x,y in bar:
    # x=Rearrange('b c t h w -> b t c h w')(x)
    y=y.squeeze(1)
    x=x.to("cuda",dtype=torch.float32)
    y=y.to("cuda",dtype=torch.long)
    out,ooo,wuguan=model(x,mem[0])
    lens=len(y)
    out=torch.argmax(out,dim=1)
    outts=sum((y==out))
    al_sam+=lens
    real+=outts

    
print(f"yangbenis{al_sam},real is {real},acc is {real/al_sam}")