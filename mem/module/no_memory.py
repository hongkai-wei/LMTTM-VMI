from no_mem_modeule import ttm  #input b c t h w ,out : out [b,clss] mem[b,size,dim]
import torch
model=ttm().cuda()
mem=None
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
citirion=torch.nn.CrossEntropyLoss()

#### shujuji
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
train_loader=data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
val_dataloader=data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_dataloader=data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
from torch.utils.tensorboard import SummaryWriter



loger=SummaryWriter(r"F:\TTM_CODE\modify_ttm\board\0.42de_memory_have_trained_memory")



pth=r"F:\TTM_CODE\modify_ttm\pth\memory_test_0.007536983633392744.pth"
a=torch.load(pth)
mem=a["mem"]
load_state=a["mdoel"]
# model.load_state_dict(load_state)
bar_all=tqdm.tqdm(range(300))
i=1
acc_i=1
for _ in bar_all:

    bar_all.set_description("train epoch is {}/300".format(_+1))
    bar=tqdm.tqdm(train_loader,leave=False)

    losses=[]
    for x,y in bar:
        model.train()  
      # x=Rearrange('b c t h w -> b t c h w')(x)
        y=y.squeeze(1)
        x=x.to("cuda",dtype=torch.float32)
        y=y.to("cuda",dtype=torch.long)
        out,mem=model(x,mem)
        optimizer.zero_grad()
        loss=citirion(out,y)
        losses.append(loss.item())
        # torch.autograd.set_detect_anomaly = True
        loss.backward()
        # torch.autograd.set_detect_anomaly = True
        optimizer.step()
        bar.set_postfix(loss=loss.item())
        loger.add_scalar("loss per step ",loss.item(),i)
        
        if i%100==0:
            avg=sum(losses)/len(losses)
            loger.add_scalar("loss per 100 step",avg,i )
            torch.save({"mem":mem,"mdoel":model.state_dict()},r"F:\TTM_CODE\modify_ttm\pth\_0.42de_memory_have_trained_memory_{}.pth".format(avg))
            for val_x,val_y in test_dataloader:
                model.eval()
                val_x=val_x.to("cuda",dtype=torch.float32)
                val_y=val_y.to("cuda",dtype=torch.long)
                out,mem=model(val_x)
                out=torch.argmax(out,dim=1)
                val_y=val_y.squeeze(1)
                all=val_y.size(0)
                result=(out==val_y).sum().item()
                acc=(result/all)*100
                loger.add_scalar("acc",acc,acc_i)
                acc_i+=1
        i+=1