'''
这段代码参考了[](https://keras.io/examples/vision/vivit/)
输入为BATCH,channal,T,H,W   batch,1,28,28,28
输出为batch,class_num   batch,11类别



'''

# import medmnist
# import torchvision.transforms as transforms
# from medmnist import INFO, Evaluator
import torch.utils.data as data
import torch
import torch.nn as nn
BATCH_SIZE=20
# embed_dim=512
# data_flag="organmnist3d"
# download = False
# root=r'C:\Users\BYounng\Documents\vscode\vivit\data'
# info = INFO[data_flag]
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])
# DataClass = getattr(medmnist, info['python_class'])
# transforms_v=transforms.Compose([transforms.ToTensor()])
# train_dataset = DataClass(split='train',root=root, download=download)
# test_dataset = DataClass(split='test',root=root, download=download)
# valid_dataset = DataClass(split='val',root=root, download=download)

input=torch.randn(6,1,28,28,28)  #C H W  Z 
# xx=nn.Conv3d(in_channels=1,out_channels=512,kernel_size=4,stride=4,padding="valid")(input)
# print(xx.shape)  # [512, 7, 7, 7])

class embed(nn.Module):
    def __init__(self,embed_dim,patch_size) -> None:
        super(embed,self).__init__()
        self.embed_dim=embed_dim
        self.conv3d=nn.Conv3d(in_channels=1,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size,padding="valid")
    def forward(self,videos):
        x=self.conv3d(videos)
        x=x.flatten(2,)
        x=x.transpose(1,2)
        return x  
    
       
# embe=embed(embed_dim=512,patch_size=4)
# out1=embe(input)
# print(out1.shape)#torch.Size([B,343, 512]

# raise KeyboardInterrupt
class pos_emb(nn.Module):#对所有input的做+pos处理
    def __init__(self,embed_dim,shape) -> None:
        super(pos_emb,self).__init__()
        self.embed_dim=embed_dim
        _,tokens,_=shape
        self.pos_emb=nn.Embedding(tokens,embed_dim)
        self.poss=torch.arange(tokens)
        self.poss=self.poss.type(torch.cuda.LongTensor)
    def forward(self,encode_token):
        encoder_position=self.pos_emb(self.poss)#自动变成 1  343   512 
        encoer_tokens=encode_token+encoder_position  #batch token dim
        return encoer_tokens


# x=torch.randn(343,512)
# attn=nn.MultiheadAttention(embed_dim,8,0.1,kdim=64)(x,x,x)
# print(attn[0].shape,attn[1].shape)

# raise KeyError


class vivit(nn.Module):
    def __init__(self,emb_,pos_emb_,embed_dim_) -> None:
        super(vivit,self).__init__()
        numclass=11
        eps=1e-6
        self.emb=emb_
        self.posemb=pos_emb_
        self.laynorm=nn.LayerNorm(embed_dim_,eps=eps)
        self.attn1=nn.MultiheadAttention(embed_dim_,8,0.1)
        self.attn2=nn.MultiheadAttention(embed_dim_,8,0.1)
        self.attn3=nn.MultiheadAttention(embed_dim_,8,0.1)
        self.laynorm2=nn.LayerNorm(embed_dim_,eps=eps)
        self.lines=nn.Sequential(
            nn.Linear(embed_dim_,embed_dim_*4),nn.GELU()
            ,
            nn.Linear(4*embed_dim_,embed_dim_),nn.GELU()
        )
        self.laynorm3=nn.LayerNorm(embed_dim_,eps=eps)
        self.finallinea=nn.Linear(embed_dim_,numclass)
    def forward(self,input):
        patch=self.emb(input)
        pos_pat=self.posemb(patch)
        out0=pos_pat
        pos_pat=self.laynorm(out0)
        pos_pat=self.attn1(pos_pat,pos_pat,pos_pat)[0]
        pos_pat=self.attn2(pos_pat,pos_pat,pos_pat)[0]
        out1=self.attn3(pos_pat,pos_pat,pos_pat)[0]
        out_1=out1+out0
        out2=self.laynorm2(out_1)
        out2=self.lines(out2)
        out3=out2+out_1
        fin=self.laynorm3(out3)
        fin=fin.transpose(1,2)
        fina=nn.AdaptiveAvgPool1d(1)(fin)
        fin=fina.squeeze(2)
        # batch token_  dim
        final_=self.finallinea(fin)
        # print(final_.shape)
        return final_



vivi=vivit(emb_=embed(512,4),pos_emb_=pos_emb(512,(1,343,512)),embed_dim_=512)
test=vivi.to("cuda")
test(input.cuda())

# raise KeyboardInterrupt
import medmnist
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
import torch.utils.data as data
BATCH_SIZE=20
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
train_loader=data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

import tqdm
from torch.utils.tensorboard import SummaryWriter
loger=SummaryWriter("tet_admdaw")
optim=torch.optim.AdamW(vivi.parameters(),1e-4,weight_decay=1e-5)
cit=nn.CrossEntropyLoss()
losses=[]
flag=1
for i in range(200):
    for x,y in tqdm.tqdm(train_loader):
        # x = x.type(torch.float32)
        x=x.to("cuda",dtype=torch.float32)
        y=y.to("cuda",dtype=torch.long)
        # y=y.type(torch.long)
        optim.zero_grad()
        out=vivi(x)
        loss=cit(out,y.squeeze(1))
        losses.append(loss.item())
        loss.backward()
        optim.step()
        loger.add_scalar("loss per step ",loss.item(),flag)
        if flag%100==0:
            save_loss=sum(losses)/len(losses)
            loger.add_scalar("loss per 100 step ",save_loss,flag)
            losses=[]
            # torch.save(vivi.state_dict(),f"num_{flag}_loss{save_loss}.pth")
        flag+=1



# x=torch.randn(5,1,28,28,28)
# y=vivi(x)
# print(y.shape)
# import torch.nn as nn
# print(y)#torch.Size([5, 343, 11]) 
# 
# def train():

