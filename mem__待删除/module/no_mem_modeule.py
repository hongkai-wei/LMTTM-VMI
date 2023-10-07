'''
MODIFY:
add linear after conv3d
reduce dim by using linear
'''
import torch
import torch.nn as nn
import einops 
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
batch=21
step=28
dim=512
in_channels=1
patch_size=4


class pre_procee(nn.Module): #输入B,C,STEP,H,W  最终得到B C STEP TOKEN -> B STEP TOKEN C 
    def __init__(self) -> None:
        super(pre_procee,self).__init__()
        self.con=nn.Conv3d(in_channels=1, out_channels=dim, kernel_size=patch_size, stride=patch_size, padding="valid")
        self.act=nn.ReLU()
    def forward(self,input):
        # input=input.transpose(1,2)
        x=self.con(input)
        x=self.act(x)
        x=x.flatten(3)
        x=x.permute(0, 2, 3, 1)
        return x
class token_mha(nn.Module):
    def __init__(self) -> None:
        super(token_mha, self).__init__()
        speical_num_token=8
        self.query=nn.Parameter(torch.randn(batch, speical_num_token, dim).cuda())
        self.attn=nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)  
    def forward(self, input):
        return self.attn(self.query,input,input)[0] #B STEP 8 C
class token_add_earse(nn.Module):#mem B,SIZE,DIM  control B,8,DIM    #OUT B,SIZE,DIM
    def __init__(self) -> None:
        super(token_add_earse,self).__init__()
        speical_num_token=8
        self.trasns_bolck1=nn.Sequential(
            nn.LayerNorm(dim),nn.Linear(dim,3*dim),nn.Linear(3*dim,speical_num_token),nn.GELU()
        )
        self.laynorm1=nn.LayerNorm(dim)
        self.laynorm2=nn.LayerNorm(dim)
        self.trasns_bolck2=nn.Sequential(nn.Linear(speical_num_token,3*dim),nn.Linear(3*dim,speical_num_token),nn.GELU())
        self.trasns_bolck2_=nn.Sequential(nn.Linear(dim,3*dim),nn.Linear(3*dim,dim),nn.GELU())
        self.trasns_bolck3=nn.Sequential(nn.Linear(speical_num_token,3*dim),nn.Linear(3*dim,speical_num_token),nn.GELU())
        self.trasns_bolck3_=nn.Sequential(nn.Linear(dim,3*dim),nn.Linear(3*dim,dim),nn.GELU())
        self.softmax_a=nn.Softmax(dim=-1)
    def forward(self,mem,control):
        select=self.trasns_bolck1(mem)
        select=select.transpose(1,2)
        select=self.softmax_a(select)
        et=self.laynorm1(control)
        et=et.transpose(1,2)
        et=self.trasns_bolck2(et)
        et=et.transpose(1,2)
        et=self.trasns_bolck2_(et)
        temp_sele=select.unsqueeze(-1).cuda()
        temp_et=et.unsqueeze(2).cuda()
        wet=temp_sele*temp_et
        wet=1-wet
        wet=torch.prod(wet,dim=1)
        output=mem*wet
        at=self.laynorm2(control)
        at=at.transpose(1,2)
        at=self.trasns_bolck3(at)
        at=at.transpose(1,2)
        at=self.trasns_bolck3_(at)
        temp_at=at.unsqueeze(2).cuda()
        wat=temp_sele*temp_at
        wat=1-wat
        wat=torch.mean(wat,dim=1)
        output=output+wat
        return output   


import torch.nn.init as init

class pos_add(nn.Module):
    def __init__(self) -> None:
        super(pos_add,self).__init__()


    def forward(self,x):# B LEN C

        mem_out_tokens=x
        posemb_init = torch.nn.Parameter(torch.empty(1, mem_out_tokens.size(1), mem_out_tokens.size(2))).cuda()
        init.normal_(posemb_init, std=0.02)
        mem_out_tokens = mem_out_tokens + posemb_init
        self.register_buffer("pos", posemb_init)
        return mem_out_tokens


class ttm_unit(nn.Module):
    def __init__(self) -> None:
        super(ttm_unit,self).__init__()
        self.token_mha=token_mha()
        self.token_add_earse=token_add_earse()
        self.mlp=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*3),nn.GELU(),nn.Linear(dim*3,dim),nn.GELU())
        self.lay=3
        self.norm=nn.LayerNorm(dim)
        self.pos_ad=pos_add()
    def forward(self,step_input,mem):
        all_token=torch.cat((mem,step_input),dim=1)
        all_token=self.pos_ad(all_token)
        all_token=self.token_mha(all_token)
        output_token=all_token
        for i in range(self.lay):
            output_token=self.mlp(output_token)
        output_token=self.norm(output_token)
        mem_out_tokens=self.token_add_earse(mem,output_token)
        return (mem_out_tokens,output_token)
    
class ttm(nn.Module):
    def __init__(self) -> None:
        self.mem_size=128
        step=28
        speical_token=8
        super(ttm,self).__init__()
        self.memmo=torch.zeros(batch,self.mem_size,dim).cuda()
        self.ttm_unit=ttm_unit()
        self.cls=nn.Linear(dim,11)
        self.pre=pre_procee()
        self.relu=nn.ReLU()
        self.linear_1=nn.Linear((step//patch_size)*speical_token, out_features=1)       
        self.laynorm=nn.LayerNorm(512)
    def forward(self,input,mem=None):
        input=self.pre(input)
        b,t,len,c=input.shape
        outs=[]
        if mem==None:
            self.mem=self.memmo
        else:
            
            self.mem=mem.detach()
        for i in range(t):
            self.mem,out=self.ttm_unit(input[:,i,:,:],self.mem)
            outs.append(out)

        outs=torch.stack(outs,dim=1)#SHAPE [B,STEP,NUM_TOKEN,DIM]
        out=outs.view(batch,-1,dim)
        out=out.transpose(1,2)##add linear 
        out=self.linear_1(out)
        # out=nn.AdaptiveAvgPool1d(1)(out)
        out=out.squeeze(2)
        out=self.relu(out)      
        self.mem=self.laynorm(self.mem)
        self.mem=mem_process_v2(self.mem)
        return self.cls(out),self.mem
    
def mem_process_v2(mem:torch.Tensor):
    b,len,dim=mem.shape
    noise=torch.randn((b,len,dim)).cuda()
    rate=0.42
    return rate*noise+1*mem















if __name__ == "__main__":
    # ttm_input=torch.randn(8,1,60,196,196).cuda()
    # model=ttm().cuda()
    # out,mem=model(ttm_input)
    # print(out.shape)
    # raise KeyboardInterrupt

    import medmnist
    import torchvision.transforms as transforms
    from medmnist import INFO, Evaluator
    import torch.utils.data as data
    BATCH_SIZE=30
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
    # val_dataloader=data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    data=data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    ###################这是  预测部分 如果要训练请把这部分注释掉
    # import tqdm
    # all_y=0
    # all_real=0

    # a=torch.load(r"F:\ttm_2\BEST__complex_TTM_10920_loss0.0161.pth")
    # load_mem=a["mem"]
    # load_state=a["mdoel"]
    # model.load_state_dict(load_state)
    # model.eval()
    # for x,y in tqdm.tqdm(data):
    #         x=Rearrange('b c t h w -> b t c h w')(x)     
    #         x=x.to("cuda",dtype=torch.float32)
    #         y=y.to("cuda",dtype=torch.long)
    #         out,mem=model(x)
            
    #         # 99.58333333333333%
    #         # out,mem=model(x,load_mem)     #acc is 38.229166666666664%
    #         out=torch.argmax(out,dim=1)
    #         y=y.squeeze(1)
    #         all=y.size(0)
    #         result=(out==y).sum().item()
    #         all_y+=all
    #         all_real+=result
            

    #  ###   B,C,STEP,H,W
    # print("总样本数：",all_y,"预测对的数目:",all_real)
    # print("acc is {}%".format((all_real/all_y)*100))
            
    # raise KeyboardInterrupt
        
    # ##############################eval 部分结束


    ######################下面是train部分



    # import tqdm
    # from torch.utils.tensorboard import SummaryWriter
    # loger=SummaryWriter("success")
    # optim=torch.optim.AdamW(model.parameters(),1e-5,weight_decay=1e-4)
    # cit=nn.CrossEntropyLoss()
    # losses=[]
    # flag=1

    # for i in range(350):
    #     for x,y in tqdm.tqdm(train_loader):
    #         # x = x.type(torch.float32)
    #         x=Rearrange('b c t h w -> b t c h w')(x)
            # x=x.to("cuda",dtype=torch.float32)
    #         y=y.to("cuda",dtype=torch.long)
    #         # y=y.type(torch.long)
    #         optim.zero_grad()
    #         out,save_mem=model(x)
    #         loss=cit(out,y.squeeze(1))
    #         losses.append(loss.item())
    #         loss.backward()
    #         optim.step()
    #         loger.add_scalar("loss per step ",loss.item(),flag)
    #         if flag%42==0:
    #             save_loss=sum(losses)/len(losses)
    #             loger.add_scalar("loss per 300 step ",save_loss,flag)
    #             print(save_loss)
    #             losses=[]
    #             # print("*"*10)
    #             torch.save({"mdoel":model.state_dict(),"mem":save_mem},f"F:\\ttm_2\\_complex_TTM_{flag}_loss{save_loss:.4f}.pth")
    #             # raise KeyboardInterrupt
    #         flag+=1






