'''
vivit结合TTM 
其中VIVIT输出shape 是[batch,step,token_num,dim]
pre处理在step维度加上time_cls_token  得到[batch,step+1,token_num,dim]

TTM的输入是[batch,step+1,token_num,dim]
输出是指定tokens数量[batch,step+1,special_token_num,dim]

初步想法是vivit做backbone提取特征,送入到带有记忆模块的TTM,TTM输出[batch,step+1,special_token_num,dim]



 在special_token_num维度上进行mean得到[batch,step+1,dim] 送入到全连接层得到[batch,step+1,class_num] 
 然后取最后一个step+1的输出得到[batch,num_num] 与label计算loss

 or在step上进行mean得到[batch,1,_num] 与label计算loss


 or把step和token_num 融合得到[batch,step*token_num,dim] 送入到全连接层得到[batch,step*token_num,class_num] 然后中间维度取mean得到[batch,1,class_num] 与label计算loss??

'''
import torch
import torch.nn as nn
import einops 
from einops import rearrange, reduce, repeat # einops是一个用于操作tensor的库，可以用于改变tensor的形状，维度等
from einops.layers.torch import Rearrange
from tokenlearner import TokenLearner

batch = 10         # 批量大小
step = 28          # 步长
dim = 16          # 维度
in_channels = 1    # 输入通道数
patch_size = 4     # Patch 大小
token_num = 64     # 标记数量
special_num_token = 8  # 特殊标记数量
# mem=torch.zeros(batch,64,512).cuda() # 创建记忆单元，64是记忆单元的数量


class pre_procee(nn.Module): ### 输入 batch step 1 28 28   变换到  batch step len dim  方便进行时序处理     #我觉得是没加cls
    """定义了一个数据预处理层，
    它包括一个卷积层，
    用于将输入数据从(batch, step, 1, 28, 28)[bs,step,c,h,w]的形状转换为(batch, step, len/num_tokens, dim)的形状
    """   
    def __init__(self) -> None:
        super(pre_procee,self).__init__()
        # num_patches = (image_size // patch_size) ** 2
        self.con=nn.Conv3d(in_channels=1,out_channels=dim,kernel_size=patch_size,stride=patch_size,padding="valid")
        # patch_dim = in_channels * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )# 输出 batch step num_patches(token_num) dim
        # self.pos=nn.Embedding(token_num+1,dim)
        # self.ar_token=torch.arange(token_num+1).cuda()
        # self.cls=nn.Parameter(torch.randn(1,1,dim).cuda())
    

    def forward(self,input):
        input=input.transpose(1,2)
        x=self.con(input)
        x=x.flatten(3) # flatten是在指定维度上对输入的张量进行压平操作，x的shape是[batch,step,token_num,dim]
        x=x.permute(0,2,3,1)
        # x=x.transpose(1,2)
        # x=self.to_patch_embedding(input).cuda()
        # b,t,token_num,dim=x.shape
        # cls=repeat(self.cls,'() n d -> b t n d',b=b,t=t)
        # cls=cls.to("cuda")
        # x=torch.cat((cls,x),dim=2)
        # x=x+self.pos(self.ar_token)
        return x

class TokenLearnerMHA(nn.Module):
    """使用MHA的TokenLearner模块。
    属性：
    num_tokens：要生成的标记数量。
    num_heads：用于点积注意力的头数。
    """ # [batch,token_num,dim] # 1024->8
    def __init__(self) -> None:
        super(TokenLearnerMHA,self).__init__()
        # special_num_token就是指定输出的token数
        self.query=nn.Parameter(torch.randn(batch,special_num_token,dim).cuda()) # Parameter是一种特殊的Tensor，会被自动添加到模型的参数列表中
        self.attn=nn.MultiheadAttention(dim,8,0.1,batch_first=True) # batch_first=True表示输入数据的形状为(batch, step, token_num, dim)
    def forward(self,input): # input.shape是[batch,step+1,token_num,dim]
        return self.attn(self.query,input,input)[0]  #返回[0]的 output shape [batch,speical_num_token,dim]

class TokenAddEraseWrite(nn.Module):
    def __init__(self) -> None:
        super(TokenAddEraseWrite,self).__init__()
        self.trasns_bolck1=nn.Sequential(
            nn.LayerNorm(dim),nn.Linear(dim,3*dim),nn.Linear(3*dim,special_num_token),nn.GELU()
        )
        self.laynorm1=nn.LayerNorm(dim)
        self.laynorm2=nn.LayerNorm(dim)
        self.trasns_bolck2=nn.Sequential(nn.Linear(special_num_token,3*dim),nn.Linear(3*dim,special_num_token),nn.GELU())
        self.trasns_bolck2_=nn.Sequential(nn.Linear(dim,3*dim),nn.Linear(3*dim,dim),nn.GELU())
        self.trasns_bolck3=nn.Sequential(nn.Linear(special_num_token,3*dim),nn.Linear(3*dim,special_num_token),nn.GELU())
        self.trasns_bolck3_=nn.Sequential(nn.Linear(dim,3*dim),nn.Linear(3*dim,dim),nn.GELU())
        self.softmax_a=nn.Softmax(dim=-1)

    def forward(self,mem,control): # control其实就是output
        select=self.trasns_bolck1(mem) # trasns_bolck1是mlp,这就是elif self.processing_unit == 'mlp':
        # ----------以下开始的AddWrite-----------
        select=select.transpose(1,2)
        select=self.softmax_a(select)   #20   8  64 

        et=self.laynorm1(control)
        et=et.transpose(1,2)
        et=self.trasns_bolck2(et)
        et=et.transpose(1,2)
        et=self.trasns_bolck2_(et)

        temp_sele=select.unsqueeze(-1).cuda() # temple_selected
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



# test=token_add_earse().cuda()
# c=test(mem,torch.randn(batch,speical_num_token,dim).cuda())
# print(c.shape)   # batch memsize mem——dim

class TokenTuringMachineUnit(nn.Module):
    def __init__(self) -> None:
        super(TokenTuringMachineUnit,self).__init__()
        self.TokenLearner=TokenLearner(in_channels=16,num_tokens=8) 
        # self.TokenLearnerMHA=TokenLearnerMHA()
        self.TokenAddEraseWrite=TokenAddEraseWrite()
        self.mlp=nn.Sequential(nn.LayerNorm(dim),
                               nn.Linear(dim,dim*3),
                               nn.GELU(),
                               nn.Linear(dim*3,dim),
                               nn.GELU())
        self.lay=3
        self.norm=nn.LayerNorm(dim)

    def forward(self,mem,step_input):
        all_token=torch.cat((mem,step_input),dim=1) # all_token的shape是[batch,mem_size+special_num_token,dim]
        all_token=self.TokenLearner(all_token)
        output_token=all_token
        for i in range(self.lay):
            output_token=self.mlp(output_token)
        output_token=self.norm(output_token)
        # mem_out_tokens=torch.cat((mem,step_input,output_token),dim=1)
        mem_out_tokens=self.TokenAddEraseWrite(mem,output_token)
        return (mem_out_tokens,output_token)



class TokenTuringMachineEncoder(nn.Module):
    def __init__(self) -> None:
        self.mem_size=95
        super(TokenTuringMachineEncoder,self).__init__()
        self.mem=torch.zeros(batch,self.mem_size,dim).cuda()
        self.TokenTuringMachineUnit=TokenTuringMachineUnit()
        self.cls=nn.Linear(dim,11) # cls是一个全连接层，输入是dim维，输出是11维
        self.pre=pre_procee()
    
    def forward(self,input,mem=None):
        input=self.pre(input)
        b,t,_,c=input.shape # b是batch，t是step，_是token_num，c是dim
        outs=[]
        if mem==None:
            self.mem=torch.zeros(b,self.mem_size,c).cuda()
        else:
            self.mem=mem
        for i in range(t):
            self.mem,out=self.TokenTuringMachineUnit(self.mem , input[:,i,:,:])
            outs.append(out)

        # 满足输出的shape---自添加
        outs=torch.stack(outs,dim=1)#SHAPE [B,STEP,NUM_TOKEN,DIM]
        out=outs.view(batch,-1,dim) # out的shape是[batch,step*token_num,dim]
        out=out.transpose(1,2)
        out=nn.AdaptiveAvgPool1d(1)(out) # AdaptiveAvgPool1d是自适应平均池化层，输出的形状是[batch,dim,1]
        out=out.squeeze(2)
        # print(out.shape)

        return self.cls(out),self.mem # 原来是正常的out和 mem
        ...

if __name__ == "__main__":
    inputs = torch.randn(batch,step,1,28,28).cuda()
    model=TokenTuringMachineEncoder().cuda()
    out,mem=model(inputs)
    print(out.shape)









