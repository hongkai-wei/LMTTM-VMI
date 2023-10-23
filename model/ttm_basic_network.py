import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
from .tokenLearner_network import TokenLearnerModuleV11
from config.configure import Config
import numpy as np

config = Config.getInstance()
config = config["model"]
drop_r = config["model"]["drop_r"]
process_unit = config["model"]["process_unit"]
memory_mode = config["model"]["memory_mode"]
in_channels = config["model"]["in_channels"]
dim = config["model"]["dim"]
memory_tokens_size = config["model"]["memory_tokens_size"]
step = config["model"]["step"]
patch_size = config["model"]["patch_size"]
num_tokens = config["model"]["num_tokens"]
Read_use_positional_embedding = config["model"]["Read_use_positional_embedding"]
Write_use_positional_embedding = config["model"]["Write_use_positional_embedding"]
import torch
import torch.nn as nn

class PreProcess(nn.Module):
    def __init__(self, patch_t=4, patch_h=4, patch_w=4) -> None:
        super(PreProcess, self).__init__()
        self.lay = nn.Sequential(Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', 
                                            pt=patch_t, ph=patch_h, pw=patch_w),
                                  nn.Linear(patch_h*patch_t*patch_w*in_channels, dim), )

    def forward(self, x):
        x = self.lay(x)
        return x

class PreProcessV2(nn.Module):  # 输入Batch,Channels,Step,H,W  最终得到Batch，STEP，TOKEN，Channels
    def __init__(self) -> None:
        super(PreProcessV2, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, 
                             out_channels=dim,
                             kernel_size=patch_size, 
                             stride=patch_size, 
                             padding="valid")
        self.relu = nn.ReLU()

    def forward(self, input):
        input = input.transpose(1, 2)
        x = self.conv(input)
        x = self.relu(x)
        x = x.flatten(3)
        x = x.permute(0, 2, 3, 1)
        return x

class TokenLearnerMHA(nn.Module):
    def __init__(self, output_tokens) -> None:
        super(TokenLearnerMHA, self).__init__()
        self.query = nn.Parameter(torch.randn(config["batch_size"], output_tokens, dim).cuda())
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)

    def forward(self, input):
        # [0]是结果，[1]是权重  B STEP 8 C
        return self.attn(self.query, input, input)[0]


# mem B,SIZE,DIM  control B,8,DIM    #OUT B,SIZE,DIM
class TokenAddEraseWrite(nn.Module):
    def __init__(self) -> None:
        super(TokenAddEraseWrite, self).__init__()
        self.mlp_block1 = nn.Sequential(nn.LayerNorm(dim), 
                                           nn.Linear(dim, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.laynorm = nn.LayerNorm(dim)
        self.mlp_block2 = nn.Sequential(nn.Linear(num_tokens, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.mlp_block3 = nn.Sequential(nn.Linear(dim, 3*dim), 
                                            nn.Linear(3*dim, dim), 
                                            nn.GELU())
        self.mlp_block4 = nn.Sequential(nn.Linear(num_tokens, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.mlp_block5 = nn.Sequential(nn.Linear(dim, 3*dim), 
                                            nn.Linear(3*dim, dim), 
                                            nn.GELU())
        self.query = nn.Parameter(torch.randn(
            config["batch_size"], memory_tokens_size, dim).cuda())
        self.trans_outdim = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, memory_tokens, control_inputs):
        selected = self.mlp_block1(memory_tokens)
        selected = selected.transpose(1, 2)
        selected = self.softmax(selected)

        et = self.laynorm(control_inputs)
        et = et.transpose(1, 2)
        et = self.mlp_block2(et)
        et = et.transpose(1, 2)
        et = self.mlp_block3(et)

        wet = selected.unsqueeze(-1).cuda() * et.unsqueeze(2).cuda()
        wet = 1 - wet
        wet = torch.prod(wet, dim=1)

        output = memory_tokens * wet

        at = self.laynorm(control_inputs)
        at = at.transpose(1,2)
        at = self.mlp_block4(at)
        at = at.transpose(1,2)
        at = self.mlp_block5(at)

        wat = selected.unsqueeze(-1).cuda() * at.unsqueeze(2).cuda()
        wat = 1 - wat
        wat = torch.mean(wat, dim=1)

        output = output + wat
        # 更改output形状
        output = self.trans_outdim(self.query, output, output)[0]



        return output


class TokenTuringMachineUnit(nn.Module):
    def __init__(self) -> None:
        super(TokenTuringMachineUnit, self).__init__()
        self.process_unit = process_unit
        self.memory_mode = memory_mode
        self.Read_use_positional_embedding = Read_use_positional_embedding
        self.Write_use_positional_embedding = Write_use_positional_embedding
        self.tokenLearner1 = TokenLearnerModuleV11(in_channels=dim, num_tokens=num_tokens, num_groups=1)
        self.tokenLearner2 = TokenLearnerModuleV11(in_channels=dim, num_tokens=memory_tokens_size, num_groups=1)
        self.transformerBlock = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 3, dropout=0.2)
        self.tokenLearnerMHA1 = TokenLearnerMHA(num_tokens)
        self.tokenLearnerMHA2 = TokenLearnerMHA(memory_tokens_size)
        self.tokenAddEraseWrite = TokenAddEraseWrite()
        self.mlpBlock = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, dim*3),
                                 nn.Dropout(drop_r),
                                 nn.GELU(),
                                 nn.Linear(dim*3, dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_r))
        self.num_layers = 3
        self.norm = nn.LayerNorm(dim)
        self.mixer_sequence_block = nn.Sequential(nn.Linear(num_tokens, num_tokens * 6),
                                                  nn.GELU(),
                                                  nn.Dropout(drop_r),
                                                  nn.Linear(num_tokens * 6, num_tokens),
                                                  nn.GELU())
        self.mixer_channels__block = nn.Sequential(nn.Linear(dim, dim * 3),
                                                   nn.GELU(),
                                                   nn.Dropout(drop_r),
                                                   nn.Linear(dim * 3, dim),
                                                   nn.GELU())
        self.dropout = nn.Dropout(drop_r)

    def forward(self, memory_tokens, input_tokens):
        all_tokens = torch.cat((memory_tokens, input_tokens), dim=1)
        # Read add posiutional
        if self.Read_use_positional_embedding:
            all_tokens = all_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, all_tokens.size(1), all_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            all_tokens = all_tokens + posemb_init

        if self.memory_mode == 'TL' or self.memory_mode == 'TL-AddErase':
            all_tokens=self.tokenLearner1(all_tokens)
        elif self.memory_mode == 'TL-MHA':
            all_tokens=self.tokenLearnerMHA1(all_tokens)

        if self.process_unit == 'transformer':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.transformerBlock(output_tokens)

        elif self.process_unit == 'mixer':
            output_tokens = all_tokens # all_tokens的shape是[batch,mem_size+special_num_token,dim]
            for _ in range(self.num_layers):
                # Token mixing，不同token互通
                x_output_tokens = output_tokens
                x_output_tokens = self.norm(x_output_tokens)
                x_output_tokens = x_output_tokens.permute(0, 2, 1) # permute是将输入张量的维度换位，output_tokens的shape是[batch,dim,mem_size+special_num_token]
                x_output_tokens = self.mixer_sequence_block(x_output_tokens) # mixer_block是一个全连接层，输入是dim维，输出是dim维
                x_output_tokens = x_output_tokens.permute(0, 2, 1) # output_tokens的shape是[batch,mem_size+special_num_token,dim]
                x_output_tokens = x_output_tokens + output_tokens # output_tokens的shape是[batch,mem_size+special_num_token,dim]
                x_output_tokens = self.dropout(x_output_tokens)

                # Channel mixing，token内部互通
                y_output_tokens = self.norm(x_output_tokens)
                y_output_tokens = self.mixer_channels__block(y_output_tokens)
                y_output_tokens = self.dropout(y_output_tokens)
                output_tokens = output_tokens + y_output_tokens
            output_tokens = self.norm(output_tokens)
        
        elif self.process_unit == 'mlp':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.norm(output_tokens)
                output_tokens = self.mlpBlock(output_tokens)
            output_tokens = self.norm(output_tokens)

        memory_input_tokens = torch.cat((memory_tokens, input_tokens, output_tokens), dim=1)

        # Write add posiutional
        if self.Write_use_positional_embedding:
            memory_input_tokens = memory_input_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, memory_input_tokens.size(1), memory_input_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            memory_input_tokens = memory_input_tokens + posemb_init

        if self.memory_mode == 'TL':
            memory_output_tokens = self.tokenLearner2(memory_input_tokens)
        elif self.memory_mode == 'TL-MHA':
            memory_output_tokens = self.tokenLearnerMHA2(memory_input_tokens)
        elif self.memory_mode == 'TL-AddErase':
            memory_output_tokens = self.tokenAddEraseWrite(memory_input_tokens,output_tokens)
        
        return (memory_output_tokens,output_tokens)


class TokenTuringMachineEncoder(nn.Module):
    def __init__(self) -> None:
        self.memory_tokens_size = memory_tokens_size
        super(TokenTuringMachineEncoder, self).__init__()
        self.memory_tokens = torch.zeros(config["batch_size"], self.memory_tokens_size, dim).cuda()
        self.tokenTuringMachineUnit = TokenTuringMachineUnit()
        self.cls = nn.Linear(dim, config["out_class_num"])
        self.pre = PreProcess()
        self.preV2 = PreProcessV2()
        self.relu = nn.ReLU()

    def forward(self, input, memory_tokens):
        if config["dataset_name"] == "UCF101":
            input = self.preV2(input)
        else:
            input = self.pre(input)
        b, t, _, c = input.shape
        # b, t, c, _, _ = input.shape # b是batch，t是step，_是token_num，c是dim
        outs=[]
        if memory_tokens == None:
            memory_tokens = torch.zeros(b,self.memory_tokens_size,c).cuda() #  c, h, w
        else:
            memory_tokens = memory_tokens.detach()
        for i in range(t):
            memory_tokens, out = self.tokenTuringMachineUnit(memory_tokens, input[:,i,:,:])
            outs.append(out)

        # 满足输出的shape---自添加
        outs = torch.stack(outs, dim=1)#SHAPE [B,STEP,NUM_TOKEN,DIM]
        out = outs.view(config["batch_size"], -1, dim) # out的shape是[batch,step*token_num,dim]
        out = out.transpose(1, 2)
        out = nn.AdaptiveAvgPool1d(1)(out) # AdaptiveAvgPool1d是自适应平均池化层，输出的形状是[batch,dim,1]
        out = out.squeeze(2)
        # print(out.shape)

        if config["load_memory_add_noise"]:
            if config["load_memory_add_noise_mode"] == "normal":
                noise = torch.randn_like(memory_tokens)
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise_rate * noise
            elif config["load_memory_add_noise_mode"] == "laplace":
                noise = torch.distributions.laplace.Laplace(loc = 10, scale = 10).sample(memory_tokens.size())
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif config["load_memory_add_noise_mode"] == "uniform":
                noise = torch.FloatTensor(memory_tokens.size()).uniform_(-0.5, 0.5)
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif config["load_memory_add_noise_mode"] == "exp":
                noise = torch.empty(memory_tokens.size()).exponential_()
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif config["load_memory_add_noise_mode"] == "gamma":
                shape = torch.tensor([2.0])  # Gamma分布的形状参数
                scale = torch.tensor([2.0])  # Gamma分布的尺度参数
                noise = torch.empty(memory_tokens.size())  # 创建与噪音张量相同的空张量
                noise = noise.cuda()
                noise.copy_(torch.from_numpy(np.random.gamma(shape.item(), scale.item(), size=noise.size())))  # 将正态分布随机数转化为Gamma分布随机数
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif config["load_memory_add_noise_mode"] == "poisson":
                rate = torch.tensor([2.0])  # 泊松分布的参数
                noise = torch.poisson(rate.expand(memory_tokens.size()))  # 生成泊松分布的噪音
                noise = noise.float()
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise * noise_rate

        return self.cls(out), memory_tokens # 原来是正常的out和 memory_tokens

# if __name__ == "__main__":
#     inputs = torch.randn(config["batch_size"], step, 1, 28, 28).cuda() # [bs, step, c, h, w]
#     model = TokenTuringMachineEncoder().cuda()
#     out, mem = model(inputs)
#     print(out.shape)
