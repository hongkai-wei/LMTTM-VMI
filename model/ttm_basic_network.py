import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
from .tokenLearner_network import TokenLearnerModuleV11
from config.configure import Config

config = Config.getInstance()
batch_size = config["batch_size"]
config = config["model"]
drop_r = config["drop_r"]
process_unit_mode = config["process_unit_mode"]
summarize_mode = config["summarize_mode"]
in_channels = config["in_channels"]
dim = config["dim"]
memory_tokens_size = config["memory_tokens_size"]
step = config["step"]
patch_size = config["patch_size"]
num_tokens = config["num_tokens"]
use_positional_embedding = config["use_positional_embedding"]


class PreProcess(nn.Module):  # 输入B,C,STEP,H,W  最终得到B C STEP TOKEN -> B STEP TOKEN C
    def __init__(self) -> None:
        super(PreProcess, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, 
                             out_channels=dim,
                             kernel_size=patch_size, 
                             stride=patch_size, 
                             padding="valid")
        self.relu = nn.ReLU()

    def forward(self, input):
        # input=input.transpose(1,2)
        x = self.conv(input)
        x = self.relu(x)
        x = x.flatten(3)
        x = x.permute(0, 2, 3, 1)

        return x


class TokenLearnerMHA(nn.Module):
    def __init__(self) -> None:
        super(TokenLearnerMHA, self).__init__()
        self.query = nn.Parameter(torch.randn(batch_size, num_tokens, dim).cuda())
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)

    def forward(self, input):
        # [0]是结果，[1]是权重  B STEP 8 C
        return self.attn(self.query, input, input)[0]


# mem B,SIZE,DIM  control B,8,DIM    #OUT B,SIZE,DIM
class TokenAddEraseWrite(nn.Module):
    def __init__(self) -> None:
        super(TokenAddEraseWrite, self).__init__()
        num_tokens = 8
        self.trasns_bolck1 = nn.Sequential(nn.LayerNorm(dim), 
                                           nn.Linear(dim, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.laynorm = nn.LayerNorm(dim)
        self.trasns_bolck2 = nn.Sequential(nn.Linear(num_tokens, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.trasns_bolck2_ = nn.Sequential(nn.Linear(dim, 3*dim), 
                                            nn.Linear(3*dim, dim), 
                                            nn.GELU())
        self.trasns_bolck3 = nn.Sequential(nn.Linear(num_tokens, 3*dim), 
                                           nn.Linear(3*dim, num_tokens), 
                                           nn.GELU())
        self.trasns_bolck3_ = nn.Sequential(nn.Linear(dim, 3*dim), 
                                            nn.Linear(3*dim, dim), 
                                            nn.GELU())
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, memory_tokens, control_inputs):
        selected = self.trasns_bolck1(memory_tokens)
        selected = selected.transpose(1, 2)
        selected = self.softmax(selected)

        et = self.laynorm(control_inputs)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2(et)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2_(et)

        wet = selected.unsqueeze(-1).cuda() * et.unsqueeze(2).cuda()
        wet = 1 - wet
        wet = torch.prod(wet, dim=1)

        output = memory_tokens * wet

        at = self.laynorm(control_inputs)
        at = at.transpose(1,2)
        at = self.trasns_bolck3(at)
        at = at.transpose(1,2)
        at = self.trasns_bolck3_(at)

        wat = selected.unsqueeze(-1).cuda() * at.unsqueeze(2).cuda()
        wat = 1 - wat
        wat = torch.mean(wat, dim=1)

        output = output + wat

        return output


class TokenTuringMachineUnit(nn.Module):
    def __init__(self) -> None:
        super(TokenTuringMachineUnit, self).__init__()
        self.process_unit_mode = process_unit_mode
        self.summarize_mode = summarize_mode
        self.use_positional_embedding = use_positional_embedding
        self.tokenLearner1 = TokenLearnerModuleV11(in_channels=dim, num_tokens=num_tokens, num_groups=1)
        self.tokenLearner2 = TokenLearnerModuleV11(in_channels=dim, num_tokens=memory_tokens_size, num_groups=1)
        self.transformerBlock = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 3, dropout=0.2)
        self.tokenLearnerMHA = TokenLearnerMHA()
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

    def forward(self, input_tokens, memory_tokens):
        all_tokens = torch.cat((memory_tokens, input_tokens), dim=1)
        # add posiutional
        if self.use_positional_embedding:
            all_tokens = all_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, all_tokens.size(1), all_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            all_tokens = all_tokens + posemb_init

        if self.summarize_mode == 'TL' or self.summarize_mode == 'TL-AddErase':
            all_tokens=self.tokenLearner1(all_tokens)
        elif self.summarize_mode == 'TL-MHA':
            all_tokens=self.tokenLearnerMHA(all_tokens)

        if self.process_unit_mode == 'transformer':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.transformerBlock(output_tokens)

        elif self.process_unit_mode == 'mixer':
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
        
        elif self.process_unit_mode == 'mlp':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.norm(output_tokens)
                output_tokens = self.mlpBlock(output_tokens)
            output_tokens = self.norm(output_tokens)

        memory_input_tokens = torch.cat((memory_tokens, input_tokens, output_tokens), dim=1)

        if self.summarize_mode == 'TL':
            memory_output_tokens = self.tokenLearner2(memory_input_tokens)
        elif self.summarize_mode == 'TL-MHA':
            memory_output_tokens = self.tokenLearnerMHA(memory_input_tokens)
        elif self.summarize_mode == 'TL-AddErase':
            memory_output_tokens = self.tokenAddEraseWrite(memory_input_tokens,output_tokens)
        
        return (memory_output_tokens,output_tokens)


class TokenTuringMachineEncoder(nn.Module):
    def __init__(self) -> None:
        self.memory_tokens_size = memory_tokens_size
        super(TokenTuringMachineEncoder, self).__init__()
        self.memory_tokens = torch.zeros(batch_size, self.memory_tokens_size, dim).cuda()
        self.tokenTuringMachineUnit = TokenTuringMachineUnit()
        self.cls = nn.Linear(dim, config["out_class_num"])
        self.pre = PreProcess()
        self.relu = nn.ReLU()

    def forward(self, input, memory_tokens=None):
        input = self.pre(input)
        b, t, _, c = input.shape
        # b, t, c, _, _ = input.shape # b是batch，t是step，_是token_num，c是dim
        outs=[]
        if memory_tokens == None:
            memory_tokens = torch.zeros(b,self.memory_tokens_size,c).cuda() #  c, h, w
        else:
            memory_tokens = self.memory_tokens
        for i in range(t):
            memory_tokens, out = self.tokenTuringMachineUnit(memory_tokens, input[:,i,:,:])
            outs.append(out)

        # 满足输出的shape---自添加
        outs = torch.stack(outs, dim=1)#SHAPE [B,STEP,NUM_TOKEN,DIM]
        out = outs.view(batch_size, -1, dim) # out的shape是[batch,step*token_num,dim]
        out = out.transpose(1, 2)
        out = nn.AdaptiveAvgPool1d(1)(out) # AdaptiveAvgPool1d是自适应平均池化层，输出的形状是[batch,dim,1]
        out = out.squeeze(2)
        # print(out.shape)

        return self.cls(out), memory_tokens # 原来是正常的out和 memory_tokens

# if __name__ == "__main__":
#     inputs = torch.randn(batch_size, step, 1, 28, 28).cuda() # [bs, step, c, h, w]
#     model = TokenTuringMachineEncoder().cuda()
#     out, mem = model(inputs)
#     print(out.shape)