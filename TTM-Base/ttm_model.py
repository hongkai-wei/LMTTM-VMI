import torch
import torch.nn as nn
# einops是一个用于操作tensor的库，可以用于改变tensor的形状，维度等
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tokenlearner import TokenLearner, TokenLearnerModuleV11
import torch.nn.init as init

batch = 64       # 批量大小
step = 28          # 步长
in_channels = 1    # 输入通道数
dim = 64          # 维度
patch_size = 3     # Patch 大小
token_num = 64     # 标记数量
num_tokens = 27  # 生成标记数量 8


# 输入 batch step 1 28 28   变换到  batch step len dim  方便进行时序处理     #我觉得是没加cls
class PreProcess(nn.Module):
    """定义了一个数据预处理层，
    它包括一个卷积层，
    用于将输入数据从(batch, step, 1, 28, 28)[bs,step,c,h,w]的形状转换为(batch, step, len/num_tokens, dim)的形状
    """

    def __init__(self) -> None:
        super(PreProcess, self).__init__()
        # num_patches = (image_size // patch_size) ** 2
        self.con = nn.Conv3d(in_channels=1, out_channels=dim,
                             kernel_size=patch_size, stride=patch_size, padding="valid")
        # patch_dim = in_channels * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )# 输出 batch step num_patches(token_num) dim
        # self.pos=nn.Embedding(token_num+1,dim)
        # self.ar_token=torch.arange(token_num+1).cuda()
        # self.cls=nn.Parameter(torch.randn(1,1,dim).cuda())

    def forward(self, input):
        input = input.transpose(1, 2)
        x = self.con(input)
        # flatten是在指定维度上对输入的张量进行压平操作，x的shape是[batch,step,token_num,dim]
        x = x.flatten(3)
        x = x.permute(0, 2, 3, 1)
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
    """  # [batch,token_num,dim] # 1024->8

    def __init__(self) -> None:
        super(TokenLearnerMHA, self).__init__()
        # special_num_token就是指定输出的token数
        # Parameter是一种特殊的Tensor，会被自动添加到模型的参数列表中
        self.query = nn.Parameter(torch.randn(batch, num_tokens, dim).cuda())
        # batch_first=True表示输入数据的形状为(batch, step, token_num, dim)，0.1是dropout
        self.attn = nn.MultiheadAttention(dim, 8, 0.1, batch_first=True)

    def forward(self, input):  # input.shape是[batch,step+1,token_num,dim]
        # 返回[0]的 output shape [batch,speical_num_token,dim]
        return self.attn(self.query, input, input)[0]


class TokenAddEraseWrite(nn.Module):
    def __init__(self) -> None:
        super(TokenAddEraseWrite, self).__init__()
        self.trasns_bolck1 = nn.Sequential(nn.LayerNorm(dim),
                                           nn.Linear(dim, 3 * dim),
                                           nn.GELU(),
                                           nn.Linear(3 * dim, num_tokens),
                                           nn.GELU())
        self.laynorm = nn.LayerNorm(dim)
        self.trasns_bolck2 = nn.Sequential(nn.Linear(num_tokens, 3 * dim),
                                           nn.GELU(),
                                           nn.Linear(3 * dim, num_tokens),
                                           nn.GELU())
        self.trasns_bolck2_ = nn.Sequential(nn.Linear(dim, 3 * dim),
                                            nn.GELU(),
                                            nn.Linear(3 * dim, dim),
                                            nn.GELU())
        self.trasns_bolck3 = nn.Sequential(nn.Linear(num_tokens, 3 * dim),
                                           nn.GELU(),
                                           nn.Linear(3 * dim, num_tokens),
                                           nn.GELU())
        self.trasns_bolck3_ = nn.Sequential(nn.Linear(dim, 3 * dim),
                                            nn.GELU(),
                                            nn.Linear(3 * dim, dim),
                                            nn.GELU())
        self.softmax_a = nn.Softmax(dim=-1)

    def forward(self, memory, control_inputs):  # control其实就是output
        selected = self.trasns_bolck1(memory)
        selected = selected.transpose(1, 2)
        selected = self.softmax_a(selected)  # 20   8  64

        et = self.laynorm(control_inputs)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2(et)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2_(et)

        wet = selected.unsqueeze(-1).cuda() * et.unsqueeze(2).cuda()
        wet = 1 - wet
        wet = torch.prod(wet, dim=1)

        output = memory * wet

        at = self.laynorm(control_inputs)
        at = at.transpose(1, 2)
        at = self.trasns_bolck3(at)
        at = at.transpose(1, 2)
        at = self.trasns_bolck3_(at)

        wat = selected.unsqueeze(-1).cuda() * at.unsqueeze(2).cuda()
        wat = 1 - wat
        wat = torch.mean(wat, dim=1)

        output = output + wat

        return output


# test=token_add_earse().cuda()
# c=test(mem,torch.randn(batch,speical_num_token,dim).cuda())
# print(c.shape)   # batch memsize mem——dim

class TokenTuringMachineUnit(nn.Module):
    def __init__(self) -> None:
        super(TokenTuringMachineUnit, self).__init__()
        self.TokenLearner1 = TokenLearnerModuleV11(
            in_channels=64, num_tokens=27, num_groups=1)
        self.TokenLearner2 = TokenLearnerModuleV11(
            in_channels=64, num_tokens=88, num_groups=1)
        self.TokenLearnerMHA = TokenLearnerMHA()
        self.TokenAddEraseWrite = TokenAddEraseWrite()
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim * 3, dropout=0.)
        self.mixer_sequence_block = nn.Sequential(nn.Linear(num_tokens, num_tokens * 6),
                                                  nn.GELU(),
                                                  nn.Dropout(0.3),
                                                  nn.Linear(
                                                      num_tokens * 6, num_tokens),
                                                  nn.GELU())
        self.mixer_channels__block = nn.Sequential(nn.Linear(dim, dim * 3),
                                                   nn.GELU(),
                                                   nn.Dropout(0.3),
                                                   nn.Linear(dim * 3, dim),
                                                   nn.GELU())
        self.mlp_block = nn.Sequential(nn.Linear(dim, dim * 3),
                                       nn.GELU(),
                                       nn.Dropout(0.),
                                       nn.Linear(dim * 3, dim),
                                       nn.GELU(),
                                       nn.Dropout(0.))
        self.dropout = nn.Dropout(0.5)
        self.num_layers = 3
        self.norm = nn.LayerNorm(dim)
        self.memory_mode = 'TL'
        self.processing_unit = 'mixer'
        self.use_positional_embedding = True

    def forward(self, memory_tokens, input_tokens):
        # all_token的shape是[batch,mem_size+special_num_token,dim]
        all_tokens = torch.cat((memory_tokens, input_tokens), dim=1)

        # 读之前也要加一个位置嵌入
        if self.use_positional_embedding:
            all_tokens = all_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, all_tokens.size(1), all_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            all_tokens = all_tokens + posemb_init
        # posemb_init的tokens是8
        # posemb_init的shape是[1,mem_size+special_num_token,dim]

        if self.memory_mode == 'TL' or self.memory_mode == 'TL-AddErase':
            all_tokens = self.TokenLearner1(all_tokens)
        elif self.memory_mode == 'TL-MHA':
            all_tokens = self.TokenLearnerMHA(all_tokens)

        if self.processing_unit == 'transformer':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.transformer_block(output_tokens)

        elif self.processing_unit == 'mixer':
            # all_tokens的shape是[batch,mem_size+special_num_token,dim]
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                # Token mixing，token混合
                x_output_tokens = output_tokens
                x_output_tokens = self.norm(x_output_tokens)
                # permute是将输入张量的维度换位，output_tokens的shape是[batch,dim,mem_size+special_num_token]
                x_output_tokens = x_output_tokens.permute(0, 2, 1)
                x_output_tokens = self.mixer_sequence_block(
                    x_output_tokens)  # mixer_block是一个全连接层，输入是dim维，输出是dim维
                # output_tokens的shape是[batch,mem_size+special_num_token,dim]
                x_output_tokens = x_output_tokens.permute(0, 2, 1)
                # output_tokens的shape是[batch,mem_size+special_num_token,dim]
                x_output_tokens = x_output_tokens + output_tokens
                x_output_tokens = self.dropout(x_output_tokens)

                # Channel mixing，通道混合
                y_output_tokens = self.norm(x_output_tokens)
                y_output_tokens = self.mixer_channels__block(y_output_tokens)
                y_output_tokens = self.dropout(y_output_tokens)
                output_tokens = output_tokens + y_output_tokens

        elif self.processing_unit == 'mlp':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.norm(output_tokens)
                output_tokens = self.mlp_block(output_tokens)
            output_tokens = self.norm(output_tokens)

        mem_out_tokens = torch.cat(
            (memory_tokens, input_tokens, output_tokens), dim=1)

# 在需要添加位置嵌入的地方
        if self.use_positional_embedding:
            mem_out_tokens = mem_out_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, mem_out_tokens.size(1), mem_out_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            mem_out_tokens = mem_out_tokens + posemb_init
            # posemb_init的shape是[1,mem_size+special_num_token,dim]

        if self.memory_mode == 'TL':
            mem_out_tokens = self.TokenLearner2(mem_out_tokens)
        elif self.memory_mode == 'TL-MHA':
            mem_out_tokens = self.TokenLearnerMHA(mem_out_tokens)
        elif self.memory_mode == 'TL-AddErase':
            mem_out_tokens = self.TokenAddEraseWrite(
                memory_tokens, output_tokens)

        return (mem_out_tokens, output_tokens)


class TokenTuringMachineEncoder(nn.Module):
    def __init__(self) -> None:
        self.mem_size = 88
        super(TokenTuringMachineEncoder, self).__init__()
        self.mem = torch.zeros(batch, self.mem_size, dim).cuda()
        self.TokenTuringMachineUnit = TokenTuringMachineUnit()
        self.cls = nn.Linear(dim, 11)  # cls是一个全连接层，输入是dim维，输出是11维
        self.pre = PreProcess()

    def forward(self, input, mem=None):
        input = self.pre(input)
        b, t, _, c = input.shape
        # b, t, c, _, _ = input.shape # b是batch，t是step，_是token_num，c是dim
        outs = []
        if mem == None:
            mem = self.mem  # c, h, w
        else:
            mem = mem
        for i in range(t):
            mem, out = self.TokenTuringMachineUnit(mem, input[:, i, :, :])
            # mem, out=self.TokenTuringMachineUnit(mem, input[:,i,:,:,:])
            outs.append(out)

        # 满足输出的shape---自添加
        outs = torch.stack(outs, dim=1)  # SHAPE [B,STEP,NUM_TOKEN,DIM]
        out = outs.view(batch, -1, dim)  # out的shape是[batch,step*token_num,dim]
        out = out.transpose(1, 2)
        # AdaptiveAvgPool1d是自适应平均池化层，输出的形状是[batch,dim,1]
        out = nn.AdaptiveAvgPool1d(1)(out)
        out = out.squeeze(2)
        # print(out.shape)

        return self.cls(out), mem  # 原来是正常的out和 mem


if __name__ == "__main__":
    inputs = torch.randn(batch, step, 1, 28, 28).cuda()  # [bs, step, c, h, w]
    model = TokenTuringMachineEncoder().cuda()
    out, mem = model(inputs)
    print(out.shape)
