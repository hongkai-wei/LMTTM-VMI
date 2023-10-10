
from config.configure import Config
from model.tokenLearner_network import TokenLearnerModuleV11
import torch.nn.init as init
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
import einops
import torch.nn as nn
import torch
import sys
sys.path.append("F:\_TTM\TTM-Pytorch")

configs = Config.getInstance()
batch = configs["dataset"]["batch_size"]
config = configs["model"]
drop_r = config["drop_r"]
process_unit_mode = config["process_unit_mode"]
# summay_mode = config["summay_mode"]
mem_mode = config["mem_mode"]
in_channels = config["in_channels"]
dim = config["dim"]

step = config["step"]

patch_size = 4
speical_num_tokens = config["speical_token"]
positional_use = config["positional_use"]


class pre_procee(nn.Module):  # 输入B,C,STEP,H,W  最终得到B C STEP TOKEN -> B STEP TOKEN C
    def __init__(self) -> None:
        super(pre_procee, self).__init__()
        self.con = nn.Conv3d(in_channels=in_channels, out_channels=dim,
                             kernel_size=patch_size, stride=patch_size, padding="valid")
        self.act = nn.ReLU()

    def forward(self, input):
        # input=input.transpose(1,2)
        x = self.con(input)
        x = self.act(x)
        x = x.flatten(3)
        x = x.permute(0, 2, 3, 1)
        return x


class tokenLearner_mha(nn.Module):
    def __init__(self) -> None:
        super(tokenLearner_mha, self).__init__()
        speical_num_token = speical_num_tokens
        self.query = nn.Parameter(torch.randn(
            batch, speical_num_token, dim).cuda())
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)

    def forward(self, input):
        # [0]是结果，[1]是权重  B STEP 8 C
        return self.attn(self.query, input, input)[0]


# mem B,SIZE,DIM  control B,8,DIM    #OUT B,SIZE,DIM
class token_add_erase_write(nn.Module):
    def __init__(self) -> None:
        super(token_add_erase_write, self).__init__()
        speical_num_token = 8
        self.trasns_bolck1 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(
                dim, 3*dim), nn.Linear(3*dim, speical_num_token), nn.GELU()
        )
        self.laynorm1 = nn.LayerNorm(dim)
        self.laynorm2 = nn.LayerNorm(dim)
        self.trasns_bolck2 = nn.Sequential(nn.Linear(
            speical_num_token, 3*dim), nn.Linear(3*dim, speical_num_token), nn.GELU())
        self.trasns_bolck2_ = nn.Sequential(
            nn.Linear(dim, 3*dim), nn.Linear(3*dim, dim), nn.GELU())
        self.trasns_bolck3 = nn.Sequential(nn.Linear(
            speical_num_token, 3*dim), nn.Linear(3*dim, speical_num_token), nn.GELU())
        self.trasns_bolck3_ = nn.Sequential(
            nn.Linear(dim, 3*dim), nn.Linear(3*dim, dim), nn.GELU())
        self.softmax_a = nn.Softmax(dim=-1)

    def forward(self, mem, control):
        select = self.trasns_bolck1(mem)
        select = select.transpose(1, 2)
        select = self.softmax_a(select)
        et = self.laynorm1(control)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2(et)
        et = et.transpose(1, 2)
        et = self.trasns_bolck2_(et)
        temp_sele = select.unsqueeze(-1).cuda()
        temp_et = et.unsqueeze(2).cuda()
        wet = temp_sele*temp_et
        wet = 1-wet
        wet = torch.prod(wet, dim=1)
        output = mem*wet
        at = self.laynorm2(control)
        at = at.transpose(1, 2)
        at = self.trasns_bolck3(at)
        at = at.transpose(1, 2)
        at = self.trasns_bolck3_(at)
        temp_at = at.unsqueeze(2).cuda()
        wat = temp_sele*temp_at
        wat = 1-wat
        wat = torch.mean(wat, dim=1)
        output = output+wat
        return output


class ttm_unit(nn.Module):
    def __init__(self) -> None:
        super(ttm_unit, self).__init__()

        self.process_unit = process_unit_mode
        self.memory_mode = mem_mode
        self.use_positional_embedding = positional_use
        self.tokenLearner1 = TokenLearnerModuleV11(
            in_channels=512, num_tokens=16, num_groups=1)
        self.tokenLearner2 = TokenLearnerModuleV11(
            in_channels=512, num_tokens=1024, num_groups=1)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim * 3, dropout=0.2)
        self.tokenLearner_mha = tokenLearner_mha()
        self.token_add_erase_write = token_add_erase_write()
        self.mlp = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, dim*3),
                                 nn.Dropout(drop_r),
                                 nn.GELU(),
                                 nn.Linear(dim*3, dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_r))
        self.lay = 3
        self.norm = nn.LayerNorm(dim)
        num_tokens = speical_num_tokens
        self.mixer_sequence_block = nn.Sequential(nn.Linear(num_tokens, num_tokens * 6),
                                                  nn.GELU(),
                                                  nn.Dropout(drop_r),
                                                  nn.Linear(
                                                      num_tokens * 6, num_tokens),
                                                  nn.GELU())
        self.mixer_channels__block = nn.Sequential(nn.Linear(dim, dim * 3),
                                                   nn.GELU(),
                                                   nn.Dropout(drop_r),
                                                   nn.Linear(dim * 3, dim),
                                                   nn.GELU())

    def forward(self, step_input, mem):
        all_tokens = torch.cat((mem, step_input), dim=1)
        # add posiutional
        if self.use_positional_embedding:
            all_tokens = all_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, all_tokens.size(1), all_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            all_tokens = all_tokens + posemb_init

        if self.memory_mode == "token_learner" or self.memory_mode == 'token_add_earse_write':  # 这里其实应该是本来token summary
            all_tokens = self.tokenLearner_mha(all_tokens)
        elif self.memory_mode == "tokenLearner_mha":
            pass
        else:
            all_tokens = self.tokenLearner_mha(all_tokens)

        if self.process_unit == "mix":
            output_tokens = all_tokens
            for _ in range(self.lay):
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
                x_output_tokens = nn.Dropout(drop_r)(x_output_tokens)

                # Channel mixing，通道混合
                y_output_tokens = self.norm(x_output_tokens)
                y_output_tokens = self.mixer_channels__block(y_output_tokens)
                y_output_tokens = nn.Dropout(drop_r)(y_output_tokens)
                output_tokens = output_tokens + y_output_tokens
            output_tokens = self.norm(output_tokens)

        elif self.process_unit == "mlp":
            output_tokens = all_tokens
            for _ in range(self.lay):
                output_tokens = self.mlp(output_tokens)
            output_tokens = self.norm(output_tokens)

        elif self.process_unit == "transformer":
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.transformer_block(output_tokens)
            output_tokens = self.norm(output_tokens)

        mem_out_tokens = torch.cat((mem, step_input, output_tokens), dim=1)

        if self.memory_mode == 'tokenLearner':
            mem_out_tokens = self.tokenLearner2(mem_out_tokens)
        elif self.memory_mode == 'tokenLearner_mha':
            mem_out_tokens = self.tokenLearner_mha(mem_out_tokens)
        elif self.memory_mode == 'token_add_erase_write':
            mem_out_tokens = self.token_add_erase_write(mem, output_tokens)

        return (mem_out_tokens, output_tokens)


class ttm(nn.Module):
    def __init__(self) -> None:
        self.mem_size = config["mem_size"]
        step = config["step"]
        speical_token = config["speical_token"]
        super(ttm, self).__init__()
        self.memmo = torch.zeros(batch, self.mem_size, dim).cuda()
        self.ttm_unit = ttm_unit()
        self.cls = nn.Linear(dim, config["out_channale"])
        self.pre = pre_procee()
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(
            (step//patch_size)*speical_token, out_features=1)
        self.laynorm = nn.LayerNorm(512)

    def forward(self, input, mem=None):  # B C T H W
        input = self.pre(input)
        b, t, len, c = input.shape  # B STEP TOKEN C
        outs = []
        if mem == None:  # 可以设置下，mem是否持续化，如果不持续化，就是每次都是新的mem
            self.mem = self.memmo
        else:
            self.mem = mem.detach()
        for i in range(t):
            self.mem, out = self.ttm_unit(input[:, i, :, :], self.mem)
            outs.append(out)

        outs = torch.stack(outs, dim=1)  # SHAPE [B,STEP,NUM_TOKEN,DIM]
        out = outs.view(batch, -1, dim)
        out = out.transpose(1, 2)  # add linear
        out = self.linear_1(out)
        out = out.squeeze(2)
        out = self.relu(out)
        self.mem = self.laynorm(self.mem)
        self.mem = add_noise(self.mem)
        return self.cls(out), self.mem


# def add_noise(data, mean, scale):
#     noise = torch.distributions.laplace.Laplace(
#         mean, scale).sample(data.size())
#     rate = 0.42
#     return data + noise*rate


def add_noise(mem: torch.Tensor):
    mean = 0
    scale = 1.42
    b, len, dim = mem.shape
    noise = torch.distributions.laplace.Laplace(
        mean, scale).sample((b, len, dim)).cuda()
    rate = 0.4
    return rate*noise+1*mem


if __name__ == "__main__":
    x = torch.randn(32, 1, 28, 28, 28).cuda()
    y = torch.randn(32, 1).cuda()
    model = ttm().cuda()
    mem = None
    out, mem = model(x, mem)
