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
import torch.nn.init as init
drop_r = 0.0
process_unit_mode = "mix"
summay_mode = "MHA"
mem_mode = "add_earse"
in_channels = 1
dim = 512
batch = 32
step = 28
in_channels = 1
patch_size = 4
speical_num_tokens = 8
positional_use = True


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


class token_mha(nn.Module):
    def __init__(self) -> None:
        super(token_mha, self).__init__()
        speical_num_token = speical_num_tokens
        self.query = nn.Parameter(torch.randn(
            batch, speical_num_token, dim).cuda())
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)

    def forward(self, input):
        # [0]是结果，[1]是权重  B STEP 8 C
        return self.attn(self.query, input, input)[0]


# mem B,SIZE,DIM  control B,8,DIM    #OUT B,SIZE,DIM
class token_add_earse(nn.Module):
    def __init__(self) -> None:
        super(token_add_earse, self).__init__()
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
        self.mem_mode = mem_mode
        self.process_unit = process_unit_mode
        self.summay_mode = summay_mode
        self.use_positional_embedding = positional_use
        self.token_mha = token_mha()
        self.token_add_earse = token_add_earse()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(
            dim, dim*3), nn.Dropout(drop_r), nn.GELU(), nn.Linear(dim*3, dim), nn.GELU(), nn.Dropout(drop_r))
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
        all_token = torch.cat((mem, step_input), dim=1)
# add posiutional
        if self.use_positional_embedding:
            all_token = all_token.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, all_token.size(1), all_token.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens的shape是[batch,mem_size+special_num_token,dim]
            all_token = all_token + posemb_init

################
        if self.summay_mode == "MHA":
            all_token = self.token_mha(all_token)
        elif self.summay_mode == "token_learner":
            '''       修改_这里加上token_learner的代码     '''
            pass

        output_token = all_token

        if self.process_unit == "mix":
            output_tokens = output_token
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
                output_token = output_tokens + y_output_tokens
        elif self.process_unit == "mlp":
            for i in range(self.lay):
                output_token = self.mlp(output_token)
            output_token = self.norm(output_token)
        elif self.process_unit == "transformer":
            pass

        mem_out_tokens = torch.cat((mem, step_input, output_token), dim=1)

        if self.mem_mode == "add_earse":
            mem_out_tokens = self.token_add_earse(mem, output_token)
        elif self.mem_mode == "token_learner":
            pass
        elif self.mem_mode == "MHA":
            pass

        return (mem_out_tokens, output_token)


class ttm(nn.Module):
    def __init__(self) -> None:
        self.mem_size = 128
        step = 28
        speical_token = 8
        super(ttm, self).__init__()
        self.memmo = torch.zeros(batch, self.mem_size, dim).cuda()
        self.ttm_unit = ttm_unit()
        self.cls = nn.Linear(dim, 11)
        self.pre = pre_procee()
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(
            (step//patch_size)*speical_token, out_features=1)
        self.laynorm = nn.LayerNorm(512)

    def forward(self, input, mem=None):
        input = self.pre(input)
        b, t, len, c = input.shape
        outs = []
        if mem == None:
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
        # out=nn.AdaptiveAvgPool1d(1)(out)
        out = out.squeeze(2)
        out = self.relu(out)
        return self.cls(out), self.mem


if __name__ == "__main__":
    pass
