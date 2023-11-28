import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
from .tokenLearner_network import TokenLearnerModule, TokenLearnerModuleV11
from config.configure import Config
import numpy as np
import torchvision.models as models
    
class PreProcess(nn.Module): 
    # Input：Batch,Channels,Step,H,W  
    # Output：Batch，Step，Tokens，Channels
    def __init__(self,config) -> None:
        super(PreProcess, self).__init__()
        self.conv = nn.Conv3d(in_channels=config["model"]["in_channels"], 
                             out_channels=config["model"]["dim"],
                             kernel_size=config["model"]["patch_size"], 
                             stride=config["model"]["patch_size"], 
                             padding="valid")
        self.relu = nn.ReLU()

    def forward(self, input):
        # input = input.transpose(1, 2)
        x = self.conv(input)
        x = self.relu(x)
        x = x.flatten(3)
        x = x.permute(0, 2, 3, 1)
        return x
    
class PreProcessV2(nn.Module):
    def __init__(self, config,patch_t=3, patch_h=3, patch_w=3) -> None:
        super(PreProcessV2, self).__init__()
        self.lay = nn.Sequential(Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', 
                                            pt=patch_t, ph=patch_h, pw=patch_w),
                                  nn.Linear(patch_h*patch_t*patch_w*config["model"]["in_channels"], config["model"]["dim"]), )

    def forward(self, x):
        x = self.lay(x)
        return x


    def __init__(self):
        super(PreProcessV2, self).__init__()
        self.resnet = models.resnet18(pretrained=False).cuda()
        self.resnet.fc = nn.Identity()
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        batch_size, channels, steps, height, width = x.size()
        x = x.view(batch_size*steps, channels, height, width)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)#128
        x = self.resnet.layer3(x) #256
        x = self.resnet.layer4(x)#512
        he,dim,h,w=x.shape
        x=x.view(batch_size,steps,-1,dim)
        return x

class TokenLearnerMHA(nn.Module):
    def __init__(self, output_tokens,config) -> None:
        super(TokenLearnerMHA, self).__init__()
        self.query = nn.Parameter(torch.randn(config["batch_size"], output_tokens, config["model"]["dim"]).cuda())
        self.attn = nn.MultiheadAttention(embed_dim=config["model"]["dim"], num_heads=8, dropout=config["model"]["drop_r"], batch_first=True)

    def forward(self, input):
        # [0]is output，[1]is weight
        return self.attn(self.query, input, input)[0]

class TokenAddEraseWrite(nn.Module):
    def __init__(self,config) -> None:
        super(TokenAddEraseWrite, self).__init__()
        self.mlp_block1 = nn.Sequential(nn.LayerNorm(config["model"]["dim"]), 
                                           nn.Linear(config["model"]["dim"], 3*config["model"]["dim"]), 
                                           nn.Linear(3*config["model"]["dim"], config["model"]["summerize_num_tokens"]), 
                                           nn.GELU())
        self.laynorm = nn.LayerNorm(config["model"]["dim"])
        self.mlp_block2 = nn.Sequential(nn.Linear(config["model"]["summerize_num_tokens"], 3*config["model"]["dim"]), 
                                           nn.Linear(3*config["model"]["dim"], config["model"]["summerize_num_tokens"]), 
                                           nn.GELU())
        self.mlp_block3 = nn.Sequential(nn.Linear(config["model"]["dim"], 3*config["model"]["dim"]), 
                                            nn.Linear(3*config["model"]["dim"], config["model"]["dim"]), 
                                            nn.GELU())
        self.mlp_block4 = nn.Sequential(nn.Linear(config["model"]["summerize_num_tokens"], 3*config["model"]["dim"]), 
                                           nn.Linear(3*config["model"]["dim"], config["model"]["summerize_num_tokens"]), 
                                           nn.GELU())
        self.mlp_block5 = nn.Sequential(nn.Linear(config["model"]["dim"], 3*config["model"]["dim"]), 
                                            nn.Linear(3*config["model"]["dim"], config["model"]["dim"]), 
                                            nn.GELU())
        self.query = nn.Parameter(torch.randn(
            config["batch_size"], config["model"]["memory_tokens_size"], config["model"]["dim"]).cuda())
        self.trans_outdim = nn.MultiheadAttention(
            embed_dim=config["model"]["dim"], num_heads=8, dropout=config["model"]["drop_r"], batch_first=True)
        AddEraseWrite_input = config["model"]["memory_tokens_size"]+config["model"]["summerize_num_tokens"]+ int((config["train"]["input_H"]-config["model"]["patch_size"])/config["model"]["patch_size"]+1) * int((config["train"]["input_W"]-config["model"]["patch_size"])/config["model"]["patch_size"]+1)
        self.fn = nn.Linear(AddEraseWrite_input, config["model"]["memory_tokens_size"])
        self.relu = nn.ReLU()
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
        output = output.transpose(1,2)
        output = self.fn(output)
        output = self.relu(output)
        output = output.transpose(1,2)
        # Changing the Output Shape
        # output = self.trans_outdim(self.query, output, output)[0]

        return output

class SimpleDNC(nn.Module):
    def __init__(self,config) -> None:
        super(SimpleDNC, self).__init__()
        self.current_flag = 0
        self.num_blocks = config['model']['num_blocks']

    def SplitMemoryTokens(self, memory_tokens):
        summerize_num_tokens = memory_tokens.size(1)
        block_size = summerize_num_tokens // self.num_blocks
        self.split_memory_tokens = []
        for i in range(self.num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            self.split_memory_tokens.append(memory_tokens[:, start:end, :])
        return self.split_memory_tokens

    def ReadFromDNC(self, memory_tokens):
        self.SplitMemoryTokens(memory_tokens)
        # 遍历每个Memory_tokens块及其相邻的2块
        k = self.current_flag % self.num_blocks
        
        current_memory_block = self.split_memory_tokens[k]
        prev_memory_block = self.split_memory_tokens[k - 1] if k > 0 else self.split_memory_tokens[7]
        next_memory_block = self.split_memory_tokens[k + 1] if k < len(self.split_memory_tokens) - 1 else self.split_memory_tokens[0]
        
        self.current_flag = self.current_flag+1

        return current_memory_block, prev_memory_block, next_memory_block   
    
    def WriteToDNC(self, write_memory_block):
        m = self.current_flag % self.num_blocks
        self.split_memory_tokens[m] = write_memory_block
        memory_tokens = torch.cat(self.split_memory_tokens, dim=1)
        return memory_tokens
    

class TokenTuringMachineUnit(nn.Module):
    def __init__(self,config) -> None:
        super(TokenTuringMachineUnit, self).__init__()
        self.tokenLearner1 = TokenLearnerModule(in_channels=config["model"]["dim"], summerize_num_tokens=config["model"]["summerize_num_tokens"], num_groups=1, dropout_rate=config["model"]["drop_r"])
        self.tokenLearner2 = TokenLearnerModule(in_channels=config["model"]["dim"], summerize_num_tokens=24, num_groups=1, dropout_rate=config["model"]["drop_r"])
        self.tokenLearnerV11_1 = TokenLearnerModuleV11(in_channels=config["model"]["dim"], summerize_num_tokens=config["model"]["summerize_num_tokens"], num_groups=1, dropout_rate=config["model"]["drop_r"])
        self.tokenLearnerV11_2 = TokenLearnerModuleV11(in_channels=config["model"]["dim"], summerize_num_tokens=config["model"]["memory_tokens_size"], num_groups=1, dropout_rate=config["model"]["drop_r"])
        self.transformerBlock = nn.TransformerEncoderLayer(d_model=config["model"]["dim"], nhead=8, dim_feedforward=config["model"]["dim"] * 3, dropout=0.2)
        self.tokenLearnerMHA1 = TokenLearnerMHA(config["model"]["summerize_num_tokens"],config)
        self.tokenLearnerMHA2 = TokenLearnerMHA(config["model"]["memory_tokens_size"],config)
        self.tokenAddEraseWrite = TokenAddEraseWrite(config)
        self.mlpBlock = nn.Sequential(nn.LayerNorm(config["model"]["dim"]),
                                 nn.Linear(config["model"]["dim"], config["model"]["dim"]*3),
                                 nn.Dropout(config["model"]["drop_r"]),
                                 nn.GELU(),
                                 nn.Linear(config["model"]["dim"]*3, config["model"]["dim"]),
                                 nn.GELU(),
                                 nn.Dropout(config["model"]["drop_r"]))
        self.num_layers = 3
        self.norm = nn.LayerNorm(config["model"]["dim"])
        self.mixer_sequence_block = nn.Sequential(nn.Linear(config["model"]["summerize_num_tokens"], config["model"]["summerize_num_tokens"] * 6),
                                                  nn.GELU(),
                                                  nn.Dropout(config["model"]["drop_r"]),
                                                  nn.Linear(config["model"]["summerize_num_tokens"] * 6, config["model"]["summerize_num_tokens"]),
                                                  nn.GELU())
        self.mixer_channels__block = nn.Sequential(nn.Linear(config["model"]["dim"], config["model"]["dim"] * 3),
                                                   nn.GELU(),
                                                   nn.Dropout(config["model"]["drop_r"]),
                                                   nn.Linear(config["model"]["dim"] * 3, config["model"]["dim"]),
                                                   nn.GELU())
        self.dropout = nn.Dropout(config["model"]["drop_r"])
        self.config = config

    def forward(self, current_memory_block, prev_memory_block, next_memory_block, input_tokens):

        current_all_tokens = torch.cat((current_memory_block, input_tokens), dim=1)
        prev_all_tokens = torch.cat((prev_memory_block, input_tokens), dim=1)
        next_all_tokens = torch.cat((next_memory_block, input_tokens), dim=1)

        # Read add posiutional
        if self.config["model"]["Read_use_positional_embedding"]:
            current_all_tokens = current_all_tokens.cuda()
            prev_all_tokens = prev_all_tokens.cuda()
            next_all_tokens = next_all_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, current_all_tokens.size(1), current_all_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            current_all_tokens = current_all_tokens + posemb_init
            prev_all_tokens = prev_all_tokens + posemb_init
            next_all_tokens = next_all_tokens + posemb_init

        if self.config["model"]["memory_mode"] == 'TL' or self.config["model"]["memory_mode"] == 'TL-AddErase':
            current_all_tokens=self.tokenLearner1(current_all_tokens)
            prev_all_tokens=self.tokenLearner1(prev_all_tokens)
            next_all_tokens=self.tokenLearner1(next_all_tokens)
        elif self.config["model"]["memory_mode"] == 'TL-MHA':
            current_all_tokens=self.tokenLearnerMHA1(current_all_tokens)
            prev_all_tokens=self.tokenLearnerMHA1(prev_all_tokens)
            next_all_tokens=self.tokenLearnerMHA1(next_all_tokens)

        all_tokens = torch.cat((current_all_tokens, prev_all_tokens, next_all_tokens), dim=1)

        if self.config["model"]["process_unit"] == 'transformer':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.transformerBlock(output_tokens)

        elif self.config["model"]["process_unit"] == 'mixer':
            output_tokens = all_tokens # all_tokens shape is [batch,mem_size+special_num_token,config["model"]["dim"]]
            for _ in range(self.num_layers):
                # Token mixing，different token interoperability
                x_output_tokens = output_tokens
                x_output_tokens = self.norm(x_output_tokens)
                x_output_tokens = x_output_tokens.permute(0, 2, 1) 
                x_output_tokens = self.mixer_sequence_block(x_output_tokens)
                x_output_tokens = x_output_tokens.permute(0, 2, 1)
                x_output_tokens = x_output_tokens + output_tokens
                x_output_tokens = self.dropout(x_output_tokens)

                # Channel mixing，internal token interoperability
                y_output_tokens = self.norm(x_output_tokens)
                y_output_tokens = self.mixer_channels__block(y_output_tokens)
                y_output_tokens = self.dropout(y_output_tokens)
                output_tokens = output_tokens + y_output_tokens
            output_tokens = self.norm(output_tokens)
        
        elif self.config["model"]["process_unit"] == 'mlp':
            output_tokens = all_tokens
            for _ in range(self.num_layers):
                output_tokens = self.norm(output_tokens)
                output_tokens = self.mlpBlock(output_tokens)
            output_tokens = self.norm(output_tokens)

        memory_input_tokens = torch.cat((current_memory_block, prev_memory_block, next_memory_block, input_tokens, output_tokens), dim=1)

        # Write add posiutional
        if self.config["model"]["Write_use_positional_embedding"]:
            memory_input_tokens = memory_input_tokens.cuda()
            posemb_init = torch.nn.Parameter(torch.empty(
                1, memory_input_tokens.size(1), memory_input_tokens.size(2))).cuda()
            init.normal_(posemb_init, std=0.02)
            # mem_out_tokens shape is [batch,mem_size+special_num_token,config["model"]["dim"]]
            memory_input_tokens = memory_input_tokens + posemb_init

        if self.config["model"]["memory_mode"] == 'TL':
            memory_output_tokens = self.tokenLearner2(memory_input_tokens)
        elif self.config["model"]["memory_mode"] == 'TL-MHA':
            memory_output_tokens = self.tokenLearnerMHA2(memory_input_tokens)
        elif self.config["model"]["memory_mode"] == 'TL-AddErase':
            memory_output_tokens = self.tokenAddEraseWrite(memory_input_tokens,output_tokens)
        
        return (memory_output_tokens,output_tokens)




class TokenTuringMachineEncoder(nn.Module):
    def __init__(self,config) -> None:
        super(TokenTuringMachineEncoder, self).__init__()
        self.memory_tokens = torch.zeros(config["batch_size"], config["model"]["memory_tokens_size"], config["model"]["dim"]).cuda()
        self.tokenTuringMachineUnit = TokenTuringMachineUnit(config)
        self.simpleDNC = SimpleDNC(config)
        self.cls = nn.Linear(config["model"]["dim"], config["model"]["out_class_num"])
        self.pre = PreProcess(config)
        self.relu = nn.ReLU()
        self.pre_dim =nn.Linear(512, config["model"]["dim"])
        self.config = config

    def forward(self, input, memory_tokens):
        input = self.pre(input)
        b, t, _, c = input.shape
        outs=[]
        if memory_tokens == None:
            memory_tokens = torch.zeros(b,self.config["model"]["memory_tokens_size"],c).cuda() #  c, h, w
            # np.random.seed(42)
            # random_tokens = torch.rand(b, self.config["model"]["memory_tokens_size"], c).cuda()
            # memory_tokens = torch.exp(random_tokens)
        else:
            memory_tokens = memory_tokens.detach()

        
        for i in range(t):
            # 将Memory_tokens分成8块
            current_memory_block, prev_memory_block, next_memory_block = self.simpleDNC.ReadFromDNC(memory_tokens)
            # 遍历每个Memory_tokens块及其相邻的2块
            write_memory_block, out = self.tokenTuringMachineUnit(current_memory_block, prev_memory_block, next_memory_block, input[:, i, :, :])
            memory_tokens = self.simpleDNC.WriteToDNC(write_memory_block)
            outs.append(out)
    
        outs = torch.stack(outs, dim=1)
        out = outs.view(self.config["batch_size"], -1, self.config["model"]["dim"])
        out = out.transpose(1, 2)
        out = nn.AdaptiveAvgPool1d(1)(out) 
        out = out.squeeze(2)

        if self.config["model"]["load_memory_add_noise"]:
            np.random.seed(42)
            if self.config["model"]["load_memory_add_noise_mode"] == "normal":
                noise = torch.randn_like(memory_tokens)
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise_rate * noise
            elif self.config["model"]["load_memory_add_noise_mode"] == "laplace":
                noise = torch.distributions.laplace.Laplace(loc = 10, scale = 10).sample(memory_tokens.size())
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif self.config["model"]["load_memory_add_noise_mode"] == "uniform":
                noise = torch.FloatTensor(memory_tokens.size()).uniform_(-0.5, 0.5)
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif self.config["model"]["load_memory_add_noise_mode"] == "exp":
                noise = torch.empty(memory_tokens.size()).exponential_()
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif self.config["model"]["load_memory_add_noise_mode"] == "gamma":
                shape = torch.tensor([2.0])  # Shape parameters of the Gamma distribution
                scale = torch.tensor([2.0])  # Scale parameters of the Gamma distribution
                noise = torch.empty(memory_tokens.size())  # Create the same empty tensor as the noise tensor
                noise = noise.cuda()
                noise.copy_(torch.from_numpy(np.random.gamma(shape.item(), scale.item(), size=noise.size())))  # 将正态分布随机数转化为Gamma分布随机数
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise*noise_rate
            elif self.config["model"]["load_memory_add_noise_mode"] == "poisson":
                rate = torch.tensor([2.0])  # Parameters of the Poisson distribution
                noise = torch.poisson(rate.expand(memory_tokens.size()))  # Generating Poisson distributed noise
                noise = noise.float()
                noise = noise.cuda()
                noise_rate = 0.3
                memory_tokens = memory_tokens + noise * noise_rate

        return self.cls(out), memory_tokens
    
# if __name__ == "__main__":
#     inputs = torch.randn(config["batch_size"], config["model"]["step"], 1, 28, 28).cuda() # [bs, config["model"]["step"], c, h, w]
#     model = TokenTuringMachineEncoder().cuda()
#     out, mem = model(inputs)
#     print(out.shape)
