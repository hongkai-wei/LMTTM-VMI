import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearnerModule(nn.Module):
# The value of dropout_rate is 0. which means that dropout is not used.
    def __init__(self, in_channels, summerize_num_tokens, num_groups, dropout_rate):

        super(TokenLearnerModule, self).__init__()
        self.in_channels = in_channels
        self.summerize_num_tokens = summerize_num_tokens
        # in_channels and out_channels must both be divisible by groups
        self.num_groups = num_groups
        # num_groups is the number of groups for grouped convolution, 
        # in_channels is the number of input channels,
        # and summerize_num_tokens is the number of tokens.

        # Operates on the last axis (c) of the input data.
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.GELU(),
            nn.Conv1d(self.in_channels, self.summerize_num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        # After conversion to 1D convolution, the shape of the inputs and outputs need to be adjusted accordingly.

        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        selected = inputs
        selected = self.norm(selected) # Shape:  [bs, dim, mem_size+special_num_token]
        selected = selected.permute(0, 2, 1)  # Shape:  [bs, dim, mem_size+special_num_token]
        selected = self.attention_maps(selected) # Shape:  [bs, num_tokens, mem_size+special_num_token]
        selected = F.softmax(selected, dim=-1) # Shape:  [bs, num_tokens, mem_size+special_num_token]

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 2, 1) # Shape:  [bs, dim, mem_size+special_num_token]
        feat = self.feat_conv(feat) # Shape:  [bs, dim, mem_size+special_num_token]
        feat = self.gelu(feat) # Shape:  [bs, dim, mem_size+special_num_token]
        feat = feat.permute(0, 2, 1) # Shape:  [bs, mem_size+special_num_token, dim]
        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        # 64 8 784   64         
        outputs = self.dropout(outputs)

        return outputs


class TokenLearnerModuleV11(nn.Module):

    # The value of dropout_rate is 0. which means that dropout is not used.
    def __init__(self, in_channels, summerize_num_tokens, num_groups, dropout_rate):

        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.summerize_num_tokens = summerize_num_tokens
        # in_channels and out_channels must both be divisible by groups
        self.num_groups = num_groups
        # num_groups is the number of groups for grouped convolution, 
        # in_channels is the number of input channels,
        # and summerize_num_tokens is the number of tokens.

        # Operates on the last axis (c) of the input data.
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(nn.Linear(self.in_channels, self.in_channels),
                                            nn.GELU(),
                                            nn.Linear(self.in_channels, self.summerize_num_tokens))
        # After conversion to 1D convolution, the shape of the inputs and outputs need to be adjusted accordingly.
        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, inputs):

        selected = inputs
        selected = self.norm(selected)
        # Shape:  [bs, dim, mem_size+special_num_token]
        selected = selected.permute(0, 2, 1)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = selected.transpose(1, 2)
        selected = self.attention_maps(selected)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        # Shape:  [bs, dim, mem_size+special_num_token]
        feat = feat.permute(0, 2, 1)
        # Shape: [bs, dim, mem_size+special_num_token].
        feat = self.feat_conv(feat)
        feat = self.gelu(feat)
        # Shape: [bs, mem_size+special_num_token, dim].
        feat = feat.permute(0, 2, 1)
        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs
    
class TokenLearnerModuleV12(nn.Module):
# Unlike the first two TokenLearners, this one is Memory-driven, not input-driven.
# The value of dropout_rate is 0. which means that dropout is not used.
    def __init__(self, in_tokens, summerize_num_tokens, num_groups, dropout_rate, dim):

        super(TokenLearnerModuleV12, self).__init__()
        self.in_tokens = in_tokens
        self.summerize_num_tokens = summerize_num_tokens
        # in_channels and out_channels must both be divisible by groups
        self.num_groups = num_groups
        # num_groups is the number of groups for grouped convolution, 
        # in_channels is the number of input channels,
        # and summerize_num_tokens is the number of tokens.

        # Operates on the last axis (c) of the input data.
        self.norm = nn.LayerNorm(self.in_tokens)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(in_tokens, in_tokens, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.GELU(),
            nn.Conv1d(self.in_tokens, self.summerize_num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        # After conversion to 1D convolution, the shape of the inputs and outputs need to be adjusted accordingly.

        self.feat_conv = nn.Conv1d(
            self.in_tokens, dim, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        selected = inputs
        selected = selected.permute(0, 2, 1)
        selected = self.norm(selected) # Shape:  [bs, dim, mem_size+special_num_token]
        selected = selected.permute(0, 2, 1)
        selected = self.attention_maps(selected) # Shape:  [bs, num_tokens, mem_size+special_num_token]
        selected = F.softmax(selected, dim=-1) # Shape:  [bs, num_tokens, mem_size+special_num_token]

        # Reshape the input to align it with the output of the conv block.
        feat = inputs

        feat = self.feat_conv(feat) # Shape:  [bs, dim, mem_size+special_num_token]
        feat = self.gelu(feat) # Shape:  [bs, dim, mem_size+special_num_token]
        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs