import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    """TokenLearner module.

    This is the module used for the experiments in the paper.

    Attributes:
      num_tokens: Number of tokens.
      use_sum_pooling: Whether to use the sum/average to aggregate the spatial feature after spatial attention
    TokenLearner 模块。

    这是用于论文中实验的模块。

    属性：
    num_tokens：标记的数量。
    use_sum_pooling：是否在空间注意力之后使用求和/平均值来汇总空间特征。"
    """
    def __init__(self, in_channels, num_tokens, use_sum_pooling=False): # in_channels是输入通道数，num_tokens是标记的数量，use_sum_pooling是是否在空间注意力之后使用求和/平均值来汇总空间特征。
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.

          对2D输入应用可学习的tokenization。

        参数：
        inputs：形状为 [bs, h, w, c] 的输入。

        返回值：
        形状为 [bs, n_token, c] 的输出。    h * w -> n_token x 
        """
        super(TokenLearner, self).__init__()
        self.in_channels = in_channels # 输入通道数
        self.num_tokens = num_tokens # 标记的数量
        self.use_sum_pooling = use_sum_pooling
        self.norm = nn.LayerNorm(self.in_channels)  # 归一化层，对输入张量的channels维度进行归一化，in_channels是输入通道数。

        # attention_maps的shape是(bs, n_token, h, w)
        self.attention_maps = nn.Sequential( # attention_maps是注意力图，in_channels是输入通道数，num_tokens是标记的数量。
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            # GELU是一种激活函数，GELU(x)=x*Φ(x)，其中Φ(x)是高斯累积分布函数。
            # 2D卷积层，in_channels是输入通道数，num_tokens是标记的数量。
            # 输入张量的形状是 [bs, in_channels, h, w]，输出张量的形状是 [bs, num_tokens, h, w]。
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            # This conv layer will generate the attention maps
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid()  # Note sigmoid for [0, 1] output
        )
        
    def forward(self, inputs): # input的shape是[batch,mem_size+special_num_token,dim]
        if inputs.dim() == 3: # inputs.dim()是输入张量的维度，如果维度为3，则执行以下操作。
          bs, hw, c = inputs.shape  # 获取输入数据的形状
          h = int(hw ** 0.5)  # 计算输入数据的平方根以获取高度
          inputs = inputs.view(bs, h, h, c)  # 使用view函数重新形状化张量
          # inputs.view(bs, h, h, c)是在保持张量元素总数不变的情况下，将张量变换为指定的形状。
          if h * h != hw:
            raise ValueError('Only square inputs supported.')

        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        # selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)                  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)                 # Shape:  [bs, n_token, h, w]
        selected = selected.permute(0, 2, 3, 1)                  # Shape: [bs, h, w, n_token].
        # contiguous是为了保证内存中的连续性，view是为了改变张量的形状。
        # feature_shape[1] * feature_shape[2]是h * w，feature_shape[0]是bs。
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)[..., None]   # Shape: [bs, n_token, h*w, 1].
        # 通过Reshape将输入张量的形状与卷积块的输出对齐。
        feat = inputs
        feat = feat.view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)[:, None, ...] 
        # 对注意力图和输入进行逐元素乘法。
        # feat是输入张量，selected是注意力图，attended_inputs是经过注意力机制后的张量，shape是(bs, n_token, h*w, c)。
        attended_inputs = feat * selected  # feat.shape是(bs, 1, h*w, c)，selected.shape是(bs, n_token, h*w, 1)。

        if self.use_sum_pooling: # 如果use_sum_pooling为True，则使用求和池化来汇总空间特征。否则，使用平均池化来汇总空间特征。
            outputs = torch.sum(attended_inputs, dim=2)   
        else:
            outputs = torch.mean(attended_inputs, dim=2) # 池化后的shape是(bs, n_token, c)

        return outputs


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses 2 grouped conv. layers with more channels. It
    also uses softmax instead of sigmoid. We confirmed that this version works
    better when having limited training data, such as training with ImageNet1K
    from scratch.

    Attributes:
      num_tokens: Number of tokens.
      dropout_rate: Dropout rate.
      
    TokenLearner 模块版本 1.1，使用略有不同的卷积层。

    与使用小通道的 4 个卷积层来实现空间注意力不同，这个版本使用 2 个带有更多通道的分组卷积层。它还使用 softmax 而不是 sigmoid。
    我们确认了当具有有限的训练数据时（例如从头开始使用 ImageNet1K 进行训练），这个版本效果更好。

    属性：
    num_tokens：token 的数量。
    dropout_rate：dropout 率。"
    """

    def __init__(self, in_channels, num_tokens, num_groups, dropout_rate=0.):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups  # in_channels and out_channels must both be divisible by groups
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)  # Shape: [bs, n_token, h, w].
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2],
                                 -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w].
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)   # Shape:  [bs, c, h, w]
        feat = self.feat_conv(feat)      # Shape: [bs, c, h, w].
        feat = feat.permute(0, 2, 3, 1)   # Shape: [bs, h, w, c].
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token, c)
        outputs = self.dropout(outputs)

        return outputs