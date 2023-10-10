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

    # in_channels是输入通道数，num_tokens是标记的数量，use_sum_pooling是是否在空间注意力之后使用求和/平均值来汇总空间特征。
    def __init__(self, in_channels, num_tokens, use_sum_pooling=False):
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
        self.in_channels = in_channels  # 输入通道数
        self.num_tokens = num_tokens  # 标记的数量
        self.use_sum_pooling = use_sum_pooling
        # 归一化层，对输入张量的channels维度进行归一化，in_channels是输入通道数。
        self.norm = nn.LayerNorm(self.in_channels)

        # attention_maps的shape是(bs, n_token, h, w)
        self.attention_maps = nn.Sequential(  # attention_maps是注意力图，in_channels是输入通道数，num_tokens是标记的数量。
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            # GELU是一种激活函数，GELU(x)=x*Φ(x)，其中Φ(x)是高斯累积分布函数。
            # 2D卷积层，in_channels是输入通道数，num_tokens是标记的数量。
            # 输入张量的形状是 [bs, in_channels, h, w]，输出张量的形状是 [bs, num_tokens, h, w]。
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(
                3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(
                3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(
                3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            # This conv layer will generate the attention maps
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(
                3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid()  # Note sigmoid for [0, 1] output
        )

    # input的shape是[batch,mem_size+special_num_token,dim]
    def forward(self, inputs):
        if inputs.dim() == 3:  # inputs.dim()是输入张量的维度，如果维度为3，则执行以下操作。
            bs, hw, c = inputs.shape  # 获取输入数据的形状
            h = int(hw ** 0.5)  # 计算输入数据的平方根以获取高度
            inputs = inputs.view(bs, h, h, c)  # 使用view函数重新形状化张量
            # inputs.view(bs, h, h, c)是在保持张量元素总数不变的情况下，将张量变换为指定的形状。
            if h * h != hw:
                raise ValueError('Only square inputs supported.')

        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        # selected = self.norm(selected)
        # Shape:  [bs, c, h, w]
        selected = selected.permute(0, 3, 1, 2)
        # Shape:  [bs, n_token, h, w]
        selected = self.attention_maps(selected)
        # Shape: [bs, h, w, n_token].
        selected = selected.permute(0, 2, 3, 1)
        # contiguous是为了保证内存中的连续性，view是为了改变张量的形状。
        # feature_shape[1] * feature_shape[2]是h * w，feature_shape[0]是bs。
        # Shape: [bs, h*w, n_token].
        selected = selected.contiguous().view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)
        # Shape: [bs, n_token, h*w, 1].
        selected = selected.permute(0, 2, 1)[..., None]
        # 通过Reshape将输入张量的形状与卷积块的输出对齐。
        feat = inputs
        feat = feat.view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)[:, None, ...]
        # 对注意力图和输入进行逐元素乘法。
        # feat是输入张量，selected是注意力图，attended_inputs是经过注意力机制后的张量，shape是(bs, n_token, h*w, c)。
        # feat.shape是(bs, 1, h*w, c)，selected.shape是(bs, n_token, h*w, 1)。
        attended_inputs = feat * selected

        if self.use_sum_pooling:  # 如果use_sum_pooling为True，则使用求和池化来汇总空间特征。否则，使用平均池化来汇总空间特征。
            outputs = torch.sum(attended_inputs, dim=2)
        else:
            # 池化后的shape是(bs, n_token, c)
            outputs = torch.mean(attended_inputs, dim=2)

        return outputs


class TokenLearnerModuleV11(nn.Module):

    # dropout_rate的值是0.代表不使用dropout。
    def __init__(self, in_channels, num_tokens, num_groups, dropout_rate=0.2):

        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        # in_channels and out_channels must both be divisible by groups
        self.num_groups = num_groups
        # num_groups是分组卷积的组数，in_channels是输入通道数，num_tokens是标记的数量。
        # Operates on the last axis (c) of the input data.
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.Conv1d(self.in_channels, self.num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        # 转换为一维卷积后，输入和输出的形状也需要相应调整。
        # 具体来说，输入的形状应为 [batch, in_channels, num_tokens],
        # 输出的形状应为 [batch, num_tokens, out_channels] 。

        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        # selected的shape是[batch,mem_size+special_num_token,dim]
        selected = inputs
        # 归一化层，对输入张量的channels维度进行归一化，in_channels是输入通道数。
        selected = self.norm(selected)
        # Shape:  [bs, dim, mem_size+special_num_token]
        selected = selected.permute(0, 2, 1)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = self.attention_maps(selected)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs  # feat的shape是[batch,mem_size+special_num_token,dim]
        # Shape:  [bs, dim, mem_size+special_num_token]
        feat = feat.permute(0, 2, 1)
        # Shape: [bs, dim, mem_size+special_num_token].
        feat = self.feat_conv(feat)
        # Shape: [bs, mem_size+special_num_token, dim].
        feat = feat.permute(0, 2, 1)
        # Produced the attended inputs.
        # batch都有，所以s对应n_token，i对应mem_size+special_num_token，d对应dim，si+id => sd即n_token,dim
        # outputs的shape是[batch,n_token,dim]
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs
