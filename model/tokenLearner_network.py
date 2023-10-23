import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):
    """TokenLearner module.

    This is the module used for the experiment in the paper.

    Attributes:
      num_tokens: Number of tokens.
      use_sum_pooling: Whether to use the sum/average to aggregate the spatial feature after spatial attention
    """

    # in_channels is the number of input channels.
    # num_tokens is the number of tokens that.
    # use_sum_pooling is whether to use summation after spatial attention.
    def __init__(self, in_channels, num_tokens, use_sum_pooling=False):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.

        """
        super(TokenLearner, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.use_sum_pooling = use_sum_pooling
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            # 3 layers of conv with gelu activation as suggested。
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

    def forward(self, inputs):
        if inputs.dim() == 3:
            bs, hw, c = inputs.shape
            h = int(hw ** 0.5)
            inputs = inputs.view(bs, h, h, c)
            # inputs.view(bs, h, h, c) is to transform the tensor to the specified shape while keeping the total number of tensor elements constant.
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
        # contiguous is to ensure continuity in memory and view is to change the shape of the tensor.
        selected = selected.contiguous().view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)
        # Shape: [bs, n_token, h*w, 1].
        selected = selected.permute(0, 2, 1)[..., None]
        # Align the shape of the input tensor with the output of the convolution block via Reshape.
        feat = inputs
        feat = feat.view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)[:, None, ...]
        # Perform element-by-element multiplication on the attention graph and inputs.
        attended_inputs = feat * selected

        # If use_sum_pooling is True, use summation pooling to aggregate spatial features. 
        # Otherwise, use average pooling to aggregate spatial features.
        if self.use_sum_pooling:
            outputs = torch.sum(attended_inputs, dim=2)
        else:
            outputs = torch.mean(attended_inputs, dim=2)

        return outputs


class TokenLearnerModuleV11(nn.Module):

    # The value of dropout_rate is 0. which means that dropout is not used.
    def __init__(self, in_channels, num_tokens, num_groups, dropout_rate=0.2):

        super(TokenLearnerModuleV11, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        # in_channels and out_channels must both be divisible by groups
        self.num_groups = num_groups
        # num_groups is the number of groups for grouped convolution, 
        # in_channels is the number of input channels,
        # and num_tokens is the number of tokens.

        # Operates on the last axis (c) of the input data.
        self.norm = nn.LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1,
                      stride=1, padding=0, groups=self.num_groups, bias=False),
            nn.Conv1d(self.in_channels, self.num_tokens,
                      kernel_size=1, stride=1, padding=0, bias=False),
        )
        # After conversion to 1D convolution, the shape of the inputs and outputs need to be adjusted accordingly.

        self.feat_conv = nn.Conv1d(
            self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=self.num_groups, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        selected = inputs
        selected = self.norm(selected)
        # Shape:  [bs, dim, mem_size+special_num_token]
        selected = selected.permute(0, 2, 1)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = self.attention_maps(selected)
        # Shape: [bs, n_token, mem_size+special_num_token].
        selected = F.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        # Shape:  [bs, dim, mem_size+special_num_token]
        feat = feat.permute(0, 2, 1)
        # Shape: [bs, dim, mem_size+special_num_token].
        feat = self.feat_conv(feat)
        # Shape: [bs, mem_size+special_num_token, dim].
        feat = feat.permute(0, 2, 1)
        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)
        outputs = self.dropout(outputs)

        return outputs
