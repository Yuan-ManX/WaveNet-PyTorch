import torch
import math
from torch import nn
from torch.nn import functional as F

from conv import Conv1d as conv_Conv1d


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """
    创建并初始化一个带权重归一化的 1D 卷积层。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小。
        dropout (float, 可选): Dropout 失活概率，默认为0。
        **kwargs: 其他传递给 nn.Conv1d 的关键字参数。

    返回:
        nn.Module: 初始化后的带权重归一化的 1D 卷积层。
    """
    # 创建一个 1D 卷积层
    m = conv_Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    # 使用 Kaiming 正态分布初始化权重，激活函数为 ReLU
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    # 如果存在偏置，则将偏置初始化为常数0
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    # 应用权重归一化
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True):
    """
    创建一个 1x1 的 1D 卷积层。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        bias (bool, 可选): 是否使用偏置，默认为 True。

    返回:
        nn.Module: 初始化后的 1x1 的 1D 卷积层。
    """
    # 使用 Conv1d 函数创建一个 1x1 的卷积层
    return Conv1d(
        in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
    )


def _conv1x1_forward(conv, x, is_incremental):
    """
    执行 1x1 卷积的前向传播。

    参数:
        conv (nn.Module): 1x1 卷积层。
        x (Tensor): 输入张量。
        is_incremental (bool): 是否为增量模式。

    返回:
        Tensor: 卷积后的输出张量。
    """
    if is_incremental:
        # 如果是增量模式，则使用增量前向传播
        x = conv.incremental_forward(x)
    else:
        # 否则，正常执行前向传播
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """
    ResidualConv1dGLU 类实现了残差膨胀卷积1D（Gated Linear Unit）模块。
    该模块结合了残差连接和门控线性单元（GLU），用于处理序列数据中的长距离依赖关系。

    参数说明:
        residual_channels (int): 残差输入/输出的通道数。
        gate_channels (int): 门控激活的通道数。
        kernel_size (int): 卷积层的卷积核大小。
        skip_out_channels (int, 可选): 跳跃连接的通道数。如果未指定，则设置为与 `residual_channels` 相同。
        cin_channels (int): 局部条件输入的通道数。如果设置为负值，则禁用局部条件输入。
        dropout (float): Dropout 层的失活概率。
        padding (int, 可选): 卷积层的填充。如果未指定，则根据膨胀因子和卷积核大小计算适当的填充。
        dilation (int): 膨胀因子。
        causal (bool): 是否使用因果卷积。
        bias (bool): 是否使用偏置。
        *args: 其他传递给 Conv1d 的位置参数。
        **kwargs: 其他传递给 Conv1d 的关键字参数。
    """

    def __init__(
        self,
        residual_channels,
        gate_channels,
        kernel_size,
        skip_out_channels=None,
        cin_channels=-1,
        dropout=1 - 0.95,
        padding=None,
        dilation=1,
        causal=True,
        bias=True,
        *args,
        **kwargs,
    ):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout

        if skip_out_channels is None:
            # 如果未指定跳跃连接通道数，则设置为与残差通道数相同
            skip_out_channels = residual_channels
        if padding is None:
            # no future time stamps available
            # 如果未指定填充，则根据因果性计算适当的填充
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        # 是否使用因果卷积
        self.causal = causal

        # 创建膨胀卷积层
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
            *args,
            **kwargs,
        )

        # mel conditioning
        # 局部条件输入的卷积层
        self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)

        # 门控输出通道数
        gate_out_channels = gate_channels // 2
        # 创建用于输出残差连接的卷积层
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        # 创建用于跳跃连接的卷积层
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

    def forward(self, x, c=None):
        """
        前向传播方法，执行残差膨胀卷积1D + GLU 操作。

        参数:
            x (Tensor): 输入张量，形状为 (B, C_res, T)。
            c (Tensor, 可选): 局部条件输入张量，形状为 (B, C_cin, T)。

        返回:
            Tensor: 输出张量，形状为 (B, C_res + C_skip, T)。
        """
        return self._forward(x, c, False)

    def incremental_forward(self, x, c=None):
        """
        增量前向传播方法，用于逐步处理输入序列。

        参数:
            x (Tensor): 输入张量，形状为 (B, C_res, T)。
            c (Tensor, 可选): 局部条件输入张量，形状为 (B, C_cin, T)。

        返回:
            Tuple[Tensor, Tensor]: 输出残差连接和跳跃连接张量，形状均为 (B, C_res, T)。
        """
        return self._forward(x, c, True)

    def clear_buffer(self):
        """
        清除缓冲区，清除所有卷积层的缓冲区。
        """
        for c in [
            self.conv,
            self.conv1x1_out,
            self.conv1x1_skip,
            self.conv1x1c,
        ]:
            if c is not None:
                c.clear_buffer()

    def _forward(self, x, c, is_incremental):
        """
        前向传播方法，支持增量模式。

        参数:
            x (Tensor): 输入张量，形状为 (B, C_res, T)。
            c (Tensor, 可选): 局部条件输入张量，形状为 (B, C_cin, T)。
            is_incremental (bool): 是否为增量模式。

        返回:
            Tuple[Tensor, Tensor]: 输出残差连接和跳跃连接张量，形状均为 (B, C_res, T)。
        """
        # 保存残差连接输入
        residual = x
        # 应用 Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            # 分割维度为最后一个维度
            splitdim = -1
            # 执行增量前向传播
            x = self.conv.incremental_forward(x)
        else:
            # 分割维度为第二个维度
            splitdim = 1
            # 执行卷积操作
            x = self.conv(x)
            # 如果是因果卷积，则移除未来的时间步
            x = x[:, :, : residual.size(-1)] if self.causal else x

        # 分割卷积输出为两部分
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        assert self.conv1x1c is not None
        # 执行 1x1 卷积
        c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
        # 分割条件输入
        ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
        # 条件输入与卷积输出相加
        a, b = a + ca, b + cb

        # 执行 GLU 操作
        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        # 跳跃连接
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        # 残差连接
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        # 残差连接并乘以缩放因子
        x = (x + residual) * math.sqrt(0.5)
        return x, s
