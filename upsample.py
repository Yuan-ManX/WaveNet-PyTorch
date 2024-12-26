import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Stretch2d(nn.Module):
    """
    Stretch2d 类实现了二维拉伸（缩放）操作。
    该模块通过插值方法对输入张量在高度（y 轴）和宽度（x 轴）方向上进行缩放。
    支持多种插值模式，如 'nearest'（最近邻）、'bilinear'（双线性）等。

    参数说明:
        x_scale (float): 宽度方向上的缩放因子。
        y_scale (float): 高度方向上的缩放因子。
        mode (str, 可选): 插值模式，默认为 'nearest'（最近邻）。可选模式包括 'bilinear'（双线性）、'bicubic'（双三次）等。
    """
    def __init__(self, x_scale, y_scale, mode="nearest"):
        """
        初始化 Stretch2d 类实例。

        参数:
            x_scale (float): 宽度方向上的缩放因子。
            y_scale (float): 高度方向上的缩放因子。
            mode (str, 可选): 插值模式，默认为 'nearest'（最近邻）。
        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """
        前向传播方法，执行二维拉伸（缩放）操作。
        使用 torch.nn.functional.interpolate 进行插值缩放。
        scale_factor 参数指定缩放因子，顺序为 (高度方向, 宽度方向)。
        mode 参数指定插值模式。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, H, W)。

        返回:
            Tensor: 缩放后的输出张量，形状为 (B, C, H * y_scale, W * x_scale)。
        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


def _get_activation(upsample_activation):
    """
    获取激活函数模块。

    参数:
        upsample_activation (str): 激活函数名称，如 'ReLU', 'LeakyReLU', 'PReLU' 等。

    返回:
        nn.Module: 对应的激活函数模块。

    示例:
        如果 upsample_activation 为 'ReLU'，则返回 nn.ReLU()。
    """
    nonlinear = getattr(nn, upsample_activation)
    # 返回激活函数模块
    return nonlinear


class UpsampleNetwork(nn.Module):
    """
    UpsampleNetwork 类实现了一个上采样网络，用于在频域和时间域上对输入特征图进行上采样。
    该网络通过一系列的二维拉伸（Stretch2d）和卷积层，逐步增加特征图的尺寸。
    支持多种上采样比例和激活函数。

    参数说明:
        upsample_scales (List[int]): 上采样比例列表，每个元素表示一个维度的上采样比例。
        upsample_activation (str, 可选): 上采样过程中使用的激活函数名称，如 'ReLU', 'LeakyReLU', 'PReLU' 等。默认为 'none'，表示不使用激活函数。
        upsample_activation_params (Dict, 可选): 激活函数的关键字参数，默认为空字典。
        mode (str, 可选): 插值模式，默认为 'nearest'（最近邻）。可选模式包括 'bilinear'（双线性）、'bicubic'（双三次）等。
        freq_axis_kernel_size (int, 可选): 频率轴上的卷积核大小，默认为 1。
        cin_pad (int, 可选): 输入特征图的填充量，默认为 0。
        cin_channels (int, 可选): 输入特征图的通道数，默认为 128。
    """
    def __init__(
        self,
        upsample_scales,
        upsample_activation="none",
        upsample_activation_params={},
        mode="nearest",
        freq_axis_kernel_size=1,
        cin_pad=0,
        cin_channels=128,
    ):
        super(UpsampleNetwork, self).__init__()
        # 上采样层列表
        self.up_layers = nn.ModuleList()
        # 计算总的上采样比例
        total_scale = np.prod(upsample_scales)
        # 计算需要裁剪的填充量
        self.indent = cin_pad * total_scale

        for scale in upsample_scales:
            # 计算频率轴上的填充量
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            # 定义卷积核大小，宽度为 scale * 2 + 1，高度为 freq_axis_kernel_size
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            # 定义填充量，宽度为 scale，高度为 freq_axis_padding
            padding = (freq_axis_padding, scale)
            # 创建 Stretch2d 层，用于在宽度方向上拉伸
            stretch = Stretch2d(scale, 1, mode)
            # 创建二维卷积层，卷积核大小为 k_size，填充为 padding，不使用偏置
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            # 初始化卷积核权重为 1 / (k_size[0] * k_size[1])
            conv.weight.data.fill_(1.0 / np.prod(k_size))
            # 应用权重归一化
            conv = nn.utils.weight_norm(conv)
            # 将 Stretch2d 和卷积层添加到上采样层列表中
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            # 如果上采样激活函数不是 'none'，则添加激活函数层
            if upsample_activation != "none":
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
        """
        前向传播方法，执行上采样操作。

        参数:
            c (Tensor): 输入张量，形状为 (B, C, T)。

        返回:
            Tensor: 上采样后的输出张量，形状为 (B, C, T * total_scale)。
        """
        # 在通道维度上增加一个维度，形状变为 (B, 1, C, T)
        c = c.unsqueeze(1)
        # 遍历所有上采样层
        for f in self.up_layers:
            c = f(c)

        # 移除通道维度，形状恢复为 (B, C, T)
        c = c.squeeze(1)

        # 如果有填充量，则裁剪填充部分
        if self.indent > 0:
            c = c[:, :, self.indent : -self.indent]
        return c


class ConvInUpsampleNetwork(nn.Module):
    """
    ConvInUpsampleNetwork 类实现了一个卷积输入上采样网络。
    该网络首先对条件特征进行一维卷积处理，以捕捉更广泛的上下文信息，
    然后通过上采样网络对处理后的特征进行上采样操作。

    参数说明:
        upsample_scales (List[int]): 上采样比例列表，每个元素表示一个维度的上采样比例。
        upsample_activation (str, 可选): 上采样过程中使用的激活函数名称，如 'ReLU', 'LeakyReLU', 'PReLU' 等。默认为 'none'，表示不使用激活函数。
        upsample_activation_params (Dict, 可选): 激活函数的关键字参数，默认为空字典。
        mode (str, 可选): 插值模式，默认为 'nearest'（最近邻）。可选模式包括 'bilinear'（双线性）、'bicubic'（双三次）等。
        freq_axis_kernel_size (int, 可选): 频率轴上的卷积核大小，默认为 1。
        cin_pad (int, 可选): 输入特征图的填充量，默认为 0。
        cin_channels (int, 可选): 输入特征图的通道数，默认为 128。
    """
    def __init__(
        self,
        upsample_scales,
        upsample_activation="none",
        upsample_activation_params={},
        mode="nearest",
        freq_axis_kernel_size=1,
        cin_pad=0,
        cin_channels=128,
    ):
        super(ConvInUpsampleNetwork, self).__init__()
        # 为了捕捉条件特征中的广泛上下文信息
        # 如果 cin_pad == 0，则此卷积层没有实际意义
        # 计算卷积核大小
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(
            cin_channels, cin_channels, kernel_size=ks, padding=cin_pad, bias=False
        )
        # 创建上采样网络
        self.upsample = UpsampleNetwork(
            upsample_scales,  # 上采样比例列表
            upsample_activation,  # 上采样激活函数名称
            upsample_activation_params,  # 激活函数的关键字参数
            mode,  # 插值模式
            freq_axis_kernel_size,  # 频率轴上的卷积核大小
            cin_pad=cin_pad,  # 输入特征图的填充量
            cin_channels=cin_channels,  # 输入特征图的通道数
        )

    def forward(self, c):
        """
        前向传播方法，执行卷积输入上采样操作。

        参数:
            c (Tensor): 输入条件特征张量，形状为 (B, C, T)。

        返回:
            Tensor: 上采样后的输出张量，形状为 (B, C, T * total_scale)。
        """
        # 对输入条件特征进行一维卷积处理
        c_up = self.upsample(self.conv_in(c))
        # 返回上采样后的输出
        return c_up
