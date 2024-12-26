import math
from torch import nn
from torch.nn import functional as F

from modules import Conv1d1x1, ResidualConv1dGLU
from upsample import ConvInUpsampleNetwork


def receptive_field_size(
    total_layers, num_cycles, kernel_size, dilation=lambda x: 2**x
):
    """
    计算感受野大小。

    参数说明:
        total_layers (int): 总层数。
        num_cycles (int): 循环次数。
        kernel_size (int): 卷积核大小。
        dilation (Callable[[int], int], 可选): 用于计算膨胀因子的函数。默认为 `lambda x: 2**x`，即指数级膨胀。
            例如，可以使用 `lambda x: 1` 来禁用膨胀卷积。

    返回:
        int: 感受野大小（以样本数为单位）。

    计算原理:
        感受野大小是指输入信号中影响输出特征图的一个元素的空间范围。
        对于卷积神经网络，感受野大小的计算公式为:
            R = (kernel_size - 1) * sum(dilations) + 1
        其中，dilations 是每一层的膨胀因子列表。
    """
    assert total_layers % num_cycles == 0

    # 每个循环的层数
    layers_per_cycle = total_layers // num_cycles
    # 计算每一层的膨胀因子列表
    # 计算感受野大小
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class WaveNet(nn.Module):
    """
    WaveNet 类实现了一个支持局部和全局条件输入的 WaveNet 模型。
    WaveNet 是一种深度卷积神经网络，广泛应用于音频生成和语音合成任务。
    该模型通过堆叠多个残差卷积层，逐步增加感受野，从而捕捉音频数据中的长距离依赖关系。

    参数说明:
    cfg: 配置参数对象，包含以下字段:
    VOCODER:
        SCALAR_INPUT (bool): 如果为 True，则输入为标量 ([-1, 1])；否则，输入为量化的 one-hot 向量。
        OUT_CHANNELS (int): 输出通道数。如果输入类型是 mu-law 量化的 one-hot 向量，则必须等于量化通道数；否则，为 num_mixtures x 3 (pi, mu, log_scale)。
        INPUT_DIM (int): mel 频谱维度数。
        RESIDUAL_CHANNELS (int): 残差输入/输出通道数。
        LAYERS (int): 总层数。
        STACKS (int): 膨胀循环次数。
        GATE_CHANNELS (int): 门控激活通道数。
        KERNEL_SIZE (int): 卷积层的卷积核大小。
        SKIP_OUT_CHANNELS (int): 跳跃连接通道数。
        DROPOUT (float): Dropout 层的失活概率。
        UPSAMPLE_SCALES (List[int]): 上采样比例列表。`np.prod(upsample_scales)` 必须等于跳步大小。仅在启用 upsample_conditional_features 时使用。
        MEL_FRAME_PAD (int): mel 频谱帧填充量。
    """

    def __init__(self, cfg):
        super(WaveNet, self).__init__()
        self.cfg = cfg
        self.scalar_input = self.cfg.VOCODER.SCALAR_INPUT
        self.out_channels = self.cfg.VOCODER.OUT_CHANNELS
        self.cin_channels = self.cfg.VOCODER.INPUT_DIM
        self.residual_channels = self.cfg.VOCODER.RESIDUAL_CHANNELS
        self.layers = self.cfg.VOCODER.LAYERS
        self.stacks = self.cfg.VOCODER.STACKS
        self.gate_channels = self.cfg.VOCODER.GATE_CHANNELS
        self.kernel_size = self.cfg.VOCODER.KERNEL_SIZE
        self.skip_out_channels = self.cfg.VOCODER.SKIP_OUT_CHANNELS
        self.dropout = self.cfg.VOCODER.DROPOUT
        self.upsample_scales = self.cfg.VOCODER.UPSAMPLE_SCALES
        self.mel_frame_pad = self.cfg.VOCODER.MEL_FRAME_PAD

        assert self.layers % self.stacks == 0

        # 每个循环的层数
        layers_per_stack = self.layers // self.stacks
        if self.scalar_input:
            # 如果使用标量输入，则创建 1x1 卷积层
            self.first_conv = Conv1d1x1(1, self.residual_channels)
        else:
            # 否则，创建 1x1 卷积层
            self.first_conv = Conv1d1x1(self.out_channels, self.residual_channels)

        # 卷积层列表
        self.conv_layers = nn.ModuleList()
        for layer in range(self.layers):
            # 计算当前层的膨胀因子
            dilation = 2 ** (layer % layers_per_stack)
            # 创建残差卷积层
            conv = ResidualConv1dGLU(
                self.residual_channels,
                self.gate_channels,
                kernel_size=self.kernel_size,
                skip_out_channels=self.skip_out_channels,
                bias=True,
                dilation=dilation,
                dropout=self.dropout,
                cin_channels=self.cin_channels,
            )
            # 添加到卷积层列表中
            self.conv_layers.append(conv)

        # 最后的卷积层列表
        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(inplace=True), # ReLU 激活函数
                Conv1d1x1(self.skip_out_channels, self.skip_out_channels), # 1x1 卷积层
                nn.ReLU(inplace=True), # ReLU 激活函数
                Conv1d1x1(self.skip_out_channels, self.out_channels), # 1x1 卷积层
            ]
        )

        # 创建上采样网络
        self.upsample_net = ConvInUpsampleNetwork(
            upsample_scales=self.upsample_scales,
            cin_pad=self.mel_frame_pad,
            cin_channels=self.cin_channels,
        )

        # 计算感受野大小
        self.receptive_field = receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def forward(self, x, mel, softmax=False):
        """
        前向传播方法，执行模型的前向计算。

        参数:
            x (Tensor): 音频信号的 one-hot 编码，形状为 (B x C x T)。
            mel (Tensor): 局部条件特征，形状为 (B x cin_channels x T)。
            softmax (bool): 是否应用 softmax。

        返回:
            Tensor: 输出，形状为 (B x out_channels x T)。
        """
        # 获取批量大小、通道数和序列长度
        B, _, T = x.size()

        # 对条件特征进行上采样
        mel = self.upsample_net(mel)
        assert mel.shape[-1] == x.shape[-1]

        # 执行初始卷积
        x = self.first_conv(x)
        # 初始化跳跃连接总和
        skips = 0
        for f in self.conv_layers:
            # 执行残差卷积层
            x, h = f(x, mel)
            # 累积跳跃连接
            skips += h
        # 对跳跃连接进行缩放
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # 将跳跃连接赋值给 x
        x = skips
        for f in self.last_conv_layers:
            # 执行最后的卷积层
            x = f(x)

        # 如果需要，应用 softmax
        x = F.softmax(x, dim=1) if softmax else x

        # 返回输出
        return x

    def clear_buffer(self):
        """
        清除缓冲区，清除所有卷积层的缓冲区。
        """
        # 清除初始卷积层的缓冲区
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            # 清除每个残差卷积层的缓冲区
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                # 尝试清除最后的卷积层的缓冲区
                f.clear_buffer()
            except AttributeError:
                # 如果没有缓冲区，则忽略
                pass

    def make_generation_fast_(self):
        """
        优化模型以加快生成速度。
        """
        def remove_weight_norm(m):
            try:
                # 移除权重归一化
                nn.utils.remove_weight_norm(m)
            except ValueError:  
                # 如果模块没有权重归一化，则忽略
                return

        # 应用移除权重归一化的函数到模型的所有层
        self.apply(remove_weight_norm)
