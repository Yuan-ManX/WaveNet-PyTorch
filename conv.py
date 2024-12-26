from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions"""
    """
    Conv1d 类扩展了 PyTorch 的 nn.Conv1d 模块，支持增量膨胀卷积（Incremental Dilated Convolutions）。
    该类主要用于处理需要逐步处理序列数据的场景，如在线推理或流式数据处理。

    参数说明:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核的大小。
        stride (int, 可选): 卷积步长，默认为1。
        padding (int, 可选): 输入的每一条边补充0的层数，默认为0。
        dilation (int, 可选): 卷积核元素之间的间距，默认为1。
        groups (int, 可选): 输入通道分组数，默认为1。
        bias (bool, 可选): 是否使用偏置，默认为True。
        padding_mode (str, 可选): 填充模式，默认为'zeros'。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 Conv1d 类实例。

        参数:
            *args: 传递给 nn.Conv1d 的位置参数。
            **kwargs: 传递给 nn.Conv1d 的关键字参数。
        """
        super().__init__(*args, **kwargs)
        # 清除输入缓冲区
        self.clear_buffer()
        # 线性化权重初始化为 None
        self._linearized_weight = None
        # 注册反向传播钩子，清除线性化权重
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        """
        增量前向传播方法，用于逐步处理输入序列。

        参数:
            input (Tensor): 输入张量，形状为 (B, T, C)。

        返回:
            Tensor: 输出张量，形状为 (B, 1, C_out)。
        """
        # 运行前向预钩子
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        # 重塑权重
        weight = self._get_linearized_weight()
        # 卷积核大小
        kw = self.kernel_size[0]
        # 膨胀因子
        dilation = self.dilation[0]

        # 批量大小
        bsz = input.size(0)
        if kw > 1:
            # 获取输入数据
            input = input.data
            if self.input_buffer is None:
                # 初始化输入缓冲区
                self.input_buffer = input.new(
                    bsz, kw + (kw - 1) * (dilation - 1), input.size(2)
                )
                self.input_buffer.zero_()
            else:
                # shift buffer
                # 移动缓冲区内容
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            # 添加最新的输入到缓冲区
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                # 根据膨胀因子调整输入
                input = input[:, 0::dilation, :].contiguous()
        # 执行线性变换
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        # 重塑输出形状
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        """
        清除输入缓冲区。
        """
        self.input_buffer = None

    def _get_linearized_weight(self):
        """
        获取线性化权重。

        返回:
            Tensor: 线性化后的权重张量，形状为 (out_channels, kw * in_channels)。
        """
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            # nn.Conv1d 的权重形状为 (out_channels, in_channels, kw)
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                # 转置权重
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                # fairseq.modules.conv_tbc.ConvTBC 的权重形状为 (out_channels, kw, in_channels)
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            # 重塑为二维张量
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        """
        清除线性化权重。
        """
        self._linearized_weight = None
