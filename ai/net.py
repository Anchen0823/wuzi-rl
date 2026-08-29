"""自研残差网络：策略 + 价值双头（设计见 docs/DESIGN.md §6）。

自研边界：网络结构自行设计；实现仅使用 PyTorch 张量原语
（Tensor / autograd / functional.conv2d·batch_norm·linear），
不引入任何第三方网络结构、预训练层或权重。
所有卷积/批归一化/全连接层均为本项目手写实现。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Conv2d(nn.Module):
    """自研卷积层（F.conv2d + Kaiming 初始化）。"""

    def __init__(self, in_ch: int, out_ch: int, k: int, padding: int = 0, stride: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_ch * k * k)
        nn.init.uniform_(self.bias, -bound, bound)
        self.stride, self.padding = stride, padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class BatchNorm2d(nn.Module):
    """自研批归一化（F.batch_norm，训练时内部维护 running 统计）。"""

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            training=self.training, momentum=self.momentum, eps=self.eps,
        )


class Linear(nn.Module):
    """自研全连接层（F.linear + Kaiming 初始化）。"""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ResBlock(nn.Module):
    """残差块：x → conv1→bn1→relu→conv2→bn2 → relu(x + 残差)。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, 3, padding=1)
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = Conv2d(channels, channels, 3, padding=1)
        self.bn2 = BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GomokuNet(nn.Module):
    """五子棋残差网络：输入 (B,4,15,15) → 策略 logits (B,225) + 价值 (B,1)。

    结构（自研，见 DESIGN.md §6）：
      Stem: Conv3×3(4→256) + BN + ReLU
      N × ResBlock(256)
      策略头: Conv1×1(256→2) + BN + ReLU + Flatten + Linear(450→225)
      价值头: Conv1×1(256→1) + BN + ReLU + Flatten + Linear(225→256) + ReLU + Linear(256→1) + tanh
    """

    def __init__(self, board_size: int = 15, in_channels: int = 4,
                 n_blocks: int = 6, n_filters: int = 256, value_hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            Conv2d(in_channels, n_filters, 3, padding=1),
            BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_blocks)])
        # 策略头
        self.policy_conv = Conv2d(n_filters, 2, 1)
        self.policy_bn = BatchNorm2d(2)
        self.policy_fc = Linear(2 * board_size * board_size, board_size * board_size)
        # 价值头
        self.value_conv = Conv2d(n_filters, 1, 1)
        self.value_bn = BatchNorm2d(1)
        self.value_fc1 = Linear(board_size * board_size, value_hidden)
        self.value_fc2 = Linear(value_hidden, 1)
        # 价值输出头小方差初始化：初始价值接近 0，保证早期搜索多样性
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))          # (B, 225) logits
        # 价值头
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))          # (B, 1)
        return p, v

    def forward_masked(self, x: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向并返回合法点掩码后的策略分布（legal: (B,225) 0/1）。"""
        p, v = self.forward(x)
        p = p.masked_fill(legal == 0, float("-inf"))
        p = torch.softmax(p, dim=1)
        return p, v

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
