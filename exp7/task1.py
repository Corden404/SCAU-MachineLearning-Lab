from __future__ import annotations

import torch
from torch import nn


class ImprovedLeNet(nn.Module):
    """用于 10 类彩色数字分类的 LeNet 风格卷积神经网络。"""

    def __init__(self, num_classes: int = 10) -> None:
        # 调用父类初始化方法，注册后续定义的网络层参数。
        super().__init__()

        # 特征提取部分：两组「卷积 + ReLU + 最大池化」。
        # 输入尺寸为 3 x 32 x 32。
        self.features = nn.Sequential(
            # 第 1 个卷积层：3 个输入通道对应 RGB，输出 6 个特征图。
            # kernel_size=5 且不使用 padding，所以空间尺寸从 32x32 变为 28x28。
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            # ReLU 引入非线性，inplace=True 可节省部分显存。
            nn.ReLU(inplace=True),
            # 2x2 最大池化将宽高各缩小一半：28x28 -> 14x14。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第 2 个卷积层：通道数从 6 增加到 16，尺寸 14x14 -> 10x10。
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            # 再次池化：10x10 -> 5x5，最终特征尺寸为 16 x 5 x 5。
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分类器部分：将卷积特征拉平成一维向量后，通过全连接层输出类别分数。
        self.classifier = nn.Sequential(
            # 16 个通道，每个通道 5x5，因此输入维度为 16 * 5 * 5。
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            # 最后一层输出 num_classes 个 logits，交叉熵损失会直接使用这些未归一化分数。
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义一次前向传播：图片 -> 卷积特征 -> 展平 -> 类别 logits。"""

        # 先经过卷积和池化层提取局部视觉特征。
        x = self.features(x)
        # 从第 1 维开始展平，保留 batch 维度不变。
        x = torch.flatten(x, start_dim=1)
        # 送入全连接分类器，得到每个类别的预测分数。
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要梯度更新的参数总量。"""

    # parameter.numel() 返回该参数张量中元素的个数。
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    """构造模型并用随机输入检查网络结构和张量尺寸是否正确。"""

    # 创建一个 10 分类模型，对应数字 0-9。
    model = ImprovedLeNet(num_classes=10)
    # 构造一个 batch size 为 1 的随机 RGB 图片，用来测试前向传播。
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_output = model(dummy_input)

    # 打印模型结构、输入输出尺寸和各层理论特征尺寸，便于写实验报告。
    print(model)
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(dummy_output.shape)}")
    print("Expected feature sizes:")
    print("  C1: 6 x 28 x 28")
    print("  S2: 6 x 14 x 14")
    print("  C3: 16 x 10 x 10")
    print("  S4: 16 x 5 x 5")
    print("  C5 input flatten size: 16 * 5 * 5")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print("Loss function for later tasks: torch.nn.CrossEntropyLoss")


if __name__ == "__main__":
    main()
