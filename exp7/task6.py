from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from task2 import (
    DEFAULT_DATA_DIR,
    TransformedSubset,
    create_base_dataset,
    evaluate_accuracy,
    get_eval_transform,
    get_train_transform,
    load_or_create_split_indices,
    plot_training_curves,
    save_history_csv,
    set_seed,
    train_one_epoch,
)
from task3 import collect_predictions, classification_report, confusion_matrix, plot_confusion_matrix


class WiderLeNet(nn.Module):
    """适用于 32x32 彩色 SVHN 图像的加宽版 LeNet 模型。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 卷积部分：提取局部图像特征。相比原始 LeNet，这里增加了通道数以提升表达能力。
        # 原模型通道数是 6、16；这里改成 32、64，让网络能学习更多笔画、
        # 边缘、闭环和背景区分特征，适合处理 SVHN 中更复杂的彩色数字。
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类部分：将卷积特征映射到 10 个数字类别。
        # 因为第二层卷积输出通道变成 64，展平后的输入维度也变成 64 * 5 * 5。
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入尺寸为 3x32x32，经过两次卷积和池化后展平为一维特征。
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class ClassAwareExpandedAugmentedSubset(Dataset):
    """一次性扩充训练集，并对指定类别的增强副本使用更强增强。"""

    def __init__(
        self,
        dataset: Dataset,
        indices: Iterable[int],
        original_transform: transforms.Compose,
        normal_transform: transforms.Compose,
        boosted_transform: transforms.Compose,
        boosted_classes: set[int],
        augmentation_copies: int = 1,
    ) -> None:
        if augmentation_copies < 0:
            raise ValueError("augmentation_copies must be greater than or equal to 0.")
        self.dataset = dataset
        self.indices = list(indices)
        self.original_transform = original_transform
        self.normal_transform = normal_transform
        self.boosted_transform = boosted_transform
        self.boosted_classes = boosted_classes
        self.augmentation_copies = augmentation_copies
        self.samples_per_source = 1 + augmentation_copies
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.images: list[torch.Tensor] = []
        self.targets: list[int] = []
        # 初始化时直接生成并缓存所有 Tensor，训练阶段只从内存取数据。
        # 这样既满足“训练集扩充”的要求，又避免每个 epoch 重复做随机增强。
        self._build_cache()

    def _build_cache(self) -> None:
        """初始化时生成扩充后的 Tensor 样本，训练时直接从内存读取。"""

        for dataset_index in self.indices:
            path, target = self.dataset.samples[dataset_index]
            image = self.dataset.loader(path)
            # 每张原始训练图先保留 1 份确定性预处理版本。
            self.images.append(self.original_transform(image))
            self.targets.append(int(target))
            # 对增强副本进行类别感知处理：
            # - 如果属于 boosted_classes，就用更强增强；
            # - 否则使用 baseline 的普通增强。
            transform = self.boosted_transform if target in self.boosted_classes else self.normal_transform
            for _ in range(self.augmentation_copies):
                self.images.append(transform(image))
                self.targets.append(int(target))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        return self.images[item], self.targets[item]


def get_boosted_train_transform() -> transforms.Compose:
    """构造用于重点类别的增强策略。"""

    # 重点类别通常是样本较少或前面实验中 recall 较低的类别。
    # 这里比普通增强更强：旋转角度、平移、缩放、剪切和颜色扰动幅度都略大。
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            # 随机仿射变换模拟数字在位置、大小和倾斜角度上的变化。
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),
                scale=(0.92, 1.08),
                shear=4,
            ),
            # 颜色扰动增强模型对亮度、对比度和饱和度变化的鲁棒性。
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要训练的参数数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_train_class_counts(dataset: Dataset, train_indices: list[int], num_classes: int) -> torch.Tensor:
    """统计训练集中每个类别的样本数量。"""

    # 类别样本数用于后面计算 class weight。
    # 样本少的类别会获得更大的损失权重，降低模型偏向大类的风险。
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for index in train_indices:
        _, target = dataset.samples[index]
        counts[int(target)] += 1
    return counts


def compute_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """根据类别样本数计算交叉熵损失中的类别权重。"""

    # 权重公式 total / (num_classes * count)：
    # 样本越少，count 越小，得到的权重越大。
    # 这样少数类被错分时，对总 loss 的影响会更明显。
    total = class_counts.sum()
    num_classes = class_counts.numel()
    # 样本越少的类别权重越大，clamp_min 用于避免某类样本数为 0 时除零。
    weights = total / (num_classes * class_counts.clamp_min(1.0))
    # 归一化到均值为 1，便于保持整体损失尺度稳定。
    return weights / weights.mean()


def create_task7_dataloaders(
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    test_size: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    boosted_classes: set[int],
    augmentation_copies: int,
) -> tuple[DataLoader, DataLoader, Dataset, torch.Tensor]:
    """创建训练和测试 DataLoader，并返回训练集的类别统计。"""

    # task6/task7 的数据流程与 baseline 保持一致：
    # 同一份原始数据、同一份 split_indices.json、同一份测试集。
    # 差别只在训练集：增强副本对重点类别使用更强 transform。
    base_dataset = create_base_dataset(data_dir)
    split_file = output_dir / "split_indices.json"
    # 固定划分索引，保证不同任务或多次运行使用同一训练/测试划分。
    train_indices, test_indices = load_or_create_split_indices(
        dataset=base_dataset,
        split_file=split_file,
        test_size=test_size,
        seed=seed,
    )

    # 训练集扩充为“原图 + 增强副本”；增强副本根据类别选择不同增强策略。
    train_dataset = ClassAwareExpandedAugmentedSubset(
        dataset=base_dataset,
        indices=train_indices,
        original_transform=get_eval_transform(),
        normal_transform=get_train_transform(),
        boosted_transform=get_boosted_train_transform(),
        boosted_classes=boosted_classes,
        augmentation_copies=augmentation_copies,
    )
    test_dataset = TransformedSubset(base_dataset, test_indices, get_eval_transform())

    generator = torch.Generator()
    generator.manual_seed(seed)
    # 给 DataLoader 的随机打乱设置生成器，使训练过程更容易复现。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    class_counts = compute_train_class_counts(base_dataset, train_indices, len(base_dataset.classes))
    # 因为训练集已经按 augmentation_copies 扩充，所以类别计数也同步放大。
    # 默认每张图变成 2 份，因此每类训练样本数乘以 2；权重比例不会变，
    # 但打印到报告里时能反映扩充后的真实训练样本总量。
    class_counts = class_counts * (1 + augmentation_copies)
    return train_loader, test_loader, base_dataset, class_counts


def train_weighted_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    """使用带类别权重的交叉熵损失训练模型。"""

    # 与 baseline 的 train_model 相比，这里唯一的训练目标差异是：
    # CrossEntropyLoss 接收 class_weights，让少数类/困难类在损失中更重要。
    # class_weights 会提高少数类或重点困难类在损失函数中的影响。
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # 每轮训练后立即在测试集上评估，便于观察泛化性能变化。
        # 这里仍然复用 task2.py 的 train_one_epoch 和 evaluate_accuracy，
        # 保证训练/评估统计口径与 baseline 完全一致。
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_error": 1.0 - train_accuracy,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"test_acc={test_accuracy:.4f}"
        )
    return history


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便从终端调整训练配置。"""

    parser = argparse.ArgumentParser(description="Train the task7 improved model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=1,
        help="Number of augmented copies added for each original training sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--boost-classes", type=int, nargs="+", default=[0, 3, 6, 7, 8, 9]) #这里是要数据增强的类别
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 固定随机种子，尽量保证训练结果可复现。
    set_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # boost_classes 表示需要重点增强的类别，默认来自前面实验中召回率较低的类别。
    # 默认值 [0, 3, 6, 7, 8, 9] 可以通过命令行 --boost-classes 修改。
    boosted_classes = set(args.boost_classes)
    # 创建类别感知的扩充训练集、确定性测试集，并统计扩充后的类别样本数。
    train_loader, test_loader, base_dataset, class_counts = create_task7_dataloaders(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        boosted_classes=boosted_classes,
        augmentation_copies=args.augmentation_copies,
    )
    class_weights = compute_class_weights(class_counts)
    # 使用更宽的 LeNet，而不是 task1.py 的轻量 baseline。
    model = WiderLeNet(num_classes=len(base_dataset.classes)).to(device)

    # 打印本次训练的关键配置，便于实验记录和结果复查。
    print(f"Classes: {base_dataset.classes}")
    print(f"Original train samples: {len(train_loader.dataset) // (1 + args.augmentation_copies)}")
    print(f"Augmented copies per sample: {args.augmentation_copies}")
    print(f"Expanded train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Boosted classes: {sorted(boosted_classes)}")
    print(f"Train class counts: {[int(value) for value in class_counts.tolist()]}")
    print(f"Class weights: {[round(float(value), 4) for value in class_weights.tolist()]}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print(f"Device: {device}")

    history = train_weighted_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=class_weights,
        device=device,
    )

    # 在测试集上收集预测结果，用于计算混淆矩阵、分类报告和最终准确率。
    # 这里使用和 task3.py 完全相同的评估函数，便于直接比较 baseline 与改进模型。
    y_true, y_pred = collect_predictions(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred, num_classes=len(base_dataset.classes))
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(matrix, class_names=base_dataset.classes)

    # 保存训练曲线、混淆矩阵、分类报告和模型权重，方便后续写实验报告或复现实验。
    # 虽然脚本名是 task6.py，输出文件名沿用 task7_*，对应报告里的“改进实验”。
    save_history_csv(history, args.output_dir / "task7_history.csv")
    plot_training_curves(history, args.output_dir / "task7_training_curves.png")
    np.savetxt(args.output_dir / "task7_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (args.output_dir / "task7_classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, base_dataset.classes, args.output_dir / "task7_confusion_matrix.png")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": base_dataset.class_to_idx,
            "classes": base_dataset.classes,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "boosted_classes": sorted(boosted_classes),
            "config": vars(args),
        },
        args.output_dir / "task7_wider_weighted_lenet.pth",
    )
    metrics = {
        # JSON 汇总核心指标，后面写 task.md 或做自动分析时不用再解析文本报告。
        "accuracy": accuracy,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "trainable_parameters": count_trainable_parameters(model),
        "boosted_classes": sorted(boosted_classes),
        "class_counts": [int(value) for value in class_counts.tolist()],
        "class_weights": [float(value) for value in class_weights.tolist()],
    }
    (args.output_dir / "task7_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Final test accuracy: {accuracy:.4f}")
    print(report)


if __name__ == "__main__":
    main()
