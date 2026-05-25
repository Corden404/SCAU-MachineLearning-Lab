from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from task1 import ImprovedLeNet


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data_svhn_8k" / "svhn_8k"


def set_seed(seed: int) -> None:
    """固定随机种子，使数据划分、数据打乱和模型训练尽量可复现。"""

    # Python 标准库、NumPy 和 PyTorch 都各自维护随机数生成器，需要分别设置。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关闭 cuDNN 自动寻找最快算法，改用确定性实现，减少每次运行的波动。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_transform() -> transforms.Compose:
    """训练集增强副本使用的图像预处理和轻量数据增强。"""

    # 这个 transform 只用于“额外生成出来的增强副本”，不用于测试集。
    # 也就是说，老师要求的训练集扩充发生在训练集内部，测试集仍保持原始评估难度。
    return transforms.Compose(
        [
            # 统一输入尺寸，保证能送入 LeNet 的固定全连接层。
            transforms.Resize((32, 32)),
            # 随机仿射变换模拟数字位置和角度的变化，提高泛化能力。
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            # 随机调整亮度和对比度，缓解拍摄光照差异。
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # 转成 [0, 1] 范围的 Tensor，形状从 HWC 变成 CHW。
            transforms.ToTensor(),
            # 归一化到大致 [-1, 1]，让优化过程更稳定。
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    """测试/验证阶段的图像预处理，不加入随机增强以保证评估稳定。"""

    # 这个 transform 是确定性的：同一张图片每次得到完全相同的 Tensor。
    # 原始训练副本和测试集都使用它，便于和随机增强副本区分。
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class FilenameLabelImageDataset(Dataset):
    """Read a flat image folder whose label is the filename prefix.

    Example:
        svhn_8k/0_0052.jpg -> class "0"
        svhn_8k/9_1234.jpg -> class "9"
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        # 扁平目录格式下，所有图片直接位于 root 目录，标签从文件名前缀解析。
        image_paths = [
            path
            for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        if not image_paths:
            raise RuntimeError(f"No image files found under: {root}")

        # 收集出现过的类别名并按数字大小排序，确保类别顺序稳定为 0,1,...,9。
        class_names = sorted({self._label_from_path(path) for path in image_paths}, key=int)
        self.classes = class_names
        self.class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
        # samples 保存 (图片路径, 类别编号)，与 torchvision.datasets.ImageFolder 的接口保持一致。
        self.samples = [
            (str(path), self.class_to_idx[self._label_from_path(path)])
            for path in image_paths
        ]
        self.targets = [target for _, target in self.samples]
        # 使用 torchvision 默认图片加载器，能自动处理常见图片格式。
        self.loader = default_loader

    @staticmethod
    def _label_from_path(path: Path) -> str:
        # 文件名形如 0_0052.jpg，取第一个下划线之前的部分作为标签。
        label = path.stem.split("_", maxsplit=1)[0]
        if not label.isdigit():
            raise ValueError(f"Cannot parse numeric class label from filename: {path.name}")
        return label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # 这里只返回 PIL Image 和标签；真正的 transform 在 TransformedSubset 中应用。
        path, target = self.samples[index]
        return self.loader(path), target


def _has_direct_images(path: Path) -> bool:
    """判断目录下是否直接包含图片文件。"""

    return any(
        child.is_file() and child.suffix.lower() in IMG_EXTENSIONS
        for child in path.iterdir()
    )


def _resolve_data_dir(data_dir: Path) -> Path:
    """兼容不同数据集目录结构，找到真正存放图片的目录。"""

    # 情况 1：用户传入的目录本身就直接包含图片。
    if _has_direct_images(data_dir):
        return data_dir

    # 情况 2：默认从当前实验目录启动，自动查找常见的数据集子目录。
    preferred_children = [
        data_dir / "svhn_8k",
        data_dir / "data_svhn_8k" / "svhn_8k",
        data_dir / "data" / "svhn_subset",
    ]
    for preferred_child in preferred_children:
        if preferred_child.exists() and preferred_child.is_dir() and _has_direct_images(preferred_child):
            return preferred_child

    # 情况 3：只有一个子目录直接包含图片，则自动进入这个子目录。
    image_children = [
        child
        for child in data_dir.iterdir()
        if child.is_dir() and _has_direct_images(child)
    ]
    if len(image_children) == 1:
        return image_children[0]

    return data_dir


def create_base_dataset(data_dir: Path) -> Dataset:
    """根据数据目录格式创建基础数据集对象。"""

    resolved_dir = _resolve_data_dir(data_dir)
    # 扁平图片目录使用自定义 Dataset；按类别分文件夹的目录使用 ImageFolder。
    if _has_direct_images(resolved_dir):
        return FilenameLabelImageDataset(resolved_dir)
    return datasets.ImageFolder(root=str(resolved_dir), transform=None)


class TransformedSubset(Dataset):
    """A subset wrapper that applies a transform at read time."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Iterable[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        # 保存基础数据集和要使用的样本下标，从而复用同一份训练/测试划分。
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        # 保留类别映射信息，便于后续保存模型和评估。
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        # item 是子集内下标，需要先映射回原始数据集下标。
        dataset_index = self.indices[item]
        path, target = self.dataset.samples[dataset_index]
        image = self.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ExpandedAugmentedSubset(Dataset):
    """把训练集扩充为原始样本加若干增强副本。

    数据集初始化时会一次性生成并缓存所有原图 Tensor 和增强 Tensor。这样训练集
    长度仍然会变为 len(indices) * (1 + augmentation_copies)，但每个 epoch
    不需要反复从磁盘读图、解码和执行随机增强，CPU 训练速度会稳定很多。
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Iterable[int],
        original_transform: transforms.Compose,
        augmentation_transform: transforms.Compose,
        augmentation_copies: int = 1,
    ) -> None:
        if augmentation_copies < 0:
            raise ValueError("augmentation_copies must be greater than or equal to 0.")
        self.dataset = dataset
        self.indices = list(indices)
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
        self.augmentation_copies = augmentation_copies
        self.samples_per_source = 1 + augmentation_copies
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.images: list[torch.Tensor] = []
        self.targets: list[int] = []
        self._build_cache()

    def _build_cache(self) -> None:
        """一次性生成扩充后的训练样本，避免每个 epoch 重复增强。"""

        for dataset_index in self.indices:
            # 先读取一张原始图片。dataset_index 是原始全集中的下标，
            # 不是训练子集内部的下标，因此可以和 split_indices.json 对应起来。
            path, target = self.dataset.samples[dataset_index]
            image = self.dataset.loader(path)
            # 第 1 份样本保留为“原始副本”：只做 Resize/ToTensor/Normalize。
            # 这样训练集中仍然包含未随机扭曲的原图信息。
            self.images.append(self.original_transform(image))
            self.targets.append(int(target))
            # 后续 augmentation_copies 份样本是增强副本。
            # 当前默认 augmentation_copies=1，所以每张训练图额外生成 1 张增强图，
            # 训练集长度从 6400 变成 6400 * (1 + 1) = 12800。
            for _ in range(self.augmentation_copies):
                self.images.append(self.augmentation_transform(image))
                self.targets.append(int(target))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        return self.images[item], self.targets[item]


def stratified_split_indices(
    targets: list[int],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """按类别分层生成训练集和测试集下标。"""

    # 分层划分的目的：每个数字类别都按接近 8:2 的比例进入训练集和测试集。
    # 如果直接随机划分，少数类如 8、9 可能在测试集中比例波动较大，影响指标解释。
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    # 先把所有样本下标按类别分组，保证每个类别都按相近比例划入测试集。
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    # 测试集总量尽量接近 len(targets) * test_size。
    target_test_count = int(round(len(targets) * test_size))
    allocations: list[dict[str, float | int | list[int]]] = []

    for _, class_indices in sorted(by_class.items()):
        # 每个类别内部先随机打乱，再切分前 test_count 个样本作为测试集。
        rng.shuffle(class_indices)
        if len(class_indices) <= 1:
            # 类别样本过少时不放入测试集，避免训练集中完全没有该类别。
            allocations.append(
                {
                    "indices": class_indices,
                    "test_count": 0,
                    "fraction": 0.0,
                    "count": len(class_indices),
                }
            )
            continue

        raw_test_count = len(class_indices) * test_size
        # 先向下取整，再限制在 [1, 类别样本数 - 1]，保证训练/测试都至少保留样本。
        test_count = math.floor(raw_test_count)
        test_count = max(1, min(test_count, len(class_indices) - 1))
        allocations.append(
            {
                "indices": class_indices,
                "test_count": test_count,
                "fraction": raw_test_count - math.floor(raw_test_count),
                "count": len(class_indices),
            }
        )

    current_test_count = sum(int(allocation["test_count"]) for allocation in allocations)
    # 如果各类别向下取整后测试样本不足，优先给小数部分最大的类别补一个测试样本。
    while current_test_count < target_test_count:
        candidates = [
            allocation
            for allocation in allocations
            if int(allocation["test_count"]) < int(allocation["count"]) - 1
        ]
        if not candidates:
            break
        allocation = max(candidates, key=lambda item: float(item["fraction"]))
        allocation["test_count"] = int(allocation["test_count"]) + 1
        allocation["fraction"] = float(allocation["fraction"]) - 1.0
        current_test_count += 1

    # 如果测试样本过多，则从仍能保证测试集中有样本的类别里减掉。
    while current_test_count > target_test_count:
        candidates = [
            allocation
            for allocation in allocations
            if int(allocation["test_count"]) > 1
        ]
        if not candidates:
            break
        allocation = min(candidates, key=lambda item: float(item["fraction"]))
        allocation["test_count"] = int(allocation["test_count"]) - 1
        allocation["fraction"] = float(allocation["fraction"]) + 1.0
        current_test_count -= 1

    for allocation in allocations:
        class_indices = allocation["indices"]
        test_count = int(allocation["test_count"])
        # class_indices 已被打乱，直接按数量切分即可。
        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])

    # 最后整体打乱，避免 DataLoader 读到按类别排列的样本。
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def load_or_create_split_indices(
    dataset: Dataset,
    split_file: Path,
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """读取已保存的划分；若不存在或不匹配，则重新创建并保存。"""

    # 所有任务都复用同一个 split_indices.json。
    # 这样 task2/task3/task4/task5/task6 的训练集和测试集完全一致，
    # 后面比较学习率、batch size、改进模型时才是公平的。
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        # 数据集大小变化时，旧划分下标已经不能可靠对应当前样本。
        if payload.get("num_samples") != len(dataset):
            raise ValueError(
                f"Split file {split_file} has {payload.get('num_samples')} samples, "
                f"but the current dataset has {len(dataset)} samples."
            )
        # 类别映射不一致会导致标签含义错位，必须阻止继续训练。
        if payload.get("class_to_idx") != dataset.class_to_idx:
            raise ValueError(f"Class mapping in {split_file} does not match the dataset.")
        expected_test_count = int(round(len(dataset) * test_size))
        if (
            len(payload.get("test_indices", [])) == expected_test_count
            and len(payload.get("train_indices", [])) + len(payload.get("test_indices", [])) == len(dataset)
        ):
            return payload["train_indices"], payload["test_indices"]
        print(f"Existing split in {split_file} does not match the requested size; regenerating it.")

    # 首次运行或划分参数变化时，重新生成分层划分。
    train_indices, test_indices = stratified_split_indices(
        targets=list(dataset.targets),
        test_size=test_size,
        seed=seed,
    )
    split_file.parent.mkdir(parents=True, exist_ok=True)
    # 保存划分结果，确保 task3/task4/task5 使用完全相同的训练/测试样本。
    with split_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "num_samples": len(dataset),
                "test_size": test_size,
                "seed": seed,
                "class_to_idx": dataset.class_to_idx,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            file,
            indent=2,
        )
    return train_indices, test_indices


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    test_size: float,
    seed: int,
    split_file: Path,
    augmentation_copies: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, Dataset]:
    """创建训练和测试 DataLoader，并返回基础数据集以便查询类别信息。"""

    # 这个函数是 task2/task4/task5 共用的数据入口：
    # 1. 读取图片路径和标签；
    # 2. 复用或生成训练/测试划分；
    # 3. 只扩充训练集，不扩充测试集；
    # 4. 包装成 PyTorch DataLoader 供训练循环使用。
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    base_dataset = create_base_dataset(data_dir)
    if len(base_dataset) == 0:
        raise RuntimeError(f"No images found under: {data_dir}")

    train_indices, test_indices = load_or_create_split_indices(
        dataset=base_dataset,
        split_file=split_file,
        test_size=test_size,
        seed=seed,
    )

    # 训练集扩充为“原图 + 增强副本”，测试集只使用确定性预处理。
    train_dataset = ExpandedAugmentedSubset(
        dataset=base_dataset,
        indices=train_indices,
        original_transform=get_eval_transform(),
        augmentation_transform=get_train_transform(),
        augmentation_copies=augmentation_copies,
    )
    test_dataset = TransformedSubset(base_dataset, test_indices, get_eval_transform())
    # 单独设置 DataLoader 的随机生成器，保证 shuffle 顺序可复现。
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, base_dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch，返回平均损失和训练准确率。"""

    # 一个 epoch 表示完整遍历一遍“扩充后的训练集”。
    # 当前默认情况下，原始 6400 张训练图已经扩充成 12800 个 Tensor 样本。
    # 切换到训练模式，启用训练阶段行为。
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        # 将数据移动到 CPU/GPU 设备上，与模型所在设备保持一致。
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮梯度，set_to_none=True 通常更省显存。
        optimizer.zero_grad(set_to_none=True)
        # 前向传播得到 logits。
        logits = model(images)
        # CrossEntropyLoss 内部包含 log_softmax，不需要在模型末尾手动加 softmax。
        loss = criterion(logits, labels)
        # 反向传播计算梯度，并由优化器更新参数。
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        # loss.item() 是 batch 平均损失，乘以 batch_size 后累加，最后再除以总样本数。
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """在不计算梯度的情况下评估准确率。"""

    # eval 模式关闭 Dropout/BatchNorm 的训练行为；本模型没有这些层，但保留是好习惯。
    model.eval()
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_correct / total_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """完整训练模型，并记录每个 epoch 的损失和准确率。"""

    # task2 的基准实验固定三件事：
    # - 模型：task1.py 中的 ImprovedLeNet；
    # - 优化器：Adam；
    # - 损失函数：普通 CrossEntropyLoss。
    # 后续 task4/task5 会在此基础上只改变一个自变量。
    # 多分类任务使用交叉熵损失；Adam 对学习率通常不太敏感，适合作为基准优化器。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        # 训练一个 epoch 后，立即在测试集上评估泛化表现。
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_accuracy = evaluate_accuracy(model, test_loader, device)
        # 保存成字典，后面可以直接写入 CSV 或绘图。
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


def save_history_csv(history: list[dict[str, float]], output_path: Path) -> None:
    """将训练历史保存为 CSV，便于实验报告引用或进一步分析。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    """绘制训练损失曲线和训练/测试准确率曲线。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in history], marker="s", label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便从终端修改数据路径、训练轮数等配置。"""

    parser = argparse.ArgumentParser(description="Train the baseline improved LeNet model.")
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
    return parser.parse_args()


def main() -> None:
    """第 2 个任务的主流程：准备数据、训练模型、保存结果。"""

    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 选择训练设备。当前机器没有 CUDA 时会自动使用 CPU，代码无需手动修改。
    # 如果机器有 CUDA 就使用 GPU，否则退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_file = args.output_dir / "split_indices.json"
    # create_dataloaders 内部会完成训练集扩充：
    # 原始训练样本数 = 6400，默认 augmentation_copies=1，
    # 因此 DataLoader 实际看到的训练样本数 = 12800。
    train_loader, test_loader, base_dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        split_file=split_file,
        augmentation_copies=args.augmentation_copies,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Classes: {base_dataset.classes}")
    print(f"Original train samples: {len(train_loader.dataset) // (1 + args.augmentation_copies)}")
    print(f"Augmented copies per sample: {args.augmentation_copies}")
    print(f"Expanded train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Device: {device}")

    model = ImprovedLeNet(num_classes=len(base_dataset.classes)).to(device)
    # 开始训练并收集每个 epoch 的指标。
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # 保存训练记录、曲线图和模型 checkpoint，供后续任务直接复用。
    # task3.py 会读取 improved_lenet_baseline.pth 来做最终测试集评估。
    # task.md 中的训练曲线和表格也来自这些输出文件。
    save_history_csv(history, args.output_dir / "task2_history.csv")
    plot_training_curves(history, args.output_dir / "task2_training_curves.png")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": base_dataset.class_to_idx,
            "classes": base_dataset.classes,
            "config": vars(args),
        },
        args.output_dir / "improved_lenet_baseline.pth",
    )


if __name__ == "__main__":
    main()
