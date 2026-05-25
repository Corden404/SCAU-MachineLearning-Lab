from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from task1 import ImprovedLeNet
from task2 import (
    DEFAULT_DATA_DIR,
    create_dataloaders,
    evaluate_accuracy,
    save_history_csv,
    set_seed,
    train_one_epoch,
)


def train_for_batch_size(
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, int], list[str], torch.nn.Module]:
    """使用指定 batch size 完整训练一次模型，并返回训练历史和模型。"""

    # 本函数是 batch size 实验的一组完整试验。
    # 控制变量法要求：除了 batch_size 以外，学习率、epoch 数、模型结构、
    # 数据划分、训练集扩充倍数和优化器都保持不变。
    # 每组 batch size 实验都重新固定随机种子，使模型初始化和数据划分一致。
    set_seed(args.seed)
    split_file = args.output_dir / "split_indices.json"
    # 只改变 batch_size，其余数据增强、测试比例、随机种子等保持一致。
    train_loader, test_loader, base_dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        test_size=args.test_size,
        seed=args.seed,
        split_file=split_file,
        augmentation_copies=args.augmentation_copies,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 每个 batch size 都训练一个全新的模型，保证实验组之间相互独立。
    # 如果复用上一个 batch size 训练出的权重，就会把实验变量混在一起。
    model = ImprovedLeNet(num_classes=len(base_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    # 学习率固定为 args.lr，本任务考察 batch size 的影响。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        # 在当前 batch size 下训练一个 epoch，并记录训练/测试表现。
        # 注意：epoch 数相同时，小 batch 每轮会产生更多次 optimizer.step()，
        # 因此它可能更快收敛，但这不完全等同于“batch size 本身更好”。
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)
        # batch_size 也写入历史记录，方便把多个实验结果合并分析。
        row = {
            "epoch": float(epoch),
            "batch_size": float(batch_size),
            "train_loss": train_loss,
            "train_error": 1.0 - train_accuracy,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        print(
            f"batch_size={batch_size} | epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | test_acc={test_accuracy:.4f}"
        )

    return history, base_dataset.class_to_idx, base_dataset.classes, model


def plot_batch_size_comparison(
    histories: dict[int, list[dict[str, float]]],
    output_path: Path,
) -> None:
    """绘制不同 batch size 下训练准确率的对比曲线。"""

    # 把 batch=32/64/128 画在同一张图上，便于比较收敛快慢。
    # 这里画训练准确率；测试准确率保存在各自 CSV 中，报告表格会引用。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for batch_size, history in histories.items():
        # 每条曲线代表一种 batch size，观察它随 epoch 的训练准确率变化。
        epochs = [int(row["epoch"]) for row in history]
        accuracies = [row["train_accuracy"] for row in history]
        ax.plot(epochs, accuracies, marker="o", label=f"batch={batch_size}")

    ax.set_title("Batch size vs training accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析 batch size 对比实验的命令行参数。"""

    parser = argparse.ArgumentParser(description="Compare batch sizes.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
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
    """第 5 个任务的主流程：遍历 batch size、训练模型、保存对比结果。"""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # histories 用 batch size 作为键，保存每组实验的训练历史。
    # 结构形如 {32: [...], 64: [...], 128: [...]}，最后统一绘图。
    histories: dict[int, list[dict[str, float]]] = {}

    # 同样训练 epoch 数时，小 batch 会产生更多次参数更新，这会影响实验解释。
    print(
        "Note: with the same epoch count, smaller batch sizes perform more optimizer "
        "updates than larger batch sizes."
    )
    for batch_size in args.batch_sizes:
        # 针对当前 batch size 独立训练一次模型。
        # 默认依次测试 32、64、128 三种批量大小。
        history, class_to_idx, classes, model = train_for_batch_size(batch_size, args, device)
        histories[batch_size] = history
        # 每个 batch size 单独保存训练记录和模型权重。
        # 单独保存的好处是：即使后面只想分析某一个 batch，也不用重新训练。
        save_history_csv(history, args.output_dir / f"task5_batch_{batch_size}_history.csv")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "classes": classes,
                "batch_size": batch_size,
                "config": vars(args),
            },
            args.output_dir / f"task5_batch_{batch_size}.pth",
        )

    # 所有 batch size 实验完成后，绘制统一对比图。
    plot_batch_size_comparison(histories, args.output_dir / "task5_batch_size_accuracy.png")


if __name__ == "__main__":
    main()
