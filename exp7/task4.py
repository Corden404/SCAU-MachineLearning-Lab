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


def float_to_name(value: float) -> str:
    """把浮点学习率转换成适合放进文件名的字符串。"""

    # 例如 0.001 -> "0_001"，避免文件名中出现多个小数点造成阅读混乱。
    return f"{value:g}".replace(".", "_")


def train_for_learning_rate(
    lr: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, int], list[str], torch.nn.Module]:
    """使用指定学习率完整训练一次模型，并返回训练历史和模型。"""

    # 本函数是学习率实验的一组完整试验。
    # 控制变量法要求：除了 lr 以外，数据划分、训练集扩充倍数、batch size、
    # epoch 数、模型结构、优化器类型都保持一致。
    # 每个学习率实验都重新设置随机种子，保证初始权重和数据打乱尽量一致。
    set_seed(args.seed)
    split_file = args.output_dir / "split_indices.json"
    # 复用 task2 的数据加载函数，保证训练/测试划分和预处理方式一致。
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

    # 每个学习率都训练一个全新的模型，避免前一次实验的权重影响后一次实验。
    # 如果复用模型继续训练，就无法判断性能差异到底来自哪个学习率。
    model = ImprovedLeNet(num_classes=len(base_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    # 本任务的自变量是 lr，因此只改变优化器学习率，其余配置保持不变。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch 后在测试集上评估，便于观察不同学习率的收敛速度。
        # 这里关注的是“同样训练预算下谁下降更快、准确率更高”。
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)
        # 保存当前学习率，后续合并或画图时可以直接区分不同实验组。
        row = {
            "epoch": float(epoch),
            "learning_rate": lr,
            "train_loss": train_loss,
            "train_error": 1.0 - train_accuracy,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        print(
            f"lr={lr:g} | epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | test_acc={test_accuracy:.4f}"
        )

    return history, base_dataset.class_to_idx, base_dataset.classes, model


def plot_learning_rate_comparison(
    histories: dict[float, list[dict[str, float]]],
    output_path: Path,
) -> None:
    """绘制不同学习率下训练准确率的对比曲线。"""

    # 折线图用于直观看出不同学习率的收敛速度差异：
    # 曲线升得快通常表示该学习率在当前训练预算下更有效。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for lr, history in histories.items():
        # 横轴为 epoch，纵轴为训练准确率；每条线对应一个学习率。
        epochs = [int(row["epoch"]) for row in history]
        accuracies = [row["train_accuracy"] for row in history]
        ax.plot(epochs, accuracies, marker="o", label=f"lr={lr:g}")

    ax.set_title("Learning rate vs training accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析学习率对比实验的命令行参数。"""

    parser = argparse.ArgumentParser(description="Compare learning rates.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.0001, 0.001])
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
    """第 4 个任务的主流程：遍历学习率、训练模型、保存对比结果。"""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # histories 用学习率作为键，保存每组实验的 epoch 指标。
    # 结构形如 {0.0001: [...每轮指标...], 0.001: [...每轮指标...]}，
    # 方便最后统一画到同一张图中。
    histories: dict[float, list[dict[str, float]]] = {}

    for lr in args.learning_rates:
        # 针对当前学习率独立训练一次模型。
        # 默认会跑两个值：0.0001 和 0.001。
        history, class_to_idx, classes, model = train_for_learning_rate(lr, args, device)
        histories[lr] = history
        name = float_to_name(lr)
        # 每个学习率单独保存 CSV 和模型权重，便于后续查表或复现实验。
        # 文件名中把小数点替换成下划线，例如 task4_lr_0_001_history.csv。
        save_history_csv(history, args.output_dir / f"task4_lr_{name}_history.csv")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "classes": classes,
                "learning_rate": lr,
                "config": vars(args),
            },
            args.output_dir / f"task4_lr_{name}.pth",
        )

    # 所有学习率训练完成后，绘制一张总对比图。
    plot_learning_rate_comparison(histories, args.output_dir / "task4_learning_rate_accuracy.png")


if __name__ == "__main__":
    main()
