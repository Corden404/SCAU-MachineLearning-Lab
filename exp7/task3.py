from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from task1 import ImprovedLeNet
from task2 import (
    DEFAULT_DATA_DIR,
    TransformedSubset,
    create_base_dataset,
    get_eval_transform,
    load_or_create_split_indices,
    set_seed,
)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """加载模型 checkpoint，并兼容不同版本 PyTorch 的 torch.load 参数。"""

    try:
        # 新版本 PyTorch 支持 weights_only 参数；这里设为 False 以读取完整字典。
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版本 PyTorch 没有 weights_only 参数，回退到普通加载方式。
        return torch.load(path, map_location=device)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """遍历测试集，收集真实标签和预测标签。"""

    # task3 不再训练模型，只做前向推理。
    # y_true 用来保存测试集真实标签，y_pred 用来保存模型预测标签，
    # 后续准确率、混淆矩阵、分类报告都由这两个数组计算出来。
    # 评估阶段关闭训练行为；@torch.no_grad() 避免构建计算图，节省显存和时间。
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        # 数据和模型必须位于同一设备上。
        images = images.to(device)
        labels = labels.to(device)
        # 模型输出每个类别的 logits，取最大值所在类别作为预测结果。
        logits = model(images)
        predictions = logits.argmax(dim=1)
        # 移回 CPU 并转为 Python list，最后统一组成 NumPy 数组。
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """手动计算混淆矩阵，行表示真实类别，列表示预测类别。"""

    # 混淆矩阵是分析错分来源的核心工具。
    # 例如 matrix[8, 6] 很大，就说明真实数字 8 经常被模型预测成 6。
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        # matrix[i, j] 表示真实为 i 但被预测为 j 的样本数量。
        matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_report(matrix: np.ndarray, class_names: list[str]) -> str:
    """根据混淆矩阵生成 precision、recall、f1-score 等分类指标。"""

    # precision/recall/f1 比单一 accuracy 更细：
    # - precision 看“预测成某类的样本里有多少是真的”；
    # - recall 看“真实某类的样本里有多少被找出来”；
    # - f1 是 precision 和 recall 的折中。
    lines = [
        "class precision recall f1-score support",
        "----- --------- ------ -------- -------",
    ]
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []

    for index, class_name in enumerate(class_names):
        # tp: 当前类预测正确；fp: 其他类被错预测成当前类；fn: 当前类被错预测成其他类。
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        # support 表示当前真实类别的样本数。
        support = int(matrix[index, :].sum())

        # 分母为 0 时用 0.0 避免除零错误。
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        lines.append(f"{class_name:>5} {precision:>9.4f} {recall:>6.4f} {f1:>8.4f} {support:>7}")

    total_support = sum(supports)
    # macro 平均：各类别同等权重；weighted 平均：按各类别样本数加权。
    weights = np.array(supports, dtype=np.float64) / total_support
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    weighted_precision = float(np.dot(weights, np.array(precisions)))
    weighted_recall = float(np.dot(weights, np.array(recalls)))
    weighted_f1 = float(np.dot(weights, np.array(f1_scores)))

    lines.append("----- --------- ------ -------- -------")
    lines.append(f"macro {macro_precision:>9.4f} {macro_recall:>6.4f} {macro_f1:>8.4f} {total_support:>7}")
    lines.append(
        f"weighted {weighted_precision:>6.4f} {weighted_recall:>6.4f} "
        f"{weighted_f1:>8.4f} {total_support:>7}"
    )
    return "\n".join(lines)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """将混淆矩阵绘制成热力图并保存为图片。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    # 使用蓝色颜色映射，数值越大颜色越深。
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")

    # 在每个格子中写入数量。背景较深时使用白字，提高可读性。
    threshold = matrix.max() * 0.55 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析评估脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Evaluate the baseline improved LeNet model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/improved_lenet_baseline.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """第 3 个任务的主流程：加载数据划分、模型权重并输出评估结果。"""

    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # task3 必须复用 task2 的测试集索引，不能重新随机划分。
    # 否则 task2 的训练曲线和 task3 的混淆矩阵就不是同一份测试集结果。
    # 创建基础数据集，并复用 task2 保存的 split_indices.json，保证测试集完全一致。
    base_dataset = create_base_dataset(args.data_dir)
    split_file = args.output_dir / "split_indices.json"
    _, test_indices = load_or_create_split_indices(
        dataset=base_dataset,
        split_file=split_file,
        test_size=args.test_size,
        seed=args.seed,
    )
    test_dataset = TransformedSubset(base_dataset, test_indices, get_eval_transform())
    # 注意：测试集只做确定性预处理，不做随机旋转/平移/颜色扰动。
    # 这保证每次运行 task3 得到的是同一套评估输入。
    # 测试集不需要随机打乱，方便结果复现和排查。
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 加载训练好的参数并恢复到同样类别数的 ImprovedLeNet 模型中。
    # checkpoint 里保存的是 state_dict，不重新训练；这里恢复的是 task2 的最终模型。
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = ImprovedLeNet(num_classes=len(base_dataset.classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 收集预测结果并计算整体准确率、混淆矩阵和分类报告。
    # accuracy 反映整体表现，classification_report 能进一步看每个数字的表现。
    y_true, y_pred = collect_predictions(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred, num_classes=len(base_dataset.classes))
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(matrix, class_names=base_dataset.classes)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(report)
    # 将数值结果和图片都保存下来，便于实验报告插图和附录引用。
    # CSV 适合后续程序读取，TXT 适合直接粘到报告，PNG 用于可视化展示。
    np.savetxt(args.output_dir / "task3_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (args.output_dir / "task3_classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, base_dataset.classes, args.output_dir / "task3_confusion_matrix.png")


if __name__ == "__main__":
    main()
