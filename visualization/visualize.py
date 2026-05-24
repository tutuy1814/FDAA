"""
可视化模块

包含:
1. 注意力图可视化
2. 伪造热力图可视化
3. ROC曲线
4. 训练曲线
5. 结果对比表格
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_attention(
    image: np.ndarray,
    attention_map: np.ndarray,
    output_path: Optional[str] = None,
    title: str = 'Attention Map',
    alpha: float = 0.5,
    cmap: str = 'jet'
) -> plt.Figure:
    """
    可视化注意力图

    Args:
        image: 原始图像 [H, W, 3] (0-255)
        attention_map: 注意力图 [h, w] (0-1)
        output_path: 输出路径
        title: 标题
        alpha: 叠加透明度
        cmap: 颜色映射
    Returns:
        fig: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 上采样注意力图到原始尺寸
    h, w = image.shape[:2]
    attention_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize((w, h))
    ) / 255.0

    # 注意力热力图
    axes[1].imshow(attention_resized, cmap=cmap)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    # 叠加图
    axes[2].imshow(image)
    axes[2].imshow(attention_resized, cmap=cmap, alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_forgery_map(
    image: np.ndarray,
    forgery_map: np.ndarray,
    label: int,
    prediction: float,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    可视化伪造检测结果

    Args:
        image: 原始图像 [H, W, 3]
        forgery_map: 伪造注意力图 [N]
        label: 真实标签
        prediction: 预测概率
        output_path: 输出路径
    Returns:
        fig: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 原始图像
    axes[0].imshow(image)
    label_text = 'Real' if label == 0 else 'Fake'
    pred_text = f'Pred: {prediction:.2%} fake'
    axes[0].set_title(f'Ground Truth: {label_text}\n{pred_text}')
    axes[0].axis('off')

    # 将forgery_map重塑为2D
    n_patches = len(forgery_map)
    side = int(np.sqrt(n_patches))
    forgery_2d = forgery_map.reshape(side, side)

    # 上采样到图像尺寸
    h, w = image.shape[:2]
    forgery_resized = np.array(
        Image.fromarray((forgery_2d * 255).astype(np.uint8)).resize((w, h))
    ) / 255.0

    # 叠加显示
    axes[1].imshow(image)
    im = axes[1].imshow(forgery_resized, cmap='hot', alpha=0.6)
    axes[1].set_title('Forgery Attention Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Optional[str] = None,
    title: str = 'ROC Curve'
) -> plt.Figure:
    """
    绘制ROC曲线

    Args:
        labels: 真实标签
        scores: 预测分数
        output_path: 输出路径
        title: 标题
    Returns:
        fig: matplotlib Figure
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.2)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
    title: str = 'ROC Curves Comparison'
) -> plt.Figure:
    """
    绘制多个方法的ROC曲线对比

    Args:
        results: {method_name: (labels, scores)}
        output_path: 输出路径
        title: 标题
    Returns:
        fig: matplotlib Figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10.colors
    for i, (method, (labels, scores)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{method} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练曲线

    Args:
        history: 训练历史 {'train_loss': [...], 'val_loss': [...], 'val_auc': [...]}
        output_path: 输出路径
    Returns:
        fig: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss曲线
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC曲线
    if 'val_auc' in history:
        axes[1].plot(epochs, history['val_auc'], 'g-', marker='o', markersize=3)
        best_epoch = np.argmax(history['val_auc']) + 1
        best_auc = max(history['val_auc'])
        axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
        axes[1].scatter([best_epoch], [best_auc], color='r', s=100, zorder=5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title(f'Validation AUC (Best: {best_auc:.4f} @ epoch {best_epoch})')
        axes[1].grid(True, alpha=0.3)

    # Accuracy曲线
    if 'val_acc' in history:
        axes[2].plot(epochs, history['val_acc'], 'm-', marker='o', markersize=3)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Validation Accuracy')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    创建结果对比表格

    Args:
        results: {method_name: {metric_name: value}}
        metrics: 要显示的指标列表
        output_path: 输出路径 (.csv 或 .md)
    Returns:
        table_str: 表格字符串
    """
    if metrics is None:
        metrics = ['accuracy', 'auc', 'ap', 'eer']

    methods = list(results.keys())

    # 创建表头
    header = ['Method'] + [m.upper() for m in metrics]
    rows = [header]

    # 找出每个指标的最佳值
    best_values = {}
    for metric in metrics:
        values = [results[m].get(metric, 0) for m in methods]
        if metric == 'eer':  # EER越小越好
            best_values[metric] = min(values)
        else:
            best_values[metric] = max(values)

    # 填充数据
    for method in methods:
        row = [method]
        for metric in metrics:
            value = results[method].get(metric, 0)
            # 标记最佳值
            if value == best_values[metric]:
                row.append(f'**{value:.4f}**')
            else:
                row.append(f'{value:.4f}')
        rows.append(row)

    # 生成Markdown表格
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]

    lines = []
    # 表头
    lines.append('| ' + ' | '.join(str(rows[0][i]).center(col_widths[i]) for i in range(len(header))) + ' |')
    # 分隔线
    lines.append('|' + '|'.join('-' * (w + 2) for w in col_widths) + '|')
    # 数据行
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(str(row[i]).center(col_widths[i]) for i in range(len(header))) + ' |')

    table_str = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(table_str)

    return table_str


def visualize_batch_predictions(
    images: List[np.ndarray],
    labels: List[int],
    predictions: List[float],
    output_path: Optional[str] = None,
    max_images: int = 16
) -> plt.Figure:
    """
    可视化批量预测结果

    Args:
        images: 图像列表
        labels: 标签列表
        predictions: 预测概率列表
        output_path: 输出路径
        max_images: 最大显示数量
    Returns:
        fig: matplotlib Figure
    """
    n = min(len(images), max_images)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])

        label = 'Real' if labels[i] == 0 else 'Fake'
        pred = predictions[i]
        pred_label = 'Real' if pred < 0.5 else 'Fake'

        # 根据预测正确与否设置颜色
        correct = (labels[i] == 0 and pred < 0.5) or (labels[i] == 1 and pred >= 0.5)
        color = 'green' if correct else 'red'

        ax.set_title(f'GT: {label}\nPred: {pred_label} ({pred:.2%})', color=color)
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_latex_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    caption: str = 'Comparison Results',
    label: str = 'tab:comparison'
) -> str:
    """
    创建LaTeX表格

    Args:
        results: {method_name: {metric_name: value}}
        metrics: 指标列表
        caption: 表格标题
        label: LaTeX标签
    Returns:
        latex_str: LaTeX表格代码
    """
    if metrics is None:
        metrics = ['accuracy', 'auc', 'ap']

    methods = list(results.keys())

    # 找最佳值
    best_values = {}
    for metric in metrics:
        values = [results[m].get(metric, 0) for m in methods]
        best_values[metric] = max(values)

    # 生成LaTeX
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append('\\begin{tabular}{l' + 'c' * len(metrics) + '}')
    lines.append('\\toprule')

    # 表头
    header = 'Method & ' + ' & '.join(m.upper() for m in metrics) + ' \\\\'
    lines.append(header)
    lines.append('\\midrule')

    # 数据行
    for method in methods:
        row = [method]
        for metric in metrics:
            value = results[method].get(metric, 0)
            if value == best_values[metric]:
                row.append(f'\\textbf{{{value:.4f}}}')
            else:
                row.append(f'{value:.4f}')
        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# 测试代码
if __name__ == "__main__":
    print("Testing visualization...")

    import tempfile

    # 测试数据
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    attention = np.random.rand(14, 14)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试注意力可视化
        fig = visualize_attention(
            image, attention,
            output_path=os.path.join(tmpdir, 'attention.png')
        )
        plt.close(fig)

        # 测试ROC曲线
        labels = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)
        fig = plot_roc_curve(labels, scores, os.path.join(tmpdir, 'roc.png'))
        plt.close(fig)

        # 测试训练曲线
        history = {
            'train_loss': list(np.exp(-np.linspace(0, 2, 50)) + np.random.rand(50) * 0.1),
            'val_loss': list(np.exp(-np.linspace(0, 2, 50)) + np.random.rand(50) * 0.15),
            'val_auc': list(1 - np.exp(-np.linspace(0, 2, 50)) + np.random.rand(50) * 0.05),
            'val_acc': list(1 - np.exp(-np.linspace(0, 2, 50)) + np.random.rand(50) * 0.05)
        }
        fig = plot_training_curves(history, os.path.join(tmpdir, 'curves.png'))
        plt.close(fig)

        # 测试对比表格
        results = {
            'Method A': {'accuracy': 0.85, 'auc': 0.90, 'ap': 0.88},
            'Method B': {'accuracy': 0.87, 'auc': 0.92, 'ap': 0.90},
            'Ours': {'accuracy': 0.91, 'auc': 0.95, 'ap': 0.93}
        }
        table = create_comparison_table(results)
        print("\nComparison Table:")
        print(table)

    print("\nVisualization test passed!")
