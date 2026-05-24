"""
评估指标模块

包含:
1. 准确率、AUC、AP等分类指标
2. EER (Equal Error Rate)
3. 混淆矩阵可视化
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple, Optional
import torch


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算分类评估指标

    Args:
        predictions: 预测概率 [N] 或 [N, 2]
        labels: 真实标签 [N]
        threshold: 分类阈值
    Returns:
        metrics: 包含各种指标的字典
    """
    # 确保格式正确
    if len(predictions.shape) == 2:
        probs = predictions[:, 1]  # 取fake类的概率
    else:
        probs = predictions

    # 二值化预测
    preds = (probs >= threshold).astype(int)

    metrics = {}

    # 准确率
    metrics['accuracy'] = accuracy_score(labels, preds)

    # AUC
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = 0.5  # 只有一个类别时

    # AP (Average Precision)
    try:
        metrics['ap'] = average_precision_score(labels, probs)
    except ValueError:
        metrics['ap'] = 0.0

    # EER
    metrics['eer'] = compute_eer(labels, probs)

    # 真阳性率和假阳性率
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 只有一个类别的情况
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(labels)) == 1:
            if labels[0] == 0:
                tn = len(labels)
            else:
                tp = len(labels)
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0

    return metrics


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    计算等错误率 (Equal Error Rate)

    Args:
        labels: 真实标签 [N]
        scores: 预测分数 [N]
    Returns:
        eer: 等错误率
    """
    # 检查是否有两个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # 只有一个类别，无法计算 EER
        return 0.5  # 返回随机猜测的错误率

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # 找到FPR和FNR相等的点
    diff = np.absolute(fnr - fpr)
    if np.all(np.isnan(diff)):
        return 0.5  # 无法计算时返回 0.5

    eer_idx = np.nanargmin(diff)
    eer = fpr[eer_idx]

    return eer


def compute_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = 'youden'
) -> Tuple[float, float]:
    """
    计算最优阈值

    Args:
        labels: 真实标签
        scores: 预测分数
        metric: 优化指标 ('youden', 'f1', 'accuracy')
    Returns:
        threshold: 最优阈值
        best_score: 该阈值下的最优分数
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    if metric == 'youden':
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx], j_scores[best_idx]

    elif metric == 'f1':
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx], f1_scores[best_idx]

    elif metric == 'accuracy':
        best_acc = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (scores >= thresh).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        return best_thresh, best_acc

    return 0.5, 0.0


def compute_per_method_metrics(
    predictions: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    按方法计算指标 (用于跨生成器评估)

    Args:
        predictions: {method_name: predictions}
        labels: {method_name: labels}
    Returns:
        per_method_metrics: {method_name: metrics}
    """
    per_method_metrics = {}

    for method in predictions.keys():
        if method in labels:
            per_method_metrics[method] = compute_metrics(
                predictions[method],
                labels[method]
            )

    # 计算平均值
    if per_method_metrics:
        avg_metrics = {}
        for metric_name in list(per_method_metrics.values())[0].keys():
            values = [m[metric_name] for m in per_method_metrics.values()]
            avg_metrics[metric_name] = np.mean(values)
        per_method_metrics['average'] = avg_metrics

    return per_method_metrics


class MetricsAccumulator:
    """
    累积计算指标的工具类
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.paths = []
        self.methods = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        paths: Optional[list] = None,
        methods: Optional[list] = None
    ):
        """更新累积器"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.predictions.extend(predictions)
        self.labels.extend(labels)

        if paths is not None:
            self.paths.extend(paths)
        if methods is not None:
            self.methods.extend(methods)

    def compute(self) -> Dict[str, float]:
        """计算累积的指标"""
        if len(self.predictions) == 0:
            print("[MetricsAccumulator] Warning: empty accumulator, returning default metrics")
            return {'accuracy': 0.0, 'auc': 0.5, 'ap': 0.0, 'eer': 0.5,
                    'tpr': 0.0, 'fpr': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        # 防止 NaN/Inf 泄入
        predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
        predictions = np.clip(predictions, 0.0, 1.0)
        return compute_metrics(predictions, labels)

    def compute_per_method(self) -> Dict[str, Dict[str, float]]:
        """按方法计算指标"""
        if not self.methods:
            return {'all': self.compute()}

        # 按方法分组
        method_preds = {}
        method_labels = {}

        for pred, label, method in zip(self.predictions, self.labels, self.methods):
            if method not in method_preds:
                method_preds[method] = []
                method_labels[method] = []
            method_preds[method].append(pred)
            method_labels[method].append(label)

        # 转换为numpy数组
        for method in method_preds:
            method_preds[method] = np.array(method_preds[method])
            method_labels[method] = np.array(method_labels[method])

        return compute_per_method_metrics(method_preds, method_labels)


# 测试代码
if __name__ == "__main__":
    print("Testing metrics...")

    # 模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 真实标签
    labels = np.random.randint(0, 2, n_samples)

    # 模拟预测 (带有一些噪声)
    noise = np.random.randn(n_samples) * 0.2
    scores = labels.astype(float) + noise
    scores = np.clip(scores, 0, 1)

    # 计算指标
    metrics = compute_metrics(scores, labels)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 测试最优阈值
    threshold, score = compute_optimal_threshold(labels, scores, 'youden')
    print(f"\nOptimal threshold (Youden): {threshold:.4f}, score: {score:.4f}")

    # 测试累积器
    accumulator = MetricsAccumulator()
    accumulator.update(torch.tensor(scores[:500]), torch.tensor(labels[:500]))
    accumulator.update(torch.tensor(scores[500:]), torch.tensor(labels[500:]))

    acc_metrics = accumulator.compute()
    print("\nAccumulated metrics:")
    for name, value in acc_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nMetrics test passed!")
