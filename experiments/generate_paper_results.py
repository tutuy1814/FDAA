#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# 可选的可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# LaTeX表格生成
# ============================================================================

def generate_ablation_table(results: Dict, caption: str = "消融实验结果") -> str:
    """生成消融实验LaTeX表格"""
    if not results:
        return "% No ablation results"

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(r"  \label{tab:ablation}")
    latex.append(r"  \begin{tabular}{l|c|c|c|c}")
    latex.append(r"    \toprule")
    latex.append(r"    Method & Params (M) & AUC (\%) & Acc (\%) & AP (\%) \\")
    latex.append(r"    \midrule")

    for name, metrics in results.items():
        if 'error' in metrics:
            continue

        params_m = metrics.get('params', 0) / 1e6
        auc = metrics.get('auc', 0) * 100
        acc = metrics.get('accuracy', 0) * 100
        ap = metrics.get('ap', 0) * 100

        # 高亮最佳结果
        if name.startswith('Full') or name.startswith('Ours'):
            latex.append(f"    \\textbf{{{name}}} & {params_m:.1f} & "
                        f"\\textbf{{{auc:.2f}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{ap:.2f}}} \\\\")
        else:
            latex.append(f"    {name} & {params_m:.1f} & {auc:.2f} & {acc:.2f} & {ap:.2f} \\\\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_sota_table(results: Dict, caption: str = "SOTA方法对比") -> str:
    """生成SOTA对比LaTeX表格"""
    if not results:
        return "% No SOTA results"

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(r"  \label{tab:sota}")
    latex.append(r"  \begin{tabular}{l|c|c|c|c|c}")
    latex.append(r"    \toprule")
    latex.append(r"    Method & Venue & Params (M) & AUC (\%) & Acc (\%) & AP (\%) \\")
    latex.append(r"    \midrule")

    # 方法年份信息
    venues = {
        'CNNDetection': 'CVPR 2020',
        'F3-Net': 'ECCV 2020',
        'GramNet': 'CVPR 2020',
        'Spec': 'ICIP 2019',
        'UnivFD': 'CVPR 2023',
        'NPR': 'CVPR 2024',
        'FreqNet': 'AAAI 2024',
        'DIRE': 'ICCV 2023',
        'Ours': '-',
    }

    # 按AUC排序
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1].get('auc', 0),
        reverse=True
    )

    for name, metrics in sorted_results:
        params_m = metrics.get('params', 0) / 1e6
        auc = metrics.get('auc', 0) * 100
        acc = metrics.get('accuracy', 0) * 100
        ap = metrics.get('ap', 0) * 100

        venue = venues.get(name.split()[0], '-')

        if 'Ours' in name or 'FDAA' in name:
            latex.append(f"    \\textbf{{{name}}} & \\textbf{{Ours}} & {params_m:.1f} & "
                        f"\\textbf{{{auc:.2f}}} & \\textbf{{{acc:.2f}}} & \\textbf{{{ap:.2f}}} \\\\")
        else:
            latex.append(f"    {name} & {venue} & {params_m:.1f} & {auc:.2f} & {acc:.2f} & {ap:.2f} \\\\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_multi_generator_table(results: Dict, caption: str = "多生成器验证结果") -> str:
    """生成多生成器实验LaTeX表格"""
    if not results:
        return "% No multi-generator results"

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(r"  \label{tab:multi_gen}")
    latex.append(r"  \begin{tabular}{l|c|c|c|c}")
    latex.append(r"    \toprule")
    latex.append(r"    Generator & AUC (\%) & Acc (\%) & Precision (\%) & Recall (\%) \\")
    latex.append(r"    \midrule")

    for gen_name, metrics in results.items():
        if isinstance(metrics, dict) and 'auc' in metrics:
            auc = metrics.get('auc', 0) * 100
            acc = metrics.get('accuracy', 0) * 100
            prec = metrics.get('precision', 0) * 100
            recall = metrics.get('recall', 0) * 100
            latex.append(f"    {gen_name} & {auc:.2f} & {acc:.2f} & {prec:.2f} & {recall:.2f} \\\\")

    latex.append(r"    \midrule")

    # 计算平均值
    valid_results = [m for m in results.values() if isinstance(m, dict) and 'auc' in m]
    if valid_results:
        avg_auc = np.mean([m['auc'] for m in valid_results]) * 100
        avg_acc = np.mean([m['accuracy'] for m in valid_results]) * 100
        avg_prec = np.mean([m.get('precision', 0) for m in valid_results]) * 100
        avg_recall = np.mean([m.get('recall', 0) for m in valid_results]) * 100
        latex.append(f"    \\textbf{{Average}} & \\textbf{{{avg_auc:.2f}}} & "
                    f"\\textbf{{{avg_acc:.2f}}} & \\textbf{{{avg_prec:.2f}}} & \\textbf{{{avg_recall:.2f}}} \\\\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_cross_generator_table(results: Dict, caption: str = "跨生成器泛化测试") -> str:
    """生成跨生成器泛化LaTeX表格"""
    if not results or 'per_generator' not in results:
        return "% No cross-generator results"

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(r"  \label{tab:cross_gen}")
    latex.append(r"  \begin{tabular}{l|c|c|c|c}")
    latex.append(r"    \toprule")
    latex.append(r"    Generator & Seen & AUC (\%) & Acc (\%) & AP (\%) \\")
    latex.append(r"    \midrule")

    per_gen = results['per_generator']

    for gen_name, data in per_gen.items():
        metrics = data.get('metrics', data)
        is_seen = data.get('is_seen', gen_name == results.get('train_generator', ''))

        auc = metrics.get('auc', 0) * 100
        acc = metrics.get('accuracy', 0) * 100
        ap = metrics.get('ap', 0) * 100

        seen_mark = r"\checkmark" if is_seen else "-"

        if is_seen:
            latex.append(f"    \\textit{{{gen_name}}} & {seen_mark} & {auc:.2f} & {acc:.2f} & {ap:.2f} \\\\")
        else:
            latex.append(f"    {gen_name} & {seen_mark} & {auc:.2f} & {acc:.2f} & {ap:.2f} \\\\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_efficiency_table(results: Dict, caption: str = "效率对比") -> str:
    """生成效率分析LaTeX表格"""
    if not results:
        return "% No efficiency results"

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(r"  \label{tab:efficiency}")
    latex.append(r"  \begin{tabular}{l|c|c|c|c}")
    latex.append(r"    \toprule")
    latex.append(r"    Method & Params (M) & FLOPs (G) & Latency (ms) & Throughput (img/s) \\")
    latex.append(r"    \midrule")

    for name, metrics in results.items():
        if 'error' in metrics:
            continue

        params_m = metrics.get('params', 0) / 1e6
        flops_g = metrics.get('flops', 0) / 1e9 if metrics.get('flops') else '-'
        latency = metrics.get('latency', {}).get('batch_1', {}).get('mean_ms', 0)
        throughput = metrics.get('latency', {}).get('batch_16', {}).get('throughput', 0)

        if 'Ours' in name or 'FDAA' in name:
            flops_str = f"{flops_g:.2f}" if isinstance(flops_g, float) else flops_g
            latex.append(f"    \\textbf{{{name}}} & {params_m:.1f} & {flops_str} & "
                        f"{latency:.1f} & {throughput:.1f} \\\\")
        else:
            flops_str = f"{flops_g:.2f}" if isinstance(flops_g, float) else flops_g
            latex.append(f"    {name} & {params_m:.1f} & {flops_str} & "
                        f"{latency:.1f} & {throughput:.1f} \\\\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


# ============================================================================
# 可视化图表生成
# ============================================================================

def plot_ablation_comparison(results: Dict, output_path: str):
    """绘制消融实验对比图"""
    if not HAS_MATPLOTLIB or not results:
        return

    names = []
    aucs = []
    accs = []

    for name, metrics in results.items():
        if 'error' not in metrics:
            names.append(name.replace(' ', '\n'))
            aucs.append(metrics.get('auc', 0) * 100)
            accs.append(metrics.get('accuracy', 0) * 100)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, aucs, width, label='AUC (%)', color='steelblue')
    bars2 = ax.bar(x + width/2, accs, width, label='Accuracy (%)', color='coral')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Performance (%)')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend()
    ax.set_ylim([50, 105])

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存消融实验图: {output_path}")


def plot_sota_comparison(results: Dict, output_path: str):
    """绘制SOTA方法对比图"""
    if not HAS_MATPLOTLIB or not results:
        return

    # 按AUC排序
    sorted_items = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1].get('auc', 0),
        reverse=True
    )

    names = [k for k, v in sorted_items]
    aucs = [v.get('auc', 0) * 100 for k, v in sorted_items]
    params = [v.get('params', 0) / 1e6 for k, v in sorted_items]

    if not names:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['green' if 'Ours' in n or 'FDAA' in n else 'steelblue' for n in names]
    bars = ax1.bar(names, aucs, color=colors, alpha=0.8)

    ax1.set_xlabel('Method')
    ax1.set_ylabel('AUC (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim([min(aucs) * 0.95, 102])

    # 第二Y轴显示参数量
    ax2 = ax1.twinx()
    ax2.plot(names, params, 'r--o', label='Params (M)')
    ax2.set_ylabel('Parameters (M)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('SOTA Methods Comparison')
    plt.xticks(rotation=45, ha='right')

    # 添加数值标签
    for bar, auc in zip(bars, aucs):
        ax1.annotate(f'{auc:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存SOTA对比图: {output_path}")


def plot_multi_generator_radar(results: Dict, output_path: str):
    """绘制多生成器雷达图"""
    if not HAS_MATPLOTLIB or not results:
        return

    generators = []
    metrics_data = {'AUC': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

    for gen_name, metrics in results.items():
        if isinstance(metrics, dict) and 'auc' in metrics:
            generators.append(gen_name)
            metrics_data['AUC'].append(metrics.get('auc', 0) * 100)
            metrics_data['Accuracy'].append(metrics.get('accuracy', 0) * 100)
            metrics_data['Precision'].append(metrics.get('precision', 0) * 100)
            metrics_data['Recall'].append(metrics.get('recall', 0) * 100)

    if not generators:
        return

    # 简化为条形图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (metric_name, values) in zip(axes.flatten(), metrics_data.items()):
        ax.bar(generators, values, color='steelblue')
        ax.set_title(metric_name)
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_ylim([min(values) * 0.9 if values else 0, 105])
        ax.tick_params(axis='x', rotation=45)

        for i, v in enumerate(values):
            ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=8)

    plt.suptitle('Multi-Generator Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存多生成器图: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='论文结果生成脚本')
    parser.add_argument('--input', type=str, required=True, help='输入JSON结果文件')
    parser.add_argument('--output_dir', type=str, default='./outputs/paper_results')
    parser.add_argument('--format', type=str, default='all',
                       choices=['all', 'latex', 'figures'])

    args = parser.parse_args()

    # 加载结果
    input_path = Path(args.input)
    if input_path.is_dir():
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            print("未找到JSON文件")
            return
        input_path = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]

    print(f"加载结果: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 生成LaTeX表格
    if args.format in ['all', 'latex']:
        latex_output = []
        latex_output.append("% Auto-generated LaTeX tables")
        latex_output.append(f"% Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        latex_output.append("")

        if 'ablation' in results:
            latex_output.append("% === Ablation Study ===")
            latex_output.append(generate_ablation_table(results['ablation']))
            latex_output.append("")

        if 'sota' in results or 'sota_comparison' in results:
            sota_data = results.get('sota', results.get('sota_comparison', {}))
            latex_output.append("% === SOTA Comparison ===")
            latex_output.append(generate_sota_table(sota_data))
            latex_output.append("")

        if 'multi_generator' in results:
            latex_output.append("% === Multi-Generator ===")
            latex_output.append(generate_multi_generator_table(results['multi_generator']))
            latex_output.append("")

        if 'cross_generator' in results:
            latex_output.append("% === Cross-Generator ===")
            latex_output.append(generate_cross_generator_table(results['cross_generator']))
            latex_output.append("")

        if 'efficiency' in results:
            latex_output.append("% === Efficiency ===")
            latex_output.append(generate_efficiency_table(results['efficiency']))
            latex_output.append("")

        latex_file = os.path.join(args.output_dir, 'tables.tex')
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex_output))
        print(f"保存LaTeX表格: {latex_file}")

    # 生成图表
    if args.format in ['all', 'figures'] and HAS_MATPLOTLIB:
        if 'ablation' in results:
            plot_ablation_comparison(
                results['ablation'],
                os.path.join(args.output_dir, 'ablation_comparison.png')
            )

        if 'sota' in results or 'sota_comparison' in results:
            sota_data = results.get('sota', results.get('sota_comparison', {}))
            plot_sota_comparison(
                sota_data,
                os.path.join(args.output_dir, 'sota_comparison.png')
            )

        if 'multi_generator' in results:
            plot_multi_generator_radar(
                results['multi_generator'],
                os.path.join(args.output_dir, 'multi_generator.png')
            )

    print(f"\n完成! 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
