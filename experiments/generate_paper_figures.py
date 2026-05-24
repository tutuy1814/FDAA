#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ========== SCI标准全局样式 ==========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.grid': False,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
FIG_DIR = os.path.join(OUTPUT_DIR, 'paper_figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path)
    print(f'  [Saved] {name}.pdf/.png')
    plt.close(fig)


# =====================================================================
# Fig 1: 鲁棒性衰减折线图 — 3种扰动类型 (最核心图)
# =====================================================================
def fig1_robustness_degradation_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)

    methods = {
        'FDAA-Net (Ours)': {'c': '#C0392B', 'ls': '-',  'm': 'o', 'z': 10, 'lw': 1.6},
        'F3Net':            {'c': '#2980B9', 'ls': '--', 'm': 's', 'z': 5, 'lw': 0.9},
        'CNNDetection':     {'c': '#27AE60', 'ls': '--', 'm': '^', 'z': 5, 'lw': 0.9},
        'FreqNet':          {'c': '#8E44AD', 'ls': '--', 'm': 'D', 'z': 5, 'lw': 0.9},
        'UnivFD':           {'c': '#E67E22', 'ls': '--', 'm': 'v', 'z': 5, 'lw': 0.9},
    }
    clean = {'FDAA-Net (Ours)': 99.93, 'F3Net': 99.75, 'CNNDetection': 99.40,
             'FreqNet': 99.07, 'UnivFD': 99.13}

    # (a) JPEG
    jpeg = {
        'FDAA-Net (Ours)': [99.01, 98.24, 96.33],
        'F3Net':            [97.04, 95.94, 93.66],
        'CNNDetection':     [96.54, 95.65, 93.96],
        'FreqNet':          [96.61, 95.59, 93.40],
        'UnivFD':           [95.18, 93.79, 90.98],
    }
    ax = axes[0]
    ax.set_title('(a) JPEG Compression', fontsize=9, pad=4)
    for name, cfg in methods.items():
        ax.plot(['Clean', 'Q=70', 'Q=50', 'Q=30'], [clean[name]] + jpeg[name],
                color=cfg['c'], linestyle=cfg['ls'], marker=cfg['m'],
                markersize=4, markeredgewidth=0.4, markeredgecolor='white',
                label=name, zorder=cfg['z'], linewidth=cfg['lw'])
    ax.set_ylabel('AUC (%)')
    ax.set_ylim(88, 101)

    # (b) Blur
    blur = {
        'FDAA-Net (Ours)': [99.39, 94.53],
        'F3Net':            [98.96, 89.39],
        'CNNDetection':     [98.42, 85.66],
        'FreqNet':          [97.28, 84.92],
        'UnivFD':           [97.04, 88.08],
    }
    ax = axes[1]
    ax.set_title('(b) Gaussian Blur', fontsize=9, pad=4)
    for name, cfg in methods.items():
        ax.plot(['Clean', r'$\sigma$=1.0', r'$\sigma$=2.0'], [clean[name]] + blur[name],
                color=cfg['c'], linestyle=cfg['ls'], marker=cfg['m'],
                markersize=4, markeredgewidth=0.4, markeredgecolor='white',
                zorder=cfg['z'], linewidth=cfg['lw'])
    ax.annotate('', xy=(2, 94.53), xytext=(2, 89.39),
                arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=0.8))
    ax.text(2.15, 91.6, r'$\Delta$5.1', fontsize=7, color='#C0392B', weight='bold')

    # (c) Noise
    noise = {
        'FDAA-Net (Ours)': [99.08, 97.97],
        'F3Net':            [96.78, 89.49],
        'CNNDetection':     [96.23, 92.08],
        'FreqNet':          [96.17, 93.08],
        'UnivFD':           [96.30, 92.95],
    }
    ax = axes[2]
    ax.set_title('(c) Gaussian Noise', fontsize=9, pad=4)
    for name, cfg in methods.items():
        ax.plot(['Clean', r'$\sigma$=0.02', r'$\sigma$=0.05'], [clean[name]] + noise[name],
                color=cfg['c'], linestyle=cfg['ls'], marker=cfg['m'],
                markersize=4, markeredgewidth=0.4, markeredgecolor='white',
                zorder=cfg['z'], linewidth=cfg['lw'])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.13),
               ncol=5, frameon=True, fancybox=False, edgecolor='#CCCCCC',
               columnspacing=1.0, handletextpad=0.3, fontsize=7.5)

    for ax in axes:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(w_pad=0.5)
    save_fig(fig, 'fig1_robustness_degradation_curves')


# =====================================================================
# Fig 2: 消融域内 vs 鲁棒性对比 (FDAA核心论据)
# =====================================================================
def fig2_ablation_domain_vs_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    variants = ['Baseline', '+ FDAA', '+ MGFP', 'Full']
    colors = ['#BDC3C7', '#3498DB', '#2ECC71', '#C0392B']

    # (a) In-domain
    ax = axes[0]
    domain_auc = [99.49, 99.69, 99.87, 99.87]
    bars = ax.bar(variants, domain_auc, color=colors, edgecolor='white', width=0.6, zorder=3, alpha=0.88)
    ax.set_ylim(99.2, 100.05)
    ax.set_ylabel('AUC (%)')
    ax.set_title('(a) In-domain Detection', fontsize=9, pad=4)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    for i, auc in enumerate(domain_auc):
        ax.text(i, auc + 0.02, f'{auc:.2f}', ha='center', fontsize=7, weight='bold')
    ax.annotate(r'$\Delta$+0.20', xy=(1, 99.72), xytext=(1.5, 99.9),
                fontsize=7, ha='center', color='#3498DB', weight='bold',
                arrowprops=dict(arrowstyle='->', color='#3498DB', lw=0.6))

    # (b) Robustness
    ax = axes[1]
    rob_avg = [94.53, 95.59, 97.40, 97.79]
    bars = ax.bar(variants, rob_avg, color=colors, edgecolor='white', width=0.6, zorder=3, alpha=0.88)
    ax.set_ylim(93, 99.5)
    ax.set_ylabel('Avg Robustness AUC (%)')
    ax.set_title('(b) Robustness (Avg over 7 Perturbations)', fontsize=9, pad=4)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    for i, auc in enumerate(rob_avg):
        ax.text(i, auc + 0.1, f'{auc:.1f}', ha='center', fontsize=7, weight='bold')
    ax.annotate(r'$\Delta$+1.06', xy=(1, 95.75), xytext=(1, 96.8),
                fontsize=7, ha='center', color='#3498DB', weight='bold',
                arrowprops=dict(arrowstyle='->', color='#3498DB', lw=0.6))
    ax.annotate(r'5.3$\times$ larger', xy=(1.05, 97.0), xytext=(2.5, 96.0),
                fontsize=7, ha='center', color='#C0392B', weight='bold',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.6))

    for ax in axes:
        ax.tick_params(axis='x', rotation=12)
    plt.tight_layout(w_pad=1.0)
    save_fig(fig, 'fig2_ablation_domain_vs_robustness')


# =====================================================================
# Fig 3: 消融鲁棒性分组柱状图 (28组数据全展示)
# =====================================================================
def fig3_ablation_robustness_grouped_bar():
    rob = load_json('ablation_robustness_results.json')
    variants = ['abl_baseline', 'abl_baseline+fdaa', 'abl_baseline+mgfp', 'abl_full']
    labels = ['Baseline', '+ FDAA', '+ MGFP', 'Full (Ours)']
    colors = ['#BDC3C7', '#3498DB', '#2ECC71', '#C0392B']
    hatches = ['', '///', '\\\\\\', '']

    tests = ['jpeg_70', 'jpeg_50', 'jpeg_30', 'blur_1.0', 'blur_2.0', 'noise_0.02', 'noise_0.05']
    test_labels = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\n$\\sigma$=1', 'Blur\n$\\sigma$=2', 'Noise\n$\\sigma$=.02', 'Noise\n$\\sigma$=.05']

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    n_var = len(variants)
    x = np.arange(len(tests))
    bw = 0.82 / n_var

    for i, (var, label, color, hatch) in enumerate(zip(variants, labels, colors, hatches)):
        vals = [rob[t][var]['auc'] * 100 for t in tests]
        offset = (i - n_var / 2 + 0.5) * bw
        ax.bar(x + offset, vals, bw * 0.9, label=label, color=color, hatch=hatch,
               edgecolor='white' if not hatch else color, linewidth=0.3, zorder=3, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(test_labels)
    ax.set_ylabel('AUC (%)')
    ax.set_ylim(86, 101)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=8)

    # ΔAUC on JPEG-30
    b_j30 = rob['jpeg_30']['abl_baseline']['auc'] * 100
    f_j30 = rob['jpeg_30']['abl_full']['auc'] * 100
    ax.text(2, f_j30 + 0.6, f'+{f_j30 - b_j30:.1f}pp', fontsize=7, ha='center', color='#C0392B', weight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig3_ablation_robustness_grouped_bar')


# =====================================================================
# Fig 4: 跨数据集雷达图
# =====================================================================
def fig4_cross_dataset_radar():
    categories = ['UniversalFakeDetect', 'ForenSynths', 'Synthbuster', 'DiffusionForensics']
    data = {
        'FDAA-Net (Ours)': [96.31, 94.2, 82.2, 95.65],
        'UnivFD':           [88.63, 90.8, 79.6, 92.92],
        'F3Net':            [89.88, 83.6, 78.3, 88.89],
        'DIRE':             [91.47, 74.7, 71.0, 95.03],
        'CNNDetection':     [91.04, 77.5, 75.3, 92.84],
    }
    colors = ['#C0392B', '#E67E22', '#2980B9', '#8E44AD', '#27AE60']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    for (name, vals), color in zip(data.items(), colors):
        values = vals + vals[:1]
        lw = 1.8 if 'Ours' in name else 0.9
        ms = 4.5 if 'Ours' in name else 2.5
        ax.plot(angles, values, 'o-', color=color, linewidth=lw, markersize=ms,
                markeredgecolor='white', markeredgewidth=0.3, label=name)
        if 'Ours' in name:
            ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylim(60, 100)
    ax.set_yticks([70, 80, 90, 100])
    ax.set_yticklabels(['70', '80', '90', '100'], fontsize=6.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=7,
              frameon=True, fancybox=False, edgecolor='#CCCCCC')
    plt.tight_layout()
    save_fig(fig, 'fig4_cross_dataset_radar')


# =====================================================================
# Fig 5: 泛化热力图 (ForenSynths + Synthbuster)
# =====================================================================
def fig5_generalization_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    methods = ['FDAA-Net', 'UnivFD', 'F3Net', 'DIRE', 'CNNDet']

    # (a) ForenSynths
    gens_a = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'CycleGAN', 'GauGAN', 'BigGAN']
    data_a = np.array([
        [99.5, 94.5, 96.6, 94.7, 99.9, 100.0],
        [98.3, 84.7, 89.6, 91.1, 99.2, 99.9],
        [78.2, 73.3, 67.0, 81.5, 74.9, 78.5],
        [82.3, 68.1, 71.2, 68.2, 43.8, 66.7],
        [79.8, 70.7, 78.2, 68.2, 50.8, 73.2],
    ])
    ax = axes[0]
    im = ax.imshow(data_a, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax.set_xticks(range(len(gens_a)))
    ax.set_xticklabels(gens_a, rotation=35, ha='right', fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7.5)
    ax.set_title('(a) ForenSynths (GAN)', fontsize=9, pad=4)
    for i in range(len(methods)):
        for j in range(len(gens_a)):
            v = data_a[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if v < 65 else 'black', weight='bold' if i == 0 else 'normal')

    # (b) Synthbuster
    gens_b = ['SD 1.3', 'SD 1.4', 'SD 2.0', 'SD XL', 'MJ v5']
    data_b = np.array([
        [78.5, 76.6, 73.6, 90.8, 93.1],
        [74.0, 72.9, 72.1, 88.1, 87.7],
        [73.9, 75.7, 60.4, 89.4, 93.2],
        [64.3, 64.0, 58.6, 81.6, 86.7],
        [70.5, 72.3, 66.8, 85.7, 85.9],
    ])
    ax = axes[1]
    im2 = ax.imshow(data_b, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax.set_xticks(range(len(gens_b)))
    ax.set_xticklabels(gens_b, rotation=35, ha='right', fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7.5)
    ax.set_title('(b) Synthbuster (Diffusion)', fontsize=9, pad=4)
    for i in range(len(methods)):
        for j in range(len(gens_b)):
            v = data_b[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if v < 62 else 'black', weight='bold' if i == 0 else 'normal')

    cbar = fig.colorbar(im2, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout(w_pad=0.8)
    save_fig(fig, 'fig5_generalization_heatmap')


# =====================================================================
# Fig 6: 性能-鲁棒性 散点图 (Pareto优势)
# =====================================================================
def fig6_performance_robustness_scatter():
    pts = {
        'FDAA-Net (Ours)': (99.93, 97.79, 19.11, '#C0392B'),
        'F3Net':            (99.75, 94.47, 5.18,  '#2980B9'),
        'CNNDet':           (99.40, 94.08, 23.51, '#27AE60'),
        'FreqNet':          (99.07, 93.86, 23.38, '#8E44AD'),
        'UnivFD':           (99.13, 93.47, 0.39,  '#E67E22'),
        'DIRE':             (99.45, 92.73, 23.53, '#1ABC9C'),
        'LaRE$^2$':         (98.17, 92.07, 37.96, '#95A5A6'),
        'NPR':              (97.47, 90.92, 11.37, '#34495E'),
        'C2P-CLIP':         (89.79, 85.90, 18.37, '#BDC3C7'),
        'SPEC':             (97.12, 79.29, 1.55,  '#7F8C8D'),
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for name, (dom, rob, par, color) in pts.items():
        is_ours = 'Ours' in name
        ax.scatter(dom, rob, s=max(par * 2.5, 25), c=color,
                   alpha=1.0 if is_ours else 0.6, zorder=10 if is_ours else 5,
                   edgecolor='#C0392B' if is_ours else 'white',
                   linewidth=1.5 if is_ours else 0.5)
        ax.annotate(name, (dom, rob), textcoords='offset points', xytext=(5, 5),
                    fontsize=6.5, weight='bold' if is_ours else 'normal', color=color)

    ax.set_xlabel('In-domain AUC (%)')
    ax.set_ylabel('Avg Robustness AUC (%)')
    ax.set_xlim(88, 101)
    ax.set_ylim(77, 100)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=97.79, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.axvline(x=99.93, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    save_fig(fig, 'fig6_performance_robustness_scatter')


# =====================================================================
# Fig 7: LOO柱状图
# =====================================================================
def fig7_leave_one_out():
    generators = ['BigGAN', 'GLIDE', 'VQDM', 'SDv4', 'ADM', 'Midjourney']
    aucs = [99.66, 99.37, 99.37, 98.80, 91.84, 81.25]
    avg = np.mean(aucs)

    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    colors = ['#27AE60' if a > 95 else '#E67E22' if a > 85 else '#E74C3C' for a in aucs]
    ax.bar(generators, aucs, color=colors, edgecolor='white', width=0.6, zorder=3, alpha=0.85)
    ax.axhline(y=avg, color='#2C3E50', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(5.4, avg + 0.5, f'Avg: {avg:.1f}%', fontsize=7, color='#2C3E50')
    for i, auc in enumerate(aucs):
        ax.text(i, auc + 0.5, f'{auc:.1f}', ha='center', fontsize=7, weight='bold')
    ax.set_ylabel('AUC (%)')
    ax.set_ylim(75, 103)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=12)
    plt.tight_layout()
    save_fig(fig, 'fig7_leave_one_out')


# =====================================================================
# Fig 8: 鲁棒性SOTA对比柱状图 (Top6 × 7扰动)
# =====================================================================
def fig8_robustness_sota_bar():
    methods = ['FDAA-Net\n(Ours)', 'F3Net', 'CNNDet', 'FreqNet', 'UnivFD', 'DIRE']
    tests = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\n$\\sigma$=1', 'Blur\n$\\sigma$=2', 'Noise\n$\\sigma$=.02', 'Noise\n$\\sigma$=.05']
    data = {
        'FDAA-Net\n(Ours)': [99.01, 98.24, 96.33, 99.39, 94.53, 99.08, 97.97],
        'F3Net':             [97.04, 95.94, 93.66, 98.96, 89.39, 96.78, 89.49],
        'CNNDet':            [96.54, 95.65, 93.96, 98.42, 85.66, 96.23, 92.08],
        'FreqNet':           [96.61, 95.59, 93.40, 97.28, 84.92, 96.17, 93.08],
        'UnivFD':            [95.18, 93.79, 90.98, 97.04, 88.08, 96.30, 92.95],
        'DIRE':              [96.28, 94.81, 92.69, 98.39, 84.94, 94.89, 87.14],
    }

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    n_m = len(methods)
    x = np.arange(len(tests))
    bw = 0.82 / n_m
    colors = ['#C0392B'] + list(plt.cm.Blues(np.linspace(0.35, 0.8, n_m - 1)))

    for i, method in enumerate(methods):
        offset = (i - n_m / 2 + 0.5) * bw
        ax.bar(x + offset, data[method], bw * 0.9, label=method, color=colors[i],
               edgecolor='white', linewidth=0.2, zorder=3, alpha=0.92 if i == 0 else 0.72)

    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.set_ylabel('AUC (%)')
    ax.set_ylim(82, 102)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=6,
              frameon=True, fancybox=False, edgecolor='#CCCCCC',
              columnspacing=0.6, handletextpad=0.3, handlelength=1.0, fontsize=7)

    avg_ours = np.mean(data['FDAA-Net\n(Ours)'])
    ax.axhline(y=avg_ours, color='#C0392B', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.text(6.5, avg_ours + 0.3, f'Avg:{avg_ours:.1f}', fontsize=6.5, color='#C0392B', weight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig8_robustness_sota_bar')


# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating SCI Paper Figures for FDAA-Net")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    fig1_robustness_degradation_curves()
    fig2_ablation_domain_vs_robustness()
    fig3_ablation_robustness_grouped_bar()
    fig4_cross_dataset_radar()
    fig5_generalization_heatmap()
    fig6_performance_robustness_scatter()
    fig7_leave_one_out()
    fig8_robustness_sota_bar()

    print(f"\n{'=' * 60}")
    print(f"[Done] All 8 figures saved to {FIG_DIR}")
    print("=" * 60)
