#!/usr/bin/env python3
"""
Synthbuster 跨数据集评估
包含: SD 1.3, SD 1.4, SD 2.0, SD XL, DALL-E 2, DALL-E 3, Midjourney v5, Firefly, Stable Cascade
"""

import sys, os, json, glob
from pathlib import Path
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
import pyarrow.ipc as ipc

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt,
    evaluate_model, create_sota_model, get_multi_source_transforms,
)


class SynthbusterArrowDataset(Dataset):
    """从 HF arrow 文件加载 Synthbuster 数据
    Synthbuster 路径: generator/filename.png
    real_RAISE_1k = 真实图像, 其他文件夹 = 伪造图像
    评估时: 用 real_RAISE_1k 作为 real, 指定生成器作为 fake
    """

    def __init__(self, arrow_files, generator_filter=None, transform=None, max_per_class=1000):
        self.transform = transform
        self.samples = []

        for f in arrow_files:
            reader = ipc.open_stream(f)
            table = reader.read_all()

            for i in range(table.num_rows):
                path = table['image_path'][i].as_py()
                parts = path.split('/')
                if len(parts) < 2:
                    continue
                gen = parts[0]

                # real_RAISE_1k is real images
                if gen == 'real_RAISE_1k':
                    img_data = table['image'][i].as_py()
                    self.samples.append((img_data, 0))  # real
                elif generator_filter and gen == generator_filter:
                    img_data = table['image'][i].as_py()
                    self.samples.append((img_data, 1))  # fake

        real = [(d, l) for d, l in self.samples if l == 0][:max_per_class]
        fake = [(d, l) for d, l in self.samples if l == 1][:max_per_class]
        self.samples = real + fake
        print(f"    {len(real)} real + {len(fake)} fake = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_data, label = self.samples[idx]
        try:
            img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        except:
            try:
                img = Image.open(BytesIO(img_data)).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label, 'path': ''}


def download_synthbuster_files(num_files=10):
    """Download synthbuster arrow files from HF mirror"""
    from huggingface_hub import hf_hub_download

    files = []
    for i in range(num_files):
        fname = f"synthbuster/data-{i:05d}-of-00031.arrow"
        try:
            f = hf_hub_download("nebula/DF-arrow", fname, repo_type="dataset")
            files.append(f)
            print(f"  [{i+1}/{num_files}] Downloaded {fname}")
        except Exception as e:
            print(f"  [Warning] {fname}: {e}")
            break
    return files


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    config = PAPER_CONFIG.copy()
    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # Load models
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)

    from models.detector import AIGCDetectorV2
    loaded_models = {}

    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if not os.path.exists(ours_ckpt):
        ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_6src_best.pth')

    m = AIGCDetectorV2(backbone_name=config['backbone'], num_classes=2,
                       img_size=config['img_size'], embed_dim=config['embed_dim'],
                       use_hierarchical=config['use_hierarchical'], dropout=config['dropout'])
    ckpt = torch.load(ours_ckpt, map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    loaded_models['Ours_V2'] = m.to(device).eval()
    print("  [Loaded] Ours_V2")

    for method in config['sota_methods']:
        ckpt_path = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt_path) and not os.path.islink(ckpt_path):
            try:
                sm = create_sota_model(method, num_classes=2)
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                sm.load_state_dict(ckpt['model_state_dict'])
                loaded_models[method] = sm.to(device).eval()
                print(f"  [Loaded] {method}")
            except Exception as e:
                print(f"  [Error] {method}: {e}")

    # Download synthbuster files
    print("\n" + "=" * 60)
    print("Downloading Synthbuster files from HF Mirror")
    print("=" * 60)

    # Check cache first
    cache_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/synthbuster/*.arrow")
    cached_files = sorted(glob.glob(cache_pattern))
    print(f"  Cached: {len(cached_files)} files")

    if len(cached_files) < 10:
        new_files = download_synthbuster_files(num_files=10)
        cached_files = sorted(glob.glob(cache_pattern))
    print(f"  Total: {len(cached_files)} files")

    if not cached_files:
        print("[Error] No synthbuster files!")
        return

    # Scan generators
    print("\n  Scanning generators...")
    gen_counter = Counter()
    for f in cached_files:
        reader = ipc.open_stream(f)
        table = reader.read_all()
        for path in table['image_path'].to_pylist():
            parts = path.split('/')
            gen = parts[0]
            gen_counter[gen] += 1

    print(f"\n  Available generators:")
    real_count = gen_counter.get('real_RAISE_1k', 0)
    print(f"    real_RAISE_1k (real):   {real_count}")
    for g in sorted(gen_counter):
        if g != 'real_RAISE_1k':
            print(f"    {g:<25} (fake): {gen_counter[g]}")

    # Filter to fake generators with enough samples
    valid_gens = [g for g in sorted(gen_counter) if g != 'real_RAISE_1k' and gen_counter[g] >= 20]

    # Evaluate
    print("\n" + "=" * 60)
    print("Synthbuster Cross-Dataset Evaluation")
    print("=" * 60)

    gen_display = {
        'stable_diffusion_1_3': 'SD 1.3',
        'stable_diffusion_1_4': 'SD 1.4',
        'stable_diffusion_2': 'SD 2.0',
        'stable_diffusion_xl': 'SD XL',
        'dalle2': 'DALL-E 2',
        'dalle3': 'DALL-E 3',
        'midjourney_v5': 'MJ v5',
        'midjourney5': 'MJ v5',
        'firefly': 'Firefly',
        'stable_cascade': 'Stable Cascade',
        'glide': 'GLIDE',
    }

    results = {}
    for gen in valid_gens:
        display = gen_display.get(gen, gen)
        print(f"\n--- {display} ({gen}) ---")
        dataset = SynthbusterArrowDataset(cached_files, generator_filter=gen, transform=transform)
        if len(dataset) < 10:
            print("  [Skip] Too few samples")
            continue
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        results[display] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {display}')
                results[display][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[display][model_name] = {'error': str(e)}

        save_json(results, os.path.join(results_dir, 'crossdataset_synthbuster_results.json'))
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    all_models = set()
    for gen_results in results.values():
        for m, v in gen_results.items():
            if isinstance(v, dict) and 'auc' in v:
                all_models.add(m)
    models = sorted(all_models, key=lambda x: (0 if 'Ours' in x else 1, x))
    gens = list(results.keys())

    for model in models:
        aucs = []
        for g in gens:
            auc = results.get(g, {}).get(model, {}).get('auc', 0) * 100
            aucs.append(auc)
        avg = np.mean(aucs)
        print(f"  {model:<15}: avg={avg:.2f}%")

    del loaded_models
    torch.cuda.empty_cache()
    print("\n[Done]")


if __name__ == '__main__':
    main()
