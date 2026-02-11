#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify Hindi dataset setup before training
"""

import json
import yaml
import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose, ToTensor, Normalize
from lib.datasets import HindiWordDataset, Hdf5Dataset

print("=" * 70)
print("FW-GAN Hindi Dataset Setup Test")
print("=" * 70)

# 1. Check vocabulary files
print("\n[1/7] Checking vocabulary files...")
vocab_files = [
    'data/hindi_char2idx.json',
    'data/hindi_char_vocab.txt'
]
for f in vocab_files:
    if Path(f).exists():
        print(f"  ✓ {f} exists")
    else:
        print(f"  ✗ {f} MISSING!")
        exit(1)

with open('data/hindi_char2idx.json', 'r', encoding='utf-8') as f:
    char2idx = json.load(f)
print(f"  ✓ Vocabulary size: {len(char2idx)}")
print(f"  Sample chars: {list(char2idx.keys())[:10]}")

# 2. Check config
print("\n[2/7] Checking config...")
config_path = 'configs/fw_gan_hindi.yml'
if not Path(config_path).exists():
    print(f"  ✗ {config_path} not found!")
    exit(1)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"  ✓ Config loaded")
print(f"  - dataset: {config['dataset']}")
print(f"  - img_height: {config['img_height']}")
print(f"  - n_class: {config['training']['n_class']}")
print(f"  - bottom_height: {config['GenModel']['bottom_height']}")

# Check critical values
errors = []
if config['dataset'] != 'hindi_word':
    errors.append("dataset should be 'hindi_word'")
if config['img_height'] != 128:
    errors.append("img_height should be 128")
if config['training']['n_class'] != len(char2idx):
    errors.append(f"n_class ({config['training']['n_class']}) != vocab size ({len(char2idx)})")
if config['GenModel']['bottom_height'] != 8:
    errors.append("GenModel.bottom_height should be 8 for 128px images")

if errors:
    print("\n  ✗ Config errors found:")
    for err in errors:
        print(f"    - {err}")
    exit(1)

# 3. Check metadata files
print("\n[3/7] Checking metadata files...")
if 'hindi_metadata' not in config:
    print("  ✗ 'hindi_metadata' section missing in config!")
    exit(1)

for split in ['train', 'val', 'test']:
    meta_path = config['hindi_metadata'][split]
    if Path(meta_path).exists():
        with open(meta_path, 'r') as f:
            data = json.load(f)
        print(f"  ✓ {split}: {len(data)} samples in {meta_path}")
    else:
        print(f"  ✗ {split}: {meta_path} not found!")
        exit(1)

# 4. Load dataset
print("\n[4/7] Loading Hindi dataset...")
transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])

try:
    dataset = HindiWordDataset(
        metadata_path=config['hindi_metadata']['train'],
        char2idx=char2idx,
        image_base_dir=config['hindi_metadata']['image_base_dir'],
        transforms=transforms,
        max_width=768,
        target_height=128
    )
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    print(f"  ✓ Writers: {dataset.num_writers}")
except Exception as e:
    print(f"  ✗ Failed to load dataset: {e}")
    exit(1)

# 5. Test single sample
print("\n[5/7] Testing single sample...")
try:
    img, label, wid = dataset[0]
    print(f"  ✓ Sample loaded")
    print(f"  - Image shape: {img.shape}")  # Should be (1, 128, W)
    print(f"  - Image dtype: {img.dtype}")
    print(f"  - Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  - Label shape: {label.shape}")
    print(f"  - Label: {label}")
    print(f"  - Writer ID: {wid}")

    # Decode label back to text
    text = ''.join([list(char2idx.keys())[list(char2idx.values()).index(int(i))] for i in label])
    print(f"  - Decoded text: '{text}'")
except Exception as e:
    print(f"  ✗ Failed to load sample: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test batch with collate_fn
print("\n[6/7] Testing batch loading...")
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=Hdf5Dataset.collect_fn  # Reuse existing collate
    )

    imgs, img_lens, lbs, lb_lens, wids = next(iter(loader))
    print(f"  ✓ Batch loaded")
    print(f"  - Images shape: {imgs.shape}")  # (B, 1, 128, W)
    print(f"  - Image lengths: {img_lens}")
    print(f"  - Labels shape: {lbs.shape}")
    print(f"  - Label lengths: {lb_lens}")
    print(f"  - Writer IDs: {wids}")
except Exception as e:
    print(f"  ✗ Failed to load batch: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 7. Verify image normalization
print("\n[7/7] Verifying normalization...")
img_min = imgs.min().item()
img_max = imgs.max().item()
print(f"  - Min pixel value: {img_min:.4f}")
print(f"  - Max pixel value: {img_max:.4f}")

# After Normalize([0.5], [0.5]), range should be roughly [-1, 1]
if img_min < -1.5 or img_max > 1.5:
    print(f"  ⚠ WARNING: Unusual pixel range after normalization")
else:
    print(f"  ✓ Pixel range looks correct (normalized to [-1, 1])")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nNext steps:")
print("  1. Make sure you've modified networks/model.py to load Hindi dataset")
print("  2. Run: python train.py --config configs/fw_gan_hindi.yml")
print("\nTo train from scratch:")
print("  python train.py --config configs/fw_gan_hindi.yml")
print("\nTo fine-tune from IAM weights (if available):")
print("  python train.py --config configs/fw_gan_hindi.yml")
print("=" * 70)
