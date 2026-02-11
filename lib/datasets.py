import os, h5py
import numpy as np
from PIL import Image


import json
import cv2
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from lib.alphabet import strLabelConverter
from lib.path_config import data_roots, data_paths


class Hdf5Dataset(Dataset):
    def __init__(self, root, split, transforms=None, alphabet_key='all'):
        super(Hdf5Dataset, self).__init__()
        self.root = root
        self._load_h5py(split)
        self.transforms = transforms
        self.label_converter = strLabelConverter(alphabet_key)

    def _load_h5py(self, split):
        self.file_path = os.path.join(self.root, split)
        h5f = h5py.File(self.file_path, 'r')
        self.imgs, self.lbs = h5f['imgs'][:], h5f['lbs'][:]
        self.img_seek_idxs, self.lb_seek_idxs = h5f['img_seek_idxs'][:], h5f['lb_seek_idxs'][:]
        self.img_lens, self.lb_lens = h5f['img_lens'][:], h5f['lb_lens'][:]
        self.wids = h5f['wids'][:]

    def __getitem__(self, idx):
        img_seek_idx, img_len = self.img_seek_idxs[idx], self.img_lens[idx]
        lb_seek_idx, lb_len = self.lb_seek_idxs[idx], self.lb_lens[idx]
        img = self.imgs[:, img_seek_idx : img_seek_idx + img_len]
        text = ''.join(chr(ch) for ch in self.lbs[lb_seek_idx : lb_seek_idx + lb_len])
        lb = self.label_converter.encode(text)
        wid = self.wids[idx]

        img = Image.fromarray(img, mode='L')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, lb, wid

    def __len__(self):
        return len(self.img_lens)

    @staticmethod
    def collect_fn(batch):
        def _recalc_len(leng, scale):
            tmp = leng % scale
            return leng + scale - tmp if tmp != 0 else leng

        imgs, lbs, wids, lb_lens, img_lens, pad_img_lens = [], [], [], [], [], []

        for img, lb, wid in batch:
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            imgs.append(img)
            lbs.append(lb)
            wids.append(wid)
            lb_lens.append(len(lb))
            img_lens.append(img.shape[-1])
            pad_img_lens.append(_recalc_len(img.shape[-1], img.shape[-2] // 2))

        bz = len(lb_lens)
        imgHeight = imgs[0].shape[-2]
        max_img_len = max(pad_img_lens)
        pad_imgs = -np.ones((bz, 1, imgHeight, max_img_len))
        for i, (img, img_len) in enumerate(zip(imgs, img_lens)):
            pad_imgs[i, 0, :, :img_len] = img

        max_lb_len = max(lb_lens)
        pad_lbs = np.zeros((bz, max_lb_len))
        for i, (lb, lb_len) in enumerate(zip(lbs, lb_lens)):
            pad_lbs[i, :lb_len] = lb

        imgs = torch.from_numpy(pad_imgs).float()
        img_lens = torch.Tensor(pad_img_lens).int()
        lbs = torch.from_numpy(pad_lbs).int()
        lb_lens = torch.Tensor(lb_lens).int()
        wids = torch.Tensor(wids).long()
        return imgs, img_lens, lbs, lb_lens, wids

    @staticmethod
    def sort_collect_fn(batch):
        imgs, lbs, wids = zip(*batch)
        img_lens = np.array([img.size(-1) for img in imgs]).astype(np.int32)
        idx = np.argsort(img_lens)[::-1]
        imgs = [imgs[i] for i in idx]
        lbs = [lbs[i] for i in idx]
        wids = [wids[i] for i in idx]
        return Hdf5Dataset.collect_fn(zip(imgs, lbs, wids))

    @staticmethod
    def merge_batch(batch1, batch2, device):
        imgs1, img_lens1, lbs1, lb_lens1, wids1 = batch1
        imgs2, img_lens2, lbs2, lb_lens2, wids2 = batch2
        bz1, bz2 = imgs1.size(0), imgs2.size(0)

        max_img_len = max(img_lens1.max(), img_lens2.max()).item()
        pad_imgs = -torch.ones((bz1 + bz2, imgs1.size(1), imgs1.size(2), max_img_len)).float().to(device)
        pad_imgs[:bz1, :, :, :imgs1.size(-1)] = imgs1
        pad_imgs[bz1:, :, :, :imgs2.size(-1)] = imgs2

        max_lb_len = max(lb_lens1.max(), lb_lens2.max()).item()
        pad_lbs = torch.zeros((bz1 + bz2, max_lb_len)).long().to(device)
        pad_lbs[:bz1, :lbs1.size(-1)] = lbs1
        pad_lbs[bz1:, :lbs2.size(-1)] = lbs2

        merge_img_lens = torch.cat([img_lens1, img_lens2]).to(device)
        merge_lb_lens = torch.cat([lb_lens1, lb_lens2]).to(device)
        merge_wids = torch.cat([wids1, wids2]).long().to(device)

        return pad_imgs, merge_img_lens, pad_lbs, merge_lb_lens, merge_wids


def get_dataset(name, split):
    tag = '_'.join(name.split('_')[:2])
    alphabet_key = 'rimes_word' if tag.startswith('rimes') else 'all'
    transforms = [ToTensor(), Normalize([0.5], [0.5])]
    dataset = Hdf5Dataset(data_roots[tag],
                          data_paths[name][split],
                          transforms=Compose(transforms),
                          alphabet_key=alphabet_key)
    return dataset


def get_collect_fn(sort_input=False):
    if sort_input:
        return Hdf5Dataset.sort_collect_fn
    else:
        return Hdf5Dataset.collect_fn


def get_max_image_width(dset):
    max_image_width = 0
    for img, _, _ in dset:
        max_image_width = max(max_image_width, img.size(-1))
    return max_image_width


# ============================================================
# Hindi Word Dataset (JSON-based, no HDF5 needed)
# ============================================================


class HindiWordDataset(Dataset):
    """
    Dataset loader for Hindi handwriting using metadata.json
    Compatible with FW-GAN's existing data pipeline
    """

    def __init__(
        self,
        metadata_path: str,
        char2idx: Dict[str, int],
        image_base_dir: str = '.',
        transforms=None,
        max_width: int = 768,
        target_height: int = 128
    ):
        super(HindiWordDataset, self).__init__()
        self.image_base_dir = Path(image_base_dir)
        self.char2idx = char2idx
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.transforms = transforms
        self.max_width = max_width
        self.target_height = target_height

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"✓ Loaded {len(self.metadata)} samples from {metadata_path}")

        # Build writer ID mapping
        self.writer_map = self._build_writer_map()
        print(f"✓ Found {len(self.writer_map)} unique writers")

    def _build_writer_map(self) -> Dict[str, int]:
        """Extract writer IDs from original paths."""
        writers = set()
        for item in self.metadata:
            path_parts = Path(item['original_path']).parts
            if len(path_parts) > 2:
                writer_id = path_parts[2]
                writers.add(writer_id)
        return {w: idx for idx, w in enumerate(sorted(writers))}

    def _get_writer_id(self, original_path: str) -> int:
        """Get writer ID from original path."""
        path_parts = Path(original_path).parts
        if len(path_parts) > 2:
            writer_str = path_parts[2]
            return self.writer_map.get(writer_str, 0)
        return 0

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        item = self.metadata[idx]

        # Load preprocessed image
        img_path = self.image_base_dir / item['processed_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0

        # Pad to max_width if needed
        h, w = img.shape
        if w < self.max_width:
            padded = np.ones((h, self.max_width), dtype='float32')
            padded[:, :w] = img
            img = padded
        elif w > self.max_width:
            img = img[:, :self.max_width]

        # Convert to PIL Image for transforms compatibility
        img_uint8 = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')

        # Apply transforms (same as IAM: ToTensor, Normalize)
        if self.transforms is not None:
            img = self.transforms(img_pil)

        # Encode label (character-level)
        label_text = item['label']
        label_encoded = [self.char2idx.get(ch, self.char2idx.get('<UNK>', 0))
                        for ch in label_text]
        label_encoded = np.array(label_encoded, dtype=np.int32)

        # Get writer ID
        wid = self._get_writer_id(item['original_path'])

        return img, label_encoded, wid

    @property
    def num_writers(self) -> int:
        return len(self.writer_map)


def get_hindi_dataset(metadata_path, char2idx, image_base_dir, transforms=None):
    """
    Helper function to create Hindi dataset with same interface as get_dataset()
    """
    dataset = HindiWordDataset(
        metadata_path=metadata_path,
        char2idx=char2idx,
        image_base_dir=image_base_dir,
        transforms=transforms,
        max_width=768,
        target_height=128
    )
    return dataset
