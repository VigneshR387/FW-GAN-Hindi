import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_hdf5_dataset(metadata_path, output_hdf5, char2idx, image_base_dir='dataset'):
    """Convert your preprocessed dataset to FW-GAN HDF5 format."""

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    n_samples = len(metadata)

    # Determine max dimensions
    max_height = max(item['final_shape'][0] for item in metadata)
    max_width = max(item['final_shape'][1] for item in metadata)
    max_label_len = max(len(item['label']) for item in metadata)

    print(f"Creating HDF5 with {n_samples} samples")
    print(f"Max dimensions: {max_height}x{max_width}")
    print(f"Max label length: {max_label_len}")

    with h5py.File(output_hdf5, 'w') as hf:
        # Create datasets
        img_dset = hf.create_dataset('images',
                                     shape=(n_samples, max_height, max_width),
                                     dtype='float32',
                                     compression='gzip')

        label_dset = hf.create_dataset('labels',
                                       shape=(n_samples, max_label_len),
                                       dtype='int32',
                                       fillvalue=char2idx['<PAD>'])

        label_len_dset = hf.create_dataset('label_lengths',
                                           shape=(n_samples,),
                                           dtype='int32')

        # Optional: writer IDs (use image folder hierarchy or assign dummy)
        writer_dset = hf.create_dataset('writer_ids',
                                        shape=(n_samples,),
                                        dtype='int32',
                                        fillvalue=0)  # Will fill below

        # Process each sample
        writer_map = {}  # Map folder structure to writer ID
        next_writer_id = 0

        for idx, item in enumerate(tqdm(metadata, desc="Converting")):
            # Load preprocessed image
            img_path = Path(image_base_dir) / item['processed_path']
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0

            # Pad to max dimensions if needed
            h, w = img.shape
            padded = np.ones((max_height, max_width), dtype='float32')  # White padding
            padded[:h, :w] = img

            img_dset[idx] = padded

            # Encode label
            label_text = item['label']
            label_encoded = [char2idx.get(ch, char2idx['<UNK>']) for ch in label_text]
            label_len = len(label_encoded)

            label_dset[idx, :label_len] = label_encoded
            label_len_dset[idx] = label_len

            # Extract writer ID from original path (e.g., "HindiSeg/test/9/2/22.jpg" → writer "9")
            # Adjust this logic based on your actual writer grouping
            path_parts = Path(item['original_path']).parts
            if len(path_parts) > 2:
                writer_folder = path_parts[2]  # e.g., "9"
                if writer_folder not in writer_map:
                    writer_map[writer_folder] = next_writer_id
                    next_writer_id += 1
                writer_dset[idx] = writer_map[writer_folder]

        # Save metadata as attributes
        hf.attrs['num_samples'] = n_samples
        hf.attrs['max_height'] = max_height
        hf.attrs['max_width'] = max_width
        hf.attrs['num_writers'] = len(writer_map)

    print(f"✓ Saved {output_hdf5}")
    print(f"  Unique writers: {len(writer_map)}")
    return len(writer_map)

# Run conversion
char2idx_path = './data/hindi_char2idx.json'
with open(char2idx_path, 'r', encoding='utf-8') as f:
    char2idx = json.load(f)

# Convert each split
splits = {
    'train': ('dataset/train/metadata.json', './data/train_hindi.hdf5'),
    'val': ('dataset/val/metadata.json', './data/val_hindi.hdf5'),
    'test': ('dataset/test/metadata.json', './data/test_hindi.hdf5')
}


num_writers_total = 0
for split_name, (meta_path, hdf5_path) in splits.items():
    print(f"\n=== Processing {split_name} split ===")
    n_writers = create_hdf5_dataset(meta_path, hdf5_path, char2idx, image_base_dir='dataset')
    num_writers_total = max(num_writers_total, n_writers)

print(f"\n>>> Update config: n_writer: {num_writers_total}")
