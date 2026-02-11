import json
from pathlib import Path

def count_writers_from_metadata(base_dir='.'):
    """Extract unique writers from processed_path in metadata.json."""
    all_writers = set()
    writer_counts = {}

    for split in ['train', 'val', 'test']:
        meta_path = Path(base_dir) / split / 'metadata.json'

        if not meta_path.exists():
            print(f"Warning: {meta_path} not found")
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        split_writers = set()
        for item in data:
            # Parse: "test/HindiSeg/test/9/2/22.jpg" → writer_id = "9"
            #        parts[0] / parts[1] / parts[2] / parts[3] / ...
            path_parts = Path(item['processed_path']).parts
            if len(path_parts) >= 4:
                writer_id = path_parts[3]  # Fourth component
                all_writers.add(writer_id)
                split_writers.add(writer_id)

        writer_counts[split] = len(split_writers)
        print(f"{split:5s}: {len(split_writers):2d} writers, {len(data):5d} images")

    return all_writers, writer_counts

# Run
writers, counts = count_writers_from_metadata()
num_writers = len(writers)

print(f"\n{'='*50}")
print(f"Total unique writers: {num_writers}")
print(f"Writer IDs: {sorted(writers, key=int)}")
print(f"{'='*50}")
print(f"\n>>> Update fw_gan_hindi.yml:")
print(f"    WidModel:")
print(f"      n_writer: {num_writers}")
