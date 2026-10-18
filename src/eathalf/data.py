"""Deterministic, content-deduplicated splits without copying dataset images."""
import csv
import hashlib
from pathlib import Path

CLASSES = (
    'apple_pie', 'chocolate_cake', 'donuts', 'falafel', 'french_fries',
    'hot_dog', 'ice_cream', 'nachos', 'onion_rings', 'pancakes', 'pizza',
    'ravioli', 'samosa', 'spring_rolls', 'strawberry_shortcake', 'tacos', 'waffles',
)


def prepare(data_dir, output_dir, seed=42):
    """Create a new 70/15/15 custom split. Refuse to overwrite an existing run."""
    from PIL import Image
    from sklearn.model_selection import train_test_split

    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError('Output exists. Choose a new run directory to preserve evidence.')
    if not data_dir.is_dir():
        raise ValueError(f'Dataset directory does not exist: {data_dir}')
    rows, invalid, duplicates, hashes = [], [], [], {}
    missing = [name for name in CLASSES if not (data_dir / name).is_dir()]
    if missing:
        raise ValueError(f'Missing class directories: {missing}. Point --data-dir at the class folders.')
    for label in CLASSES:
        for path in sorted((data_dir / label).iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            except (OSError, ValueError) as exc:
                invalid.append({'path': str(path.relative_to(data_dir)), 'reason': str(exc)})
                continue
            relative = path.relative_to(data_dir).as_posix()
            if digest in hashes:
                previous = hashes[digest]
                if previous['label'] != label:
                    raise ValueError(f'Identical image has conflicting labels: {previous["path"]}, {relative}')
                duplicates.append({'path': relative, 'kept': previous['path'], 'sha256': digest})
                continue
            row = {'path': relative, 'label': label, 'sha256': digest}
            hashes[digest] = row
            rows.append(row)
    counts = {name: sum(row['label'] == name for row in rows) for name in CLASSES}
    if min(counts.values()) < 10:
        raise ValueError(f'Need at least 10 valid unique images per class: {counts}')
    train, holdout = train_test_split(rows, test_size=.30, stratify=[r['label'] for r in rows], random_state=seed)
    val, test = train_test_split(holdout, test_size=.50, stratify=[r['label'] for r in holdout], random_state=seed)
    output_dir.mkdir(parents=True)
    for name, group in [('train', train), ('val', val), ('test', test)]:
        write_csv(output_dir / f'{name}.csv', group, ['path', 'label', 'sha256'])
    write_csv(output_dir / 'invalid_images.csv', invalid, ['path', 'reason'])
    write_csv(output_dir / 'removed_duplicates.csv', duplicates, ['path', 'kept', 'sha256'])
    import json
    (output_dir / 'split_summary.json').write_text(json.dumps({
        'seed': seed, 'split': 'custom_stratified_70_15_15', 'classes': list(CLASSES),
        'valid_unique_images': len(rows), 'class_counts': counts,
        'split_counts': {'train': len(train), 'val': len(val), 'test': len(test)},
        'invalid_images': len(invalid), 'duplicates_removed': len(duplicates),
        'deduplication': 'SHA256 of RGB dimensions and pixel bytes; near-duplicates not detected',
    }, indent=2), encoding='utf-8')
    return output_dir


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))
