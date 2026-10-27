#!/usr/bin/env python3
"""Create an ImageFolder copy with the shorter image side resized to 320."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile

from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Plain console fallback when tqdm is unavailable."""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        @staticmethod
        def write(message):
            print(message, flush=True)

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def convert(task):
    """Preserve aspect ratio and filename; atomically publish each RGB image."""
    source, destination, size, quality = task
    temporary = None
    try:
        if destination.exists():
            # Validate completed files before skipping them on resume.
            with Image.open(destination) as image:
                if min(image.size) != size:
                    raise ValueError('Existing output has an unexpected size.')
                image.verify()
            return 'skipped', str(source), ''
        with Image.open(source) as image:
            # Keep the same orientation convention as torchvision's PIL loader.
            image = image.convert('RGB')
            width, height = image.size
            if width <= height:
                new_size = (size, int(size * height / width))
            else:
                new_size = (int(size * width / height), size)
            if image.size != new_size:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.resize-', suffix=destination.suffix,
                                             dir=destination.parent)
            os.close(fd)
            options = {'quality': quality, 'subsampling': 0} if destination.suffix.lower() in {'.jpg', '.jpeg'} else {}
            image.save(temporary, **options)
            os.replace(temporary, destination)
            temporary = None
        return 'written', str(source), ''
    except Exception as error:
        return 'failed', str(source), str(error)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Original train directory containing class folders.')
    parser.add_argument('--output', required=True, type=Path, help='New train directory; originals are preserved.')
    parser.add_argument('--size', type=int, default=320, help='Target shorter side, in pixels.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--quality', type=int, default=95, help='JPEG encoding quality.')
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error('Input directory does not exist.')
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error('Input and output must be separate, non-nested directories.')
    if args.size < 1 or args.workers < 1 or not 1 <= args.quality <= 100:
        parser.error('Invalid size, workers or quality.')
    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if not classes:
        parser.error('No class directories found.')
    destination.mkdir(parents=True, exist_ok=True)
    settings = {'source': str(source), 'size': args.size, 'quality': args.quality,
                'interpolation': 'LANCZOS', 'jpeg_subsampling': 0}
    manifest = destination / 'resize_settings.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            parser.error('Output was created with different settings; choose a new output directory.')
    elif any(destination.iterdir()):
        parser.error('Output must be empty for a first run.')
    else:
        manifest.write_text(json.dumps(settings, indent=2))
    counts = dict(written=0, skipped=0, failed=0)
    error_path = destination / 'resize_errors.jsonl'
    with error_path.open('w') as errors, ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Submit one class at a time to bound queued work on large ImageNet datasets.
        for folder in tqdm(classes, desc='Classes', unit='class'):
            files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            (destination / folder.name).mkdir(exist_ok=True)
            tasks = ((p, destination / p.relative_to(source), args.size, args.quality) for p in files)
            for status, filename, error in pool.map(convert, tasks):
                counts[status] += 1
                if status == 'failed':
                    errors.write(json.dumps({'file': filename, 'error': error}) + '\n')
            tqdm.write(f'{folder.name}: written={counts["written"]}, skipped={counts["skipped"]}, failed={counts["failed"]} (total)')
    print(json.dumps(counts, indent=2))
    if counts['failed']:
        print(f'Inspect failed images in {error_path}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
