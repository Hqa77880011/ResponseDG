"""Acquire public CIFAR-10 PNGs and prepare grouped appearance domains."""

import argparse
import csv
import json
import io
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SOURCE = "https://www.cs.toronto.edu/~kriz/cifar.html"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
MIRROR_SOURCE = "https://registry.opendata.aws/fast-ai-imageclas/"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
DOMAINS = ["clean", "warm", "low_contrast", "heldout_cold_blur"]


def download(archive):
    """Download once; only expose a complete transfer at the final path."""
    archive = Path(archive)
    if archive.exists():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    with urlopen(URL, timeout=120) as response, partial.open("wb") as target:
        shutil.copyfileobj(response, target)
        expected = response.headers.get("Content-Length")
    if expected and partial.stat().st_size != int(expected):
        raise OSError(f"Incomplete download: {partial}")
    partial.replace(archive)


def read_records(archive, split):
    """Read the fast.ai CIFAR-10 PNG distribution without extracting paths."""
    expected = 5000 if split == "train" else 1000
    records = []
    with tarfile.open(archive, "r:gz") as bundle:
        members = []
        for member in bundle.getmembers():
            parts = PurePosixPath(member.name).parts
            if (len(parts) == 4 and parts[:2] == ("cifar10", split)
                    and parts[2] in CLASSES and parts[3].endswith(".png")):
                if not member.isfile():
                    raise ValueError(f"Expected a regular PNG: {member.name}")
                members.append(member)
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate image identities in CIFAR archive")
        counts = {label: 0 for label in CLASSES}
        for member in sorted(members, key=lambda m: m.name):
            label = PurePosixPath(member.name).parts[2]
            with bundle.extractfile(member) as stream:
                payload = stream.read()
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != (32, 32):
                    raise ValueError("CIFAR images must have native 32x32 resolution")
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            record = np.empty(3073, dtype=np.uint8)
            record[0] = CLASSES.index(label)
            record[1:] = pixels.transpose(2, 0, 1).reshape(-1)
            records.append(record)
            counts[label] += 1
        if any(count != expected for count in counts.values()):
            raise ValueError(f"Unexpected per-class {split} population: {counts}")
    return np.stack(records)


def select_ids(labels, train_per_class, val_per_class, seed):
    """Assign each original image once before making any domain copies."""
    rng = np.random.default_rng(seed)
    result = {}
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(labels == label))
        if len(ids) < train_per_class + val_per_class:
            raise ValueError(f"Insufficient examples for class {label}")
        result.update((int(i), "train") for i in ids[:train_per_class])
        result.update((int(i), "val") for i in ids[train_per_class:train_per_class + val_per_class])
    return result


def appearance(image, domain):
    """Fixed label-independent photometric domain, applied at native size."""
    if domain == "warm":
        pixels = np.asarray(image, dtype=float) * np.array([1.12, 1., .78])
        return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
    if domain == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(.55)
    if domain == "heldout_cold_blur":
        pixels = np.asarray(image, dtype=float) * np.array([.78, 1., 1.12])
        return Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6))
    return image.copy()


def prepare(archive, output, train_per_class=200, val_per_class=50, test_per_class=100, seed=7):
    """Export a manifest consumed directly by responsedg train/evaluate."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    training = read_records(archive, "train")
    testing = read_records(archive, "test")
    assignments = select_ids(training[:, 0], train_per_class, val_per_class, seed)
    test_ids = select_ids(testing[:, 0], test_per_class, 0, seed + 1)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for official, records, selected in [("train", training, assignments), ("test", testing, test_ids)]:
        for index, role in sorted(selected.items()):
            pixels = records[index, 1:].reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(pixels)
            label = int(records[index, 0])
            for domain in (DOMAINS[:3] if official == "train" else DOMAINS[3:]):
                relative = Path("images") / domain / CLASSES[label] / f"{official}_{index:05d}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                appearance(image, domain).save(destination)
                rows.append(dict(path=str(destination), label=label, domain=DOMAINS.index(domain),
                                 split=role if official == "train" else "test",
                                 image_id=f"{official}_{index:05d}", original_split=official))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source=SOURCE, mirror_source=MIRROR_SOURCE, archive_url=URL, seed=seed, domains=DOMAINS, classes=CLASSES,
                    target_domain=3, train_per_class=train_per_class, val_per_class=val_per_class,
                    test_per_class=test_per_class, native_size=32,
                    image_policy="RGB domain transforms at 32px; existing loader resizes to 128px",
                    split_policy="original image identity shared only within one source role; official test is target-only")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} rows in {output / 'manifest.csv'}; target-domain=3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/cifar10.tgz"))
    parser.add_argument("--output", type=Path, default=Path("data/cifar-domains"))
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offline", action="store_true", help="Read an existing fast.ai CIFAR-10 PNG archive only")
    parser.add_argument("--plan", action="store_true", help="Print input source and selected settings without I/O")
    args = parser.parse_args()
    if min(args.train_per_class, args.val_per_class, args.test_per_class) < 1:
        parser.error("Per-class counts must be positive")
    if args.train_per_class + args.val_per_class > 5000 or args.test_per_class > 1000:
        parser.error("Counts exceed the official per-class train/test populations")
    if args.plan:
        print(json.dumps(dict(source=SOURCE, mirror_source=MIRROR_SOURCE, download=URL, archive=str(args.archive), output=str(args.output),
                              domains=DOMAINS, target_domain=3,
                              train_rows=30 * args.train_per_class, val_rows=30 * args.val_per_class,
                              test_rows=10 * args.test_per_class, seed=args.seed), indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    if args.offline and not args.archive.is_file():
        parser.error("--offline requires an existing --archive")
    if not args.offline:
        download(args.archive)
    prepare(args.archive, args.output, args.train_per_class, args.val_per_class, args.test_per_class, args.seed)


if __name__ == "__main__":
    main()
