import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--target", required=True); parser.add_argument("--output", required=True); parser.add_argument("--seed", type=int, default=7); args = parser.parse_args()
    root = Path(args.root); domains = sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.target not in domains:
        parser.error("target must name a root/domain directory")
    labels = sorted({p.name for domain in domains for p in (root / domain).iterdir() if p.is_dir()})
    rng = np.random.default_rng(args.seed); rows = []
    for domain in domains:
        for label in labels:
            paths = sorted(p for p in (root / domain / label).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            for p in paths:
                split = "test" if domain == args.target else "val" if rng.random() < .2 else "train"
                rows.append({"path": str(p.resolve()), "label": labels.index(label), "domain": domains.index(domain), "split": split})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "domain", "split"]); writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({"domains": domains, "classes": labels, "target_domain": domains.index(args.target)}, indent=2))


if __name__ == "__main__":
    main()
