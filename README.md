# ResponseDG

Implementation of **ResponseDG: Mechanism-Preserving Domain Generalization via Intervention-Response Factorization**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Download CIFAR-10 from the [fast.ai public image mirror](https://registry.opendata.aws/fast-ai-imageclas/) and create four fixed appearance domains. Originals are assigned roles before source-domain copies are made; target domain 3 uses held-out official-test originals. Defaults give 6,000/1,500/1,000 train/val/test rows. `manifest.csv` and its JSON retain image grouping, domains, counts and seed. These are generated appearance domains of real images. Use `--offline --archive data/cifar10.tgz` for an existing archive. Other licensed datasets can use `prepare.py --root data/domains --target heldout --output data/manifest.csv --seed 7` with `root/domain/class/image` folders; read the target ID from its mapping JSON.

```bash
python prepare_cifar.py --plan
python prepare_cifar.py --output data/cifar-domains --seed 7
```

## Training and inference

```bash
responsedg train --manifest data/cifar-domains/manifest.csv --target-domain 3 --output runs/full --variant full --epochs 30 --rank 32 --batch-size 32 --seed 7 --lr 0.001 --gate-threshold 0.1 --response-weight 1 --signature-weight 0.1
```

## Evaluation

```bash
responsedg evaluate --manifest data/cifar-domains/manifest.csv --checkpoint runs/full/model.pt --output runs/full/evaluation --batch-size 32
```

## Ablations

Supported controls: `erm`, `no_gate`, `no_factorization`, `no_response`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
responsedg train --manifest data/cifar-domains/manifest.csv --target-domain 3 --output outputs/erm --variant erm --epochs 30 --rank 32 --batch-size 32 --seed 7 --lr 0.001 --gate-threshold 0.1 --response-weight 1 --signature-weight 0.1
```

## Outputs

Preparation writes images and manifest CSV/JSON. Training writes `model.pt`, `history.json` and `learning.png`. Evaluation writes `responses.npz`, `metrics.json` and `gate.png` under its output directory.

Source inspection: Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
