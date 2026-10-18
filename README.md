# ResponseDG

Implementation of **ResponseDG: Mechanism-Preserving Domain Generalization via Intervention-Response Factorization**.

ResponseDG measures representation displacements caused by controlled interventions, screens pairs with a semantic gate, splits accepted responses into orthogonal semantic/nuisance subspaces, and combines the original representation with a semantic-response signature for classification.

## Install

Use Python 3.11+ in an activated virtual environment:

```bash
python -m pip install .
```

## Prepare a held-out domain

Obtain PACS, VLCS, Office-Home, Terra Incognita, DomainNet or NICO++ following the dataset license. Arrange images as `root/domain/class/image`. Domain and class names are mapped to integer IDs consistently across splits.

```bash
python prepare.py --root data/PACS --target sketch --output data/pacs-sketch.csv --seed 7
```

The selected target domain becomes test data. Source images are divided into 80% training and 20% validation within their original domain folders. The adjacent JSON file records the domain/class mapping and the target domain's integer ID. When a benchmark supplies official split lists or shared-image identities, build the CSV from those lists instead of using the folder splitter.

The CSV fields are `path,label,domain,split`. Optional `context_path` points to a class-preserving context-swap image. Optional `pair_path,pair_label` provide a separately annotated intervention pair, including identity-changing interventions for evaluating semantic-screening errors. Generate swaps using object masks and a documented context protocol, retaining the original class label. These optional evaluator pairs do not enter training.

## Train and evaluate

```bash
responsedg train --manifest data/pacs-sketch.csv --target-domain 3 --output outputs/full --epochs 30
responsedg evaluate --manifest data/pacs-sketch.csv --checkpoint outputs/full/model.pt --output outputs/evaluation
```

Replace `3` with `target_domain` from the preparation JSON. Repeat once for each held-out domain and aggregate the resulting domain accuracies. Source validation accuracy selects checkpoints. Neither target labels nor intervention-pair evaluation labels select model parameters.

Training constructs four deterministic intervention views: color attenuation, contrast change, Gaussian blur, and a small local rotation. The semantic gate multiplies base-head prediction agreement (`JS < --gate-threshold`) by spatial feature correspondence. Accepted response displacements are split using complementary projectors from an orthonormal QR basis.

| Parameter | Meaning |
| --- | --- |
| `--rank 32` | Semantic response-subspace dimension out of 64 |
| `--gate-threshold 0.1` | Maximum Jensen–Shannon divergence admitted by the gate |
| `--response-weight 1` | Cross-domain semantic response and nuisance-logit penalty |
| `--signature-weight 0.1` | Consistency after exchanging same-class signatures |
| `--batch-size 32` | Number of images per optimization step |
| `--lr 0.001` | Adam learning rate |

Cross-domain response consistency is computed for class/domain groups present in a batch. Use mixed-domain batches and a sufficient batch size when increasing the class count. The nuisance penalty acts on the base-representation columns of the final classifier.

## Comparisons

```bash
responsedg train --manifest data/pacs-sketch.csv --target-domain 3 --output outputs/erm --variant erm
responsedg train --manifest data/pacs-sketch.csv --target-domain 3 --output outputs/no-gate --variant no_gate
responsedg train --manifest data/pacs-sketch.csv --target-domain 3 --output outputs/no-factorization --variant no_factorization
responsedg train --manifest data/pacs-sketch.csv --target-domain 3 --output outputs/no-response --variant no_response
```

ERM uses the same encoder and classification head with a zero response signature. The other variants remove semantic screening, orthogonal factorization, or response regularization. Keep data manifests, optimization settings and seeds identical. Use multiple paired seeds to assess run-to-run variation.

## Outputs

Training writes `model.pt`, `history.json` and `learning.png`. Evaluation writes `metrics.json`, `responses.npz`, and the gate-weight histogram `gate.png`.

The evaluator reports recognition accuracy, worst supplied test-domain accuracy, 15-bin ECE, gate acceptance, end-to-end images per second, cross-domain semantic response variance where multiple domains are supplied, and linear class/domain probe accuracy in each response subspace. Probes are fitted on source training representations and scored on source validation representations. Smaller response variance indicates greater response stability.

Context-swap accuracy drop and counterfactual prediction consistency are computed when their image paths are supplied. Semantic-preservation error is the fraction of admitted annotated pairs whose `pair_label` differs from the original label. The images are screened before accessing pair labels. Metrics requiring missing pairs or multiple test domains remain absent or `null`.

## Implementation choices

The encoder is a small three-stage convolutional network trained from scratch. It keeps the response mechanism inspectable and avoids pretrained-model downloads. The intervention bank uses color, contrast, blur, and geometry; object-mask foreground/background interventions can be supplied as evaluation pairs. Spatial correspondence is mean cosine similarity of local encoder features. Orthogonality is exact up to QR numerical precision, with nuisance response equal to the residual complement. The source-validation probes describe response information in the trained representation; they are separate from target-domain model selection.

## Selected reproducible setting

Use the included three-stage compact CNN with 64-channel features, QR response
rank 32 and the deterministic color/contrast/blur/8-degree rotation bank on
128-by-128 RGB inputs. The public input setting is CIFAR-10 with three fixed
source appearance domains (clean, warm channel balance, low contrast) and a
held-out cold/blur domain. Domain transforms do not read labels.

```bash
python -m pip install -e .
python prepare_cifar.py --plan
python prepare_cifar.py --output data/cifar-domains --seed 7
responsedg train --manifest data/cifar-domains/manifest.csv --target-domain 3 --output runs/full --variant full --epochs 30 --rank 32 --batch-size 32 --seed 7 --lr 0.001 --gate-threshold 0.1 --response-weight 1 --signature-weight 0.1
responsedg evaluate --manifest data/cifar-domains/manifest.csv --checkpoint runs/full/model.pt --output runs/full/evaluation --batch-size 32
```

Acquisition uses the [fast.ai public PNG mirror](https://registry.opendata.aws/fast-ai-imageclas/)
of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The archive is read
without extracting its paths or loading pickles. For a local copy use
`python prepare_cifar.py --offline --archive data/cifar10.tgz`.
The default subset contains 200 source-training and 50 source-validation
originals per class, each rendered in all three source domains, and 100 official
test originals per class rendered only in target domain 3. This gives
6,000/1,500/1,000 train/val/test rows. The role is assigned to each original
before making source-domain copies, so those copies cannot cross train and val.
Native images remain 32px; the existing loader resizes inputs to 128px.
`manifest.json` records the source, domains, counts and seed; `image_id` and
`original_split` in the CSV retain the grouping and source roles. Only source
validation selects the checkpoint. Cite Krizhevsky's 2009 CIFAR report when
using the data and retain the dataset's original attribution.

For custom domain folders, the existing adapter remains available:

```bash
python prepare.py --root data/domains --target heldout --output data/manifest.csv --seed 7
```

Use its recorded `target_domain` in training. Directory layout is
`data/domains/domain/class/image`. Optional context_path/pair_path and pair_label
support additional annotated interventions. The public setting's classification,
response and calibration endpoints run without those optional pairs.
