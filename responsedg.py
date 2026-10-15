import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def tensor(image):
    return torch.from_numpy(np.asarray(image.resize((128, 128)), dtype=np.float32).copy() / 255).permute(2, 0, 1)


class DomainImages(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; image = Image.open(row["path"]).convert("RGB")
        variants = [ImageEnhance.Color(image).enhance(.2), ImageEnhance.Contrast(image).enhance(1.5), image.filter(ImageFilter.GaussianBlur(1)), image.rotate(8, resample=Image.Resampling.BILINEAR)]
        return tensor(image), torch.stack([tensor(v) for v in variants]), int(row["label"]), int(row["domain"])


class ResponseDG(nn.Module):
    def __init__(self, classes, rank=32, variant="full"):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())
        self.basis = nn.Parameter(torch.eye(64) + torch.randn(64, 64) * .01)
        self.head = nn.Linear(128, classes)
        self.rank, self.variant = rank, variant

    def forward(self, x, variants, threshold=.1):
        b, k = variants.shape[:2]
        original = self.encoder(x); f = original.mean((-1, -2))
        changed = self.encoder(variants.flatten(0, 1)); v = changed.mean((-1, -2)).reshape(b, k, -1)
        displacement = v - f[:, None]
        q, _ = torch.linalg.qr(self.basis)
        semantic = (displacement @ q[:, :self.rank]) @ q[:, :self.rank].T
        nuisance = displacement - semantic
        if self.variant == "no_factorization":
            semantic, nuisance = displacement, displacement * 0
        zeros = torch.zeros_like(f)
        p = self.head(torch.cat([f, zeros], -1)).softmax(-1)
        pv = self.head(torch.cat([v, torch.zeros_like(v)], -1)).softmax(-1)
        middle = (p[:, None] + pv) / 2
        js = .5 * ((p[:, None] * (p[:, None].clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1) + (pv * (pv.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(-1))
        local = nn.functional.cosine_similarity(original[:, None], changed.reshape(b, k, *original.shape[1:]), dim=2).mean((-1, -2)).clamp(0, 1)
        gate = (js.detach() < threshold).float() * local.detach()
        if self.variant == "no_gate":
            gate = torch.ones_like(gate)
        signature = (semantic * gate[..., None]).sum(1) / gate.sum(1, keepdim=True).clamp_min(1e-6)
        if self.variant == "erm":
            signature = signature * 0
        logits = self.head(torch.cat([f, signature], -1))
        return logits, {"f": f, "semantic": semantic, "nuisance": nuisance, "signature": signature, "gate": gate, "js": js}


def cross_domain_variance(semantic, gate, labels, domains):
    terms = []
    for label in labels.unique():
        means = []
        for domain in domains.unique():
            mask = (labels == label) & (domains == domain)
            if mask.any():
                weight = gate[mask, :, None]
                means.append((semantic[mask] * weight).sum(0) / weight.sum(0).clamp_min(1e-6))
        if len(means) > 1:
            terms.append(torch.stack(means).var(0, unbiased=False).mean())
    return torch.stack(terms).mean() if terms else semantic.sum() * 0


def train(args):
    torch.manual_seed(args.seed)
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    train_rows = [r for r in rows if r["split"] == "train" and int(r["domain"]) != args.target_domain]
    val_rows = [r for r in rows if r["split"] == "val" and int(r["domain"]) != args.target_domain]
    if not train_rows or not val_rows:
        raise ValueError("Source training and validation rows required")
    model = ResponseDG(max(int(r["label"]) for r in rows) + 1, args.rank, args.variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, changed, y, domain in DataLoader(DomainImages(train_rows), batch_size=args.batch_size, shuffle=True):
            logits, state = model(x, changed, args.gate_threshold)
            loss = nn.functional.cross_entropy(logits, y)
            if args.variant != "erm":
                response = cross_domain_variance(state["semantic"], state["gate"], y, domain)
                nuisance_logits = state["nuisance"] @ model.head.weight[:, :64].T
                response = response + (nuisance_logits.square().mean(-1) * state["gate"]).sum() / state["gate"].sum().clamp_min(1)
                swapped = state["signature"].clone()
                for label in y.unique():
                    ids = torch.where(y == label)[0]; swapped[ids] = swapped[ids.roll(1)]
                paired_logits = model.head(torch.cat([state["f"], swapped], -1))
                signature_loss = nn.functional.kl_div(paired_logits.log_softmax(-1), logits.softmax(-1).detach(), reduction="batchmean")
                if args.variant != "no_response":
                    loss = loss + args.response_weight * response
                loss = loss + args.signature_weight * signature_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); correct, n = 0, 0
        with torch.no_grad():
            for x, changed, y, _ in DataLoader(DomainImages(val_rows), batch_size=args.batch_size):
                logits, _ = model(x, changed, args.gate_threshold); correct += int((logits.argmax(-1) == y).sum()); n += len(y)
        accuracy = correct / n; history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_accuracy": accuracy})
        if accuracy > best:
            best = accuracy
            torch.save({"state": model.state_dict(), "classes": model.head.out_features, "rank": args.rank, "variant": args.variant, "threshold": args.gate_threshold, "target": args.target_domain}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    plt.plot([r["loss"] for r in history]); plt.xlabel("Epoch"); plt.ylabel("Training objective"); plt.savefig(output / "learning.png", dpi=180)


def collect(model, rows, threshold, batch_size):
    probabilities, features, semantics, nuisances, gates, labels, domains = [], [], [], [], [], [], []
    with torch.no_grad():
        for x, changed, y, d in DataLoader(DomainImages(rows), batch_size=batch_size):
            logits, state = model(x, changed, threshold)
            probabilities.extend(logits.softmax(-1).numpy()); features.extend(state["f"].numpy()); semantics.extend(state["semantic"].mean(1).numpy()); nuisances.extend(state["nuisance"].mean(1).numpy()); gates.extend(state["gate"].numpy()); labels.extend(y.numpy()); domains.extend(d.numpy())
    return tuple(np.array(v) for v in [probabilities, features, semantics, nuisances, gates, labels, domains])


def evaluate(args):
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ResponseDG(checkpoint["classes"], checkpoint["rank"], checkpoint["variant"]); model.load_state_dict(checkpoint["state"]); model.eval()
    test = [r for r in rows if r["split"] == "test"]
    started = time.perf_counter(); probability, features, semantic, nuisance, gate, y, d = collect(model, test, checkpoint["threshold"], args.batch_size)
    elapsed = time.perf_counter() - started
    prediction = probability.argmax(-1); confidence = probability.max(-1)
    metrics = {"accuracy": accuracy_score(y, prediction), "worst_domain_accuracy": min(accuracy_score(y[d == domain], prediction[d == domain]) for domain in np.unique(d)), "ece": 0., "images_per_second": len(test) / elapsed, "gate_acceptance": float(np.mean(gate > 0))}
    for lo in np.linspace(0, 1, 16)[:-1]:
        mask = (confidence > lo) & (confidence <= lo + 1 / 15)
        if mask.any():
            metrics["ece"] += float(mask.mean() * abs(np.mean(prediction[mask] == y[mask]) - confidence[mask].mean()))
    class_variance = []
    for label in np.unique(y):
        means = [semantic[(y == label) & (d == domain)].mean(0) for domain in np.unique(d) if np.any((y == label) & (d == domain))]
        if len(means) > 1:
            class_variance.append(float(np.var(means, axis=0).mean()))
    metrics["response_stability_variance"] = float(np.mean(class_variance)) if class_variance else None
    for name, key in [("context", "context_path"), ("counterfactual", "pair_path")]:
        ids = [i for i, row in enumerate(test) if row.get(key)]
        if ids:
            altered = [dict(test[i], path=test[i][key]) for i in ids]
            alt_probability, *_ = collect(model, altered, checkpoint["threshold"], args.batch_size)
            alt = alt_probability.argmax(-1)
            if name == "context":
                metrics["context_swap_accuracy_drop"] = float(np.mean(prediction[ids] == y[ids]) - np.mean(alt == y[ids]))
            else:
                metrics["counterfactual_consistency"] = float(np.mean(alt == prediction[ids]))
                with torch.no_grad():
                    admitted, harmful = 0, 0
                    for i in ids:
                        if not test[i].get("pair_label"):
                            continue
                        image = tensor(Image.open(test[i]["path"]).convert("RGB"))[None]
                        pair = tensor(Image.open(test[i][key]).convert("RGB"))[None, None]
                        _, state = model(image, pair, checkpoint["threshold"])
                        if bool(state["gate"][0, 0] > 0):
                            admitted += 1; harmful += int(int(test[i]["pair_label"]) != int(y[i]))
                    metrics["semantic_preservation_error"] = harmful / admitted if admitted else None
    train_rows = [r for r in rows if r["split"] == "train" and int(r["domain"]) != checkpoint["target"]]
    val_rows = [r for r in rows if r["split"] == "val" and int(r["domain"]) != checkpoint["target"]]
    if train_rows and val_rows:
        tr = collect(model, train_rows, checkpoint["threshold"], args.batch_size); va = collect(model, val_rows, checkpoint["threshold"], args.batch_size)
        variances = []
        for label in np.unique(va[5]):
            means = [va[2][(va[5] == label) & (va[6] == domain)].mean(0) for domain in np.unique(va[6]) if np.any((va[5] == label) & (va[6] == domain))]
            if len(means) > 1:
                variances.append(float(np.var(means, axis=0).mean()))
        metrics["source_validation_response_variance"] = float(np.mean(variances)) if variances else None
        for feature_name, pos in [("semantic", 2), ("nuisance", 3)]:
            for target_name, label_pos in [("class", 5), ("domain", 6)]:
                if len(np.unique(tr[label_pos])) > 1:
                    probe = LogisticRegression(max_iter=1000).fit(tr[pos], tr[label_pos]); metrics[f"{feature_name}_{target_name}_probe_accuracy"] = accuracy_score(va[label_pos], probe.predict(va[pos]))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output / "responses.npz", probability=probability, semantic=semantic, nuisance=nuisance, gate=gate, label=y, domain=d)
    plt.hist(gate.ravel(), bins=20); plt.xlabel("Semantic gate weight"); plt.ylabel("Intervention pairs"); plt.savefig(output / "gate.png", dpi=180)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(required=True)
    q = sub.add_parser("train")
    q.add_argument("--manifest", required=True); q.add_argument("--output", required=True); q.add_argument("--target-domain", type=int, required=True)
    q.add_argument("--variant", choices=["full", "erm", "no_gate", "no_factorization", "no_response"], default="full")
    for key, val in [("epochs", 30), ("rank", 32), ("batch-size", 32), ("seed", 7)]:
        q.add_argument("--" + key, type=int, default=val)
    for key, val in [("lr", .001), ("gate-threshold", .1), ("response-weight", 1.), ("signature-weight", .1)]:
        q.add_argument("--" + key, type=float, default=val)
    q.set_defaults(func=train)
    q = sub.add_parser("evaluate"); q.add_argument("--manifest", required=True); q.add_argument("--checkpoint", required=True); q.add_argument("--output", required=True); q.add_argument("--batch-size", type=int, default=32); q.set_defaults(func=evaluate)
    args = p.parse_args()
    if hasattr(args, "rank") and not 0 < args.rank < 64:
        p.error("rank must lie between 1 and 63")
    args.func(args)


if __name__ == "__main__":
    main()
