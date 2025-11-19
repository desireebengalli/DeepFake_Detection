#!/usr/bin/env python3
"""
Training CLIP (frame-only) for binary classification (Real vs Fake) where each folder directly contains frames:
- Train:  root with folders Real/ and Fake/ (frames directly inside, no video subfolders)
- Val:    root with folders Real_val/ and Fake_val/ (or Real/ and Fake/)
- Test:   root with folders Real_test/ and Fake_test/ (or Real/ and Fake/)

No video-level structure — each frame is treated independently but we can still compute global AUC.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    import open_clip
except Exception as e:
    raise RuntimeError("Install open_clip_torch: pip install open_clip_torch") from e

CLASS_NAMES = ["Real", "Fake"]
CLASS_TO_BIN = {"Real": 0, "Fake": 1}
VALID_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _norm_class_name(name: str) -> str:
    n = name.lower()
    if n.startswith("real"):
        return "Real"
    if n.startswith("fake"):
        return "Fake"
    return ""


def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        cls = _norm_class_name(sub.name)
        if cls == "":
            continue
        for img_path in sub.iterdir():
            if img_path.suffix.lower() in VALID_IMG_EXT:
                rows.append([
                    img_path.stem,
                    str(img_path.resolve()),
                    CLASS_TO_BIN[cls],
                    cls,
                ])
    if not rows:
        raise RuntimeError(f"No images found in {root}. Expect Real/ and Fake/ folders.")
    df = pd.DataFrame(rows, columns=["frame_id", "path", "label_binary", "label_multi"])
    return df


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess=None, label_mode: str = "binary"):
        self.df = df
        self.label_mode = label_mode
        if preprocess is None:
            _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.preprocess(img)
        y = int(row["label_binary"])
        return {"image": x, "label": torch.tensor(y, dtype=torch.long)}


class TokenExtractor(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, cls.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.visual.ln_post(x)
        return x


class AttnPool(nn.Module):
    def __init__(self, dim=1024, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, tokens):
        a = torch.tanh(self.fc1(tokens))
        a = self.fc2(a).squeeze(-1)
        w = torch.softmax(a, dim=1)
        z = (w.unsqueeze(-1) * tokens).sum(1)
        return z


class ClipFrameProbe(nn.Module):
    def __init__(self, arch="ViT-L-14", pretrained="openai", num_classes=2):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.tokenizer = TokenExtractor(self.clip.visual)
        self.dim = self.clip.visual.width
        self.pool = AttnPool(dim=self.dim)
        self.head = nn.Linear(self.dim, num_classes)

    def forward(self, images):
        tokens = self.tokenizer(images)
        z = self.pool(tokens)
        return self.head(z)


def train_one_epoch(model, loader, optim, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    tot, corr = 0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        with torch.no_grad():
            pred = logits.argmax(1)
            corr += (pred == y).sum().item()
            tot += y.size(0)
    return corr / max(1, tot)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    softmax = nn.Softmax(dim=1)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        logits = model(x)
        prob = softmax(logits)[:, 1].cpu().numpy()
        y_true.extend(y.tolist())
        y_prob.extend(prob.tolist())
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--workdir", default="runs/clip_attnpool_real_fake")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building manifests...")
    train_df = build_manifest(Path(args.train_root))
    val_df = build_manifest(Path(args.val_root))
    test_df = build_manifest(Path(args.test_root))

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    train_ds = FrameDataset(train_df, preprocess)
    val_ds = FrameDataset(val_df, preprocess)
    test_ds = FrameDataset(test_df, preprocess)

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = ClipFrameProbe().to(args.device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    best_auc = -1.0
    for ep in range(1, args.epochs + 1):
        acc = train_one_epoch(model, train_ld, optim, args.device)
        auc_val = evaluate(model, val_ld, args.device)
        print(f"[Ep {ep}] acc={acc:.3f} AUC_val={auc_val:.3f}")
        if auc_val > best_auc:
            best_auc = auc_val
            torch.save(model.state_dict(), workdir / "best.pt")

    print("[INFO] Testing best model...")
    model.load_state_dict(torch.load(workdir / "best.pt", map_location=args.device))
    auc_test = evaluate(model, test_ld, args.device)
    print(f"Test AUC={auc_test:.3f}")

if __name__ == "__main__":
    main()
