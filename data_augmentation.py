#!/usr/bin/env python3
# data_augmentation.py — Offline *_ctx augmentation (CPU, torchvision/PIL)
# Drop-in replacement: same CLI & output naming as your GPU/Kornia version.

import os, argparse, random, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm.auto import tqdm

from PIL import Image, ImageFilter, ImageOps
import numpy as np
from torchvision import transforms

# -------------------------
# Helpers
# -------------------------
def is_ctx_img(p: Path) -> bool:
    n = p.name.lower()
    return (n.endswith("_ctx.jpg") or n.endswith("_ctx.jpeg")) and ("_aug" not in n)

def scan_ctx_images(roots: List[str]) -> List[Path]:
    imgs = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            print(f"[WARN] Root not found: {r}")
            continue
        # jpg & jpeg
        for p in r.rglob("*.jpg"):
            if is_ctx_img(p): imgs.append(p)
        for p in r.rglob("*.jpeg"):
            if is_ctx_img(p): imgs.append(p)
    return imgs

def compute_rel_parent(src_path: Path, first_root: Path, strip_prefix: str) -> Path:
    """Return parent path of src relative to strip_prefix (if set) or first_root."""
    if strip_prefix:
        try:
            return src_path.parent.resolve().relative_to(Path(strip_prefix).resolve())
        except Exception:
            return Path(src_path.parent.name)
    try:
        return src_path.parent.resolve().relative_to(first_root)
    except Exception:
        return Path(src_path.parent.name)

def build_dst(out_root: Path, rel_parent: Path, base_name: str, aug_idx: int) -> Path:
    # Manteniamo il suffisso _ctx nel basename e aggiungiamo _augXX_ctx.jpg
    dst_dir = out_root / rel_parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{base_name}_aug{aug_idx:02d}_ctx.jpg"

# -------------------------
# Augmentation blocks (PIL/torchvision)
# -------------------------
class RandomJPEGArtifacts:
    """Re-encode to JPEG at random quality to simulate compression artifacts."""
    def __init__(self, p: float = 0.5, qmin: int = 65, qmax: int = 92):
        self.p, self.qmin, self.qmax = p, qmin, qmax
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        q = random.randint(self.qmin, self.qmax)
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class RandomGaussianNoise:
    """Additive Gaussian noise in PIL space (approx. std ~ 0.03 on [0,1])."""
    def __init__(self, p: float = 0.4, std: float = 0.03):
        self.p, self.std = p, std
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.std, arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

def make_aug_pipeline():
    # NOTA: non ridimensioniamo (resta 512x512 come input),
    # riproduciamo flip / affine / jitter / blur / noise / jpeg artifacts.
    pil_affine = transforms.RandomAffine(
        degrees=15, translate=(0.10, 0.10), scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR
    )
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        pil_affine,
        transforms.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.10, hue=0.03),
        transforms.RandomApply([transforms.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(radius=1)))], p=0.3),
        RandomGaussianNoise(p=0.4, std=0.03),
        RandomJPEGArtifacts(p=0.5, qmin=65, qmax=92),
        # restiamo in PIL per salvare direttamente
    ])

# -------------------------
# Worker
# -------------------------
def augment_and_save(src_path: Path, out_root: Path, variants: int, jpg_quality: int,
                     input_roots: List[Path], strip_prefix: str, aug):
    # Determina il root di riferimento per path relativi (usa il primo root)
    first_root = input_roots[0]
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception:
        return 0
    base = src_path.stem  # include _ctx
    rel_parent = compute_rel_parent(src_path, first_root, strip_prefix)
    written = 0
    for k in range(variants):
        dst_path = build_dst(out_root, rel_parent, base, k)
        if dst_path.exists():
            continue
        out_img = aug(img)
        out_img.save(dst_path, format="JPEG", quality=int(jpg_quality))
        written += 1
    return written

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Augment ONLY *_ctx.jpg (CPU, torchvision/PIL). Output names match GPU script.")
    ap.add_argument("--in-roots", nargs="+", required=True,
                    help="One or more roots to recursively search for *_ctx.jpg.")
    ap.add_argument("--out-root", required=True,
                    help="Root where augmented images are written (same hierarchy).")
    ap.add_argument("--variants-per-image", type=int, default=2,
                    help="Number of variants per *_ctx.jpg (default: 2).")
    ap.add_argument("--jpg-quality", type=int, default=92,
                    help="JPEG quality for saving (default: 92).")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="(Kept for compatibility; not used in CPU pipeline).")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 8)),
                    help="Number of CPU threads for encode/I-O (default: 8).")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--strip-prefix", type=str, default="",
                    help="Optional path prefix to strip to build relative outputs.")
    ap.add_argument("--use-gpu", action="store_true",
                    help="Accepted for compatibility; ignored (CPU pipeline).")
    ap.add_argument("--progress", action="store_true",
                    help="Show a live progress bar with ETA.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip files that already exist (resume).")
    return ap.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    in_roots = [Path(p).resolve() for p in args.in_roots]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = scan_ctx_images(args.in_roots)
    print(f"[INFO] Found {len(imgs)} *_ctx images across {len(args.in_roots)} roots.")
    if not imgs:
        return

    if args.use_gpu:
        print("[INFO] --use-gpu specified, but this implementation runs on CPU (torchvision/PIL).")

    aug = make_aug_pipeline()

    total_ops = len(imgs) * args.variants_per_image
    pbar = tqdm(total=total_ops, unit="img", disable=not args.progress)

    total_written = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = []
        for p in imgs:
            futures.append(pool.submit(
                augment_and_save, p, out_root, int(args.variants_per_image), int(args.jpg_quality),
                in_roots, args.strip_prefix, aug
            ))
        for fut in as_completed(futures):
            written = fut.result()
            total_written += written
            # anche gli skipped avanzano la barra
            pbar.update(args.variants_per_image)

    pbar.close()
    print(f"[SUMMARY] source images: {len(imgs)} | variants requested: {args.variants_per_image}")
    print(f"[SUMMARY] files actually written: {total_written} | output: {out_root}")

if __name__ == "__main__":
    main()
