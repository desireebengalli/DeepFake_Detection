import os, argparse, random, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm.auto import tqdm

from PIL import Image, ImageFilter, ImageOps
import numpy as np
from torchvision import transforms

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

def build_dst_inplace(src_path: Path, aug_idx: int) -> Path:
    base = src_path.stem  # include _ctx
    return src_path.parent / f"{base}_aug{aug_idx:02d}.jpg"

def build_dst_outroot(out_root: Path, rel_parent: Path, base_name: str, aug_idx: int) -> Path:
    dst_dir = out_root / rel_parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / f"{base_name}_aug{aug_idx:02d}_ctx.jpg"

# Augmentation blocks 
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
    ])

def augment_and_save(src_path: Path, out_root: Path, variants: int, jpg_quality: int,
                     input_roots: List[Path], strip_prefix: str, aug):
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

def parse_args():
    ap = argparse.ArgumentParser(description="Augment ONLY *_ctx.jpg (CPU).")
    ap.add_argument("--in-roots", nargs="+", required=True,
                    help="Una o più radici da cui cercare ricorsivamente *_ctx.jpg.")
    ap.add_argument("--out-root", default="", help="Root di output (ignorata se --in-place).")
    ap.add_argument("--in-place", action="store_true",
                    help="Scrive i file augmented nella STESSA cartella del sorgente.")
    ap.add_argument("--variants-per-image", type=int, default=2)
    ap.add_argument("--jpg-quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    imgs = scan_ctx_images(args.in_roots)
    print(f"[INFO] Found {len(imgs)} *_ctx images in {len(args.in_roots)} root(s).")
    if not imgs: return

    out_root = Path(args.out_root).resolve() if args.out_root else None
    if not args.in_place:
        if out_root is None:
            raise SystemExit("--out-root è richiesto se non usi --in-place.")
        out_root.mkdir(parents=True, exist_ok=True)

    aug = make_aug_pipeline()
    total_ops = len(imgs) * args.variants_per_image
    pbar = tqdm(total=total_ops, unit="img", disable=not args.progress)

    def worker(src: Path):
        try:
            from PIL import Image
            img = Image.open(src).convert("RGB")
        except Exception:
            return 0
        base = src.stem  # include _ctx
        written = 0
        for k in range(args.variants_per_image):
            if args.in_place:
                dst = build_dst_inplace(src, k)
            else:
                first_root = Path(args.in_roots[0]).resolve()
                try:
                    rel_parent = src.parent.resolve().relative_to(first_root)
                except Exception:
                    rel_parent = Path(src.parent.name)
                dst = build_dst_outroot(out_root, rel_parent, base, k)
            if dst.exists() and args.skip_existing:
                continue
            out_img = aug(img)
            out_img.save(dst, format="JPEG", quality=int(args.jpg_quality))
            written += 1
        return written

    total_written = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        for w in as_completed(pool.submit(worker, p) for p in imgs):
            total_written += w.result()
            pbar.update(args.variants_per_image)
    pbar.close()
    print(f"[SUMMARY] written: {total_written} | in_place={args.in_place} | out_root={out_root if out_root else '-'}")

if __name__ == "__main__":
    main()