from pathlib import Path
import argparse
import random
import shutil
import math
import sys

def pick_dirs_for_split(src_dir: Path, percent: float, rng: random.Random):
    """Returns the list of folders selected for the split"""
    if not src_dir.exists():
        raise FileNotFoundError(f"Source not found: {src_dir}")
    video_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
    total = len(video_dirs)
    if total == 0:
        return []
    k = max(1, math.ceil(total * percent)) if percent > 0 else 0
    chosen = rng.sample(video_dirs, k) if k > 0 else []
    return chosen

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def move_or_copy_dir(src: Path, dst_root: Path, mode: str):
    dst = dst_root / src.name
    if dst.exists():
        raise FileExistsError(f"Destination already extists: {dst}")
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError("--mode can't be 'move' or 'copy'")

def main():
    parser = argparse.ArgumentParser(description="Crea /home/giadapoloni/C_validation moving/copying a percentage of videos.")
    parser.add_argument("--root", type=Path, default=Path("/home/giadapoloni/C_preprocessed_frames"), help="Directory that contains C_real and C_fake.")
    parser.add_argument("--percent", type=float, default=0.15, help="Percentage (0..1) of folders to put in validation.")
    parser.add_argument("--mode", choices=["move", "copy"], default="move", help="Move or copy the folders.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random, for reproducibiulity.")
    args = parser.parse_args()

    root = args.root
    c_real = root / "C_real"
    c_fake = root / "C_fake"
    c_val = Path("/home/giadapoloni/C_validation")
    c_val_real = c_val / "C_real"
    c_val_fake = c_val / "C_fake"

    for p in [c_real, c_fake]:
        if not p.exists():
            print(f"ERROR: not found {p}", file=sys.stderr)
            sys.exit(1)

    ensure_dir(c_val_real)
    ensure_dir(c_val_fake)

    rng = random.Random(args.seed)

    chosen_real = pick_dirs_for_split(c_real, args.percent, rng)
    chosen_fake = pick_dirs_for_split(c_fake, args.percent, rng)

    print(f"Root: {root}")
    print(f"Destination validation: {c_val}")
    print(f"Percentage: {args.percent:.2%}")
    print(f"Mode: {args.mode}")
    print(f"Selected REAL: {len(chosen_real)}")
    print(f"Selected FAKE: {len(chosen_fake)}")
    print("-" * 60)

    errors = []
    for d in chosen_real:
        try:
            move_or_copy_dir(d, c_val_real, args.mode)
            print(f"[OK] {d.name} -> {c_val_real}")
        except Exception as e:
            errors.append((d, e))
            print(f"[ERR] {d}: {e}", file=sys.stderr)

    for d in chosen_fake:
        try:
            move_or_copy_dir(d, c_val_fake, args.mode)
            print(f"[OK] {d.name} -> {c_val_fake}")
        except Exception as e:
            errors.append((d, e))
            print(f"[ERR] {d}: {e}", file=sys.stderr)

    print("-" * 60)
    if errors:
        print(f"Completed with errors: {len(errors)} problems encountered.", file=sys.stderr)
        for d, e in errors:
            print(f" - {d}: {e}", file=sys.stderr)
        sys.exit(2)
    else:
        print("Completed wiuthout errors.")

if __name__ == "__main__":
    main()

