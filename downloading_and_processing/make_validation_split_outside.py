#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sposta il 15% (o una percentuale specificata) delle cartelle-video da
/home/giadapoloni/C_preprocessed/{C_real, C_fake}
in una nuova cartella /home/giadapoloni/C_validation,
mantenendo la struttura interna (C_real / C_fake).

Uso tipico:
    python make_validation_split_outside.py \
        --root /home/giadapoloni/C_preprocessed \
        --percent 0.15 \
        --mode move \
        --seed 42
"""

from pathlib import Path
import argparse
import random
import shutil
import math
import sys

def pick_dirs_for_split(src_dir: Path, percent: float, rng: random.Random):
    """Restituisce la lista di cartelle (solo directory) selezionate per lo split."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Sorgente non trovata: {src_dir}")
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
        raise FileExistsError(f"La destinazione esiste già: {dst}")
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError("--mode deve essere 'move' o 'copy'")

def main():
    parser = argparse.ArgumentParser(description="Crea /home/giadapoloni/C_validation spostando/coppiando una percentuale di video.")
    parser.add_argument("--root", type=Path, default=Path("/home/giadapoloni/C_preprocessed_frames"), help="Directory che contiene C_real e C_fake.")
    parser.add_argument("--percent", type=float, default=0.15, help="Percentuale (0..1) di cartelle da mettere in validation.")
    parser.add_argument("--mode", choices=["move", "copy"], default="move", help="Spostare (move) o copiare (copy) le cartelle.")
    parser.add_argument("--seed", type=int, default=None, help="Seed per random, per riproducibilità.")
    args = parser.parse_args()

    root = args.root
    c_real = root / "C_real"
    c_fake = root / "C_fake"
    c_val = Path("/home/giadapoloni/C_validation")
    c_val_real = c_val / "C_real"
    c_val_fake = c_val / "C_fake"

    for p in [c_real, c_fake]:
        if not p.exists():
            print(f"ERRORE: non trovo {p}", file=sys.stderr)
            sys.exit(1)

    ensure_dir(c_val_real)
    ensure_dir(c_val_fake)

    rng = random.Random(args.seed)

    chosen_real = pick_dirs_for_split(c_real, args.percent, rng)
    chosen_fake = pick_dirs_for_split(c_fake, args.percent, rng)

    print(f"Root: {root}")
    print(f"Destinazione validation: {c_val}")
    print(f"Percentuale: {args.percent:.2%}")
    print(f"Modalità: {args.mode}")
    print(f"Selezionati REAL: {len(chosen_real)}")
    print(f"Selezionati FAKE: {len(chosen_fake)}")
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
        print(f"Completato con errori: {len(errors)} problemi riscontrati.", file=sys.stderr)
        for d, e in errors:
            print(f" - {d}: {e}", file=sys.stderr)
        sys.exit(2)
    else:
        print("Completato senza errori.")

if __name__ == "__main__":
    main()

