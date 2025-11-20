#!/usr/bin/env python3
import os
import re
import glob
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Config & Paths
# =========================

# Radice del dataset (usa il tuo path)
ROOT = "/home/giadapoloni/dataset"

# Directory da processare (assolute) con label associata
# Usa {ROOT} come base.
TARGET_DIRS: List[Tuple[str, str]] = [
    (f"{ROOT}/original_sequences/youtube/c23/videos",            "original"),
    (f"{ROOT}/manipulated_sequences/Deepfakes/c23/videos",       "Deepfakes"),
    (f"{ROOT}/manipulated_sequences/Face2Face/c23/videos",       "Face2Face"),
    (f"{ROOT}/manipulated_sequences/FaceSwap/c23/videos",        "FaceSwap"),
    (f"{ROOT}/manipulated_sequences/NeuralTextures/c23/videos",  "NeuralTextures"),
]



# Dove scrivere i frame
# OUT/real_ff/<video_id>/001.jpg
# OUT/fake_ff/<method>/<video_id>/001.jpg
OUT             = Path("/home/giadapoloni/extracted_frames")

FRAMES_TARGET   = 32
MIN_FRAMES_OK   = 16

# Azioni su file
DRY_RUN = False
USE_TRASH       = False                     # True → sposta i video in TRASH_DIR invece di cancellarli
TRASH_DIR       = Path("trash_videos")

# Parallelismo
WORKERS = min(4, (os.cpu_count() or 4))
FFMPEG_THREADS_PER_PROC = 1

# Formato immagini
IMG_EXT        = ".jpg"      # ".jpg" o ".webp"
JPEG_QSCALE    = "3"         # 2–5 tipico (più alto = più compresso)
WEBP_QUALITY   = "85"

# Accelerazione HW (decodifica)
USE_HWACCEL       = False
HWACCEL_BACKEND   = "None"   # "cuda", "vaapi", "auto", oppure "None"

# Estensioni video considerate (tuple vera, non stringa)
VIDEO_EXTS = (".mp4",)

# =========================
# Helpers
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_videos_in_dir(dir_path: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(dir_path):
        print(f"[WARN] Directory non trovata: {dir_path}")
        return paths
    for ext in VIDEO_EXTS:
        pattern = os.path.join(dir_path, "", f"*{ext}")
        paths += glob.glob(pattern, recursive=True)
    return sorted(paths)

def vid_id_from_path(p: str) -> str:
    return Path(p).stem

def slugify(label: str) -> str:
    """minuscolo + solo a-z0-9- (es. 'Face2Face' -> 'face2face')"""
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "unknown"

def safe_remove(video_path: str):
    if DRY_RUN:
        print(f"[DRY_RUN] Would remove: {video_path}")
        return
    if USE_TRASH:
        ensure_dir(TRASH_DIR)
        dst = TRASH_DIR / Path(video_path).name
        i = 1
        base = dst.stem
        ext = dst.suffix
        while dst.exists():
            dst = TRASH_DIR / f"{base}_{i}{ext}"
            i += 1
        shutil.move(video_path, dst)
        print(f"[MOVED] {video_path} -> {dst}")
    else:
        os.remove(video_path)
        print(f"[REMOVED] {video_path}")

def out_subdir_for(label: str) -> Path:
    """
    Mappa le label in cartelle di output:
    - 'original' → OUT/real_ff
    - altrimenti → OUT/fake_ff/<slug(label)>
    """
    if label.lower() == "original":
        return OUT / "real_ff"
    return OUT / "fake_ff" / slugify(label)

# =========================
# FFmpeg utils
# =========================

def video_duration(video: str) -> Optional[float]:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", video
        ], stderr=subprocess.STDOUT).decode().strip()
        d = float(out)
        return d if d > 0 else None
    except Exception:
        return None

def extract_uniform_n(video: str, dst_dir: Path, n: int) -> int:
    """Estrae ~n frame uniformi, li porta a 512x512 RGB e salva come JPG/WebP."""
    ensure_dir(dst_dir)
    dur = video_duration(video)
    fps = 1 if not dur or dur <= 0 else max(n / dur, 1e-3)

    out_pattern = str(dst_dir / ("%03d" + IMG_EXT))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin"]

    vf_chain_prefix = ""
    if USE_HWACCEL and HWACCEL_BACKEND and HWACCEL_BACKEND.lower() != "none":
        if HWACCEL_BACKEND == "cuda":
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf_chain_prefix = "hwdownload,format=nv12,"
        elif HWACCEL_BACKEND == "vaapi":
            cmd += ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128", "-hwaccel_output_format", "vaapi"]
            vf_chain_prefix = "hwdownload,format=nv12,"
        elif HWACCEL_BACKEND == "auto":
            cmd += ["-hwaccel", "auto"]
            vf_chain_prefix = ""

    # pipeline: sampling uniforme -> resize mantenendo AR -> pad a 512 -> converti in RGB
    vf_chain = (
        f"{vf_chain_prefix}"
        f"fps={fps},"
        f"scale=512:512:force_original_aspect_ratio=decrease,"
        f"pad=512:512:(ow-iw)/2:(oh-ih)/2,"
        f"format=rgb24"
    )

    cmd += [
        "-i", video,
        "-map", "v:0", "-an", "-sn",
        "-vf", vf_chain,
        "-vsync", "vfr",
        "-frames:v", str(n),
    ]

    # qualità in base al formato di output
    if IMG_EXT.lower() in (".jpg", ".jpeg"):
        cmd += ["-qscale:v", JPEG_QSCALE]
    elif IMG_EXT.lower() == ".webp":
        cmd += ["-quality", WEBP_QUALITY]

    # limita i thread interni se stai parallelizzando più processi ffmpeg
    cmd += ["-threads", str(FFMPEG_THREADS_PER_PROC), out_pattern]

    subprocess.run(cmd, check=True)
    return sum(1 for _ in dst_dir.glob(f"*{IMG_EXT}"))

# =========================
# Processing
# =========================

def process_dir(dir_path: str, label: str):
    videos = list_videos_in_dir(dir_path)
    base_out = out_subdir_for(label)
    ensure_dir(base_out)
    stats = {"ok": 0, "skip_existing": 0, "auto_deleted_existing": 0, "warn": 0, "err": 0}

    def one(vpath: str):
        dst = base_out / vid_id_from_path(vpath)
        try:
            existing = sum(1 for _ in dst.glob(f"*{IMG_EXT}")) if dst.exists() else 0

            if existing >= MIN_FRAMES_OK:
                safe_remove(vpath)
                return ("auto_deleted_existing", vpath)

            if existing > 0:
                for p in dst.glob(f"*{IMG_EXT}"):
                    try:
                        p.unlink()
                    except Exception:
                        pass

            wrote = extract_uniform_n(vpath, dst, FRAMES_TARGET)
            if wrote >= MIN_FRAMES_OK:
                safe_remove(vpath)
                return ("ok", vpath)
            else:
                print(f"[WARN] {vpath}: wrote only {wrote} frames (<{MIN_FRAMES_OK}), keeping video.")
                return ("warn", vpath)

        except subprocess.CalledProcessError:
            print(f"[ERROR] ffmpeg failed on {vpath}, keeping video.")
            return ("err", vpath)
        except Exception as e:
            print(f"[ERROR] {vpath}: {e}, keeping video.")
            return ("err", vpath)

    print(f"{label}: trovati {len(videos)} video in {dir_path}")
    if not videos:
        return

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(one, vv) for vv in videos]
        for fut in as_completed(futures):
            status, v = fut.result()
            if status not in stats:
                stats[status] = 0
            stats[status] += 1
            print(f"[{label}] {status}: {v}")

    total = len(videos)
    print(f"[SUMMARY {label}] total={total} " + " ".join(f"{k}={v}" for k, v in stats.items()))

# =========================
# Main
# =========================

def main():
    ensure_dir(OUT)
    for dir_path, label in TARGET_DIRS:
        process_dir(dir_path, label)

    ext = IMG_EXT.lstrip(".")
    print("Done. Struttura output:")
    print(f"- {OUT}/real_ff/<video_id>/001..{ext}")
    print(f"- {OUT}/fake_ff/<method>/<video_id>/001..{ext}")
    if DRY_RUN:
        print("Note: DRY_RUN=True → nessun video è stato cancellato o spostato nel cestino.")

if __name__ == "__main__":
    main()
