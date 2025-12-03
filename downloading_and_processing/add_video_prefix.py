import os
from pathlib import Path

# Root principale dei frame preprocessati
BASE = Path("/home/giadapoloni/C_preprocessed_frames/C_fake")

IMG_EXTS = [".jpg",]

def add_prefix_in_dir(video_dir: Path):
    parent = video_dir.parent.name
    video_id = video_dir.name
    prefix = f"{parent}_{video_id}_"

    for f in video_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS and not f.name.startswith(prefix):
            new_name = f"{prefix}{f.name}"
            new_path = f.with_name(new_name)
            f.rename(new_path)
            print(f"Renamed: {f.name} -> {new_name}")

def main():
    # scorri tutte le sottocartelle di preprocessed_frames
    for root, dirs, files in os.walk(BASE):
        for d in dirs:
            video_dir = Path(root) / d
            # se dentro ci sono immagini, applica la rinomina
            if any(f.is_file() and f.suffix.lower() in IMG_EXTS for f in video_dir.iterdir()):
                add_prefix_in_dir(video_dir)

if __name__ == "__main__":
    main()