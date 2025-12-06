import os, random, shutil

source_dir = "/home/giadapoloni/gdrive/CV Group Project/Celeb-DF-v2/Celeb-synthesis"
target_dir = "/home/giadapoloni/dataset"

os.makedirs(target_dir, exist_ok=True)

# select video files
video_exts = ('.mp4', '.avi', '.mov', '.mkv')
files = [f for f in os.listdir(source_dir) if f.lower().endswith(video_exts)]

# take 50% of files randomly
selected = random.sample(files, len(files)//2)

for i, f in enumerate(selected, 1):
    shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))
    if i % 10 == 0 or i == len(selected):
        print(f"Copied {i}/{len(selected)} file...")

print(f"\n Copied {len(selected)} videos on {target_dir}")
