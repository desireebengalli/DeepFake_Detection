import os, random, shutil

# Percorsi
source_dir = "/home/giadapoloni/gdrive/CV Group Project/Celeb-DF-v2/Celeb-synthesis"
target_dir = "/home/giadapoloni/dataset"

# Crea la cartella di destinazione se non esiste
os.makedirs(target_dir, exist_ok=True)

# Seleziona i file video
video_exts = ('.mp4', '.avi', '.mov', '.mkv')
files = [f for f in os.listdir(source_dir) if f.lower().endswith(video_exts)]

# Prendi il 50% dei file in modo casuale
selected = random.sample(files, len(files)//2)

# Copia i file selezionati
for i, f in enumerate(selected, 1):
    shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))
    if i % 10 == 0 or i == len(selected):
        print(f"Copiati {i}/{len(selected)} file...")

print(f"\n✅ Copiati {len(selected)} video su {target_dir}")
