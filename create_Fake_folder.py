import os
import shutil

# Percorsi sorgente
src_dirs = [
    "/home/default/DeepFake_Detection/preprocessed_512jpg/Deepfakes",
    "/home/default/DeepFake_Detection/preprocessed_512jpg/Face2Face",
    "/home/default/DeepFake_Detection/preprocessed_512jpg/FaceSwap",
    "/home/default/DeepFake_Detection/preprocessed_512jpg/NeuralTextures",
    "/home/giadapoloni/DeepFake_Detection/preprocessed_augmentation_512jpg/Deepfakes",
    "/home/giadapoloni/DeepFake_Detection/preprocessed_augmentation_512jpg/Face2Face",
    "/home/giadapoloni/DeepFake_Detection/preprocessed_augmentation_512jpg/FaceSwap",
    "/home/giadapoloni/DeepFake_Detection/preprocessed_augmentation_512jpg/NeuralTextures"
]

# Cartella di destinazione
dst_dir = "/home/desireebengalli/DeepFake_Detection/Fake"

# Crea la cartella di destinazione se non esiste
os.makedirs(dst_dir, exist_ok=True)

# Estensioni di immagini accettate
valid_exts = (".jpg")

count = 0
for src in src_dirs:
    for root, _, files in os.walk(src):
        for file in files:
            if file.lower().endswith(valid_exts):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)

                # Evita conflitti di nomi duplicati
                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(file)
                    dst_path = os.path.join(dst_dir, f"{name}_{count}{ext}")

                shutil.copy2(src_path, dst_path)
                count += 1

print(f"Copiate {count} immagini nella cartella {dst_dir}")
