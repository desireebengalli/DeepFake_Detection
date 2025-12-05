import os
import shutil

src_dirs = [
    "/home/default/DeepFake_Detection/preprocessed_512jpg/original",
    "/home/giadapoloni/DeepFake_Detection/preprocessed_augmentation_512jpg/original"
]

dst_dir = "/home/desireebengalli/DeepFake_Detection/Real"

os.makedirs(dst_dir, exist_ok=True)

valid_exts = (".jpg")

count = 0
for src in src_dirs:
    for root, _, files in os.walk(src):
        for file in files:
            if file.lower().endswith(valid_exts):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)

                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(file)
                    dst_path = os.path.join(dst_dir, f"{name}_{count}{ext}")

                shutil.copy2(src_path, dst_path)
                count += 1
                
print(f"Copied {count} images to folder {dst_dir}")