# Plotting a 3D PCA example of CLIP video embeddings (REAL vs FAKE)
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.decomposition import PCA

import clip

# Model
MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"

SAVE_PATH = "/home/giadapoloni/results2/CLIP5_ln_bias_linear_notext/clip5_linear_ln_bias_notext.pt"
FIG_SAVE_PATH = "/home/giadapoloni/visualizations/3D_example"

IMG_EXTS = {".jpg"} 


class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)


def collect_test_items(real_dir, fake_dir):
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if rroot.exists():
            for p in sorted(rroot.rglob("*")):
                if (
                    p.is_file() and 
                    p.suffix.lower() in IMG_EXTS
                ):
                    items.append({"path": p, "label": label})
    return items


def build_video_index():
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    video_frames = defaultdict(list)
    video_labels = {}

    for rec in test_items:
        p = Path(rec["path"])
        vid = str(p.parent)
        video_frames[vid].append(p)
        video_labels[vid] = rec["label"]

    real_vids = [vid for vid, lab in video_labels.items() if lab == 0]
    fake_vids = [vid for vid, lab in video_labels.items() if lab == 1]

    return real_vids, fake_vids, video_frames, video_labels


def load_clip_and_checkpoint():
    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    clip_model.float()

    embed_dim = clip_model.visual.output_dim
    head = LinearHead(embed_dim, 2).to(DEVICE)

    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])
    clip_model.visual.load_state_dict(ckpt["visual"])

    clip_model.eval()
    head.eval()

    return clip_model, head, preprocess


@torch.no_grad()
def compute_frame_embeddings_for_video_no_aug(
    clip_model,
    preprocess,
    device,
    frame_paths,
):
    frame_paths = [p for p in sorted(frame_paths) if "aug" not in p.name.lower()]

    if len(frame_paths) == 0:
        return None, []

    imgs = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        img = preprocess(img)
        imgs.append(img)

    imgs = torch.stack(imgs, dim=0).to(device)

    with torch.no_grad():
        z = F.normalize(clip_model.encode_image(imgs), dim=-1)

    return z.cpu().numpy(), frame_paths



@torch.no_grad()
def visualize_one_real_one_fake_video_3d(
    clip_model,
    preprocess,
    device,
    save_path=None,
):
    # Video's index
    real_vids, fake_vids, video_frames, _ = build_video_index()

    if len(real_vids) == 0 or len(fake_vids) == 0:
        print("Not enough real/fake videos found.")
        return

    real_vid = random.choice(real_vids)
    fake_vid = random.choice(fake_vids)

    print("Video REAL chosen:", real_vid)
    print("Video FAKE chosen:", fake_vid)

    # Computing embeddings
    emb_real, frames_real = compute_frame_embeddings_for_video_no_aug(
        clip_model, preprocess, device, video_frames[real_vid]
    )
    emb_fake, frames_fake = compute_frame_embeddings_for_video_no_aug(
        clip_model, preprocess, device, video_frames[fake_vid]
    )

    if emb_real is None or emb_fake is None:
        print("Error: no valid frames found.")
        return

    # PCA on frames
    X = np.concatenate([emb_real, emb_fake], axis=0)
    labels = np.array([0] * len(emb_real) + [1] * len(emb_fake))

    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    X_real_3d = X_3d[labels == 0]
    X_fake_3d = X_3d[labels == 1]

    # 3D plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X_real_3d[:, 0],
        X_real_3d[:, 1],
        X_real_3d[:, 2],
        marker="o",
        c="tab:blue",
        s=40,
        alpha=0.8,
        label=f"Frame REAL ({len(X_real_3d)})",
    )

    ax.scatter(
        X_fake_3d[:, 0],
        X_fake_3d[:, 1],
        X_fake_3d[:, 2],
        marker="o",
        c="tab:red",
        s=40,
        alpha=0.8,
        label=f"Frame FAKE ({len(X_fake_3d)})",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA of frames: REAL vs FAKE")

    ax.legend(loc="best")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved in: {save_path}")

    plt.show()



def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    clip_model, head, preprocess = load_clip_and_checkpoint()

    visualize_one_real_one_fake_video_3d(
        clip_model=clip_model,
        preprocess=preprocess,
        device=DEVICE,
        save_path=FIG_SAVE_PATH,
    )


if __name__ == "__main__":
    main()
