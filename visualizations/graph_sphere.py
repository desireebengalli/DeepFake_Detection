# 3D Visualisation of the CLIP embeddings (REAL vs FAKE)
# Using chekpoints from the CLIP5_ln_bias_linear_notext experiment and Celeb-test dataset.

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import clip
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/16"

# Checkpoint of the model
SAVE_PATH = "/home/giadapoloni/results2/CLIP5_ln_bias_linear_notext/clip5_linear_ln_bias_notext.pt"

# Celeb folders 
TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

IMG_EXTS = {".jpg"}

class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x):
        return self.fc(x)

def collect_test_items(real_dir, fake_dir):
    """
    Gather photos from real and fake directories.
    """
    items = []

    # REAL videos
    real_videos = [d for d in Path(real_dir).iterdir() if d.is_dir()]
    if len(real_videos) == 0:
        raise ValueError("No real video directories found.")
    real_videos = np.random.choice(real_videos, size=min(500, len(real_videos)), replace=False)

    # FAKE videos
    fake_videos = [d for d in Path(fake_dir).iterdir() if d.is_dir()]
    if len(fake_videos) == 0:
        raise ValueError("No fake video directories found.")
    fake_videos = np.random.choice(fake_videos, size=min(500, len(fake_videos)), replace=False)

    # Collect frames for REAL
    for vid in real_videos:
        for p in sorted(vid.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": 0, "video_id": "real_" + vid.name})

    # Collect frames for FAKE
    for vid in fake_videos:
        for p in sorted(vid.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": 1, "video_id": "fake_" + vid.name})

    return items

class TestFrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.preprocess(img)
        return img, rec["label"], rec["video_id"]

def build_test_loader(preprocess, batch_size=64, num_workers=4):
    """
    Build a DataLoader for (Celeb-test).
    """
    items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    assert len(items) > 0, "No image found"

    ds = TestFrameDataset(items, preprocess)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

@torch.no_grad()
def load_trained_models_for_visualization():
    """
    Loads CLIP model and linear head from checkpoint for visualization.
    """
    clip_model, clip_preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

    # Embedding size (512 per ViT-B/32)
    embed_dim = clip_model.text_projection.shape[1]

    # Initialize the head
    head = LinearHead(embed_dim, n_classes=2).to(DEVICE)

    # Load the checkpoint
    assert os.path.isfile(SAVE_PATH), f"Checkpoint not found: {SAVE_PATH}"
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)

    # Load only visual and head weights
    _ = clip_model.visual.load_state_dict(ckpt["visual"], strict=False)
    head.load_state_dict(ckpt["head"])

    clip_model.eval()
    head.eval()

    for p in clip_model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = False

    print("✓ CLIP model and head loaded from checkpoint for visualization.")
    return clip_model, head, clip_preprocess

@torch.no_grad()
def extract_embeddings_and_labels(clip_model, clip_preprocess):
    """
    Extracting embeddings from vision encoder CLIP (Celeb-test).
    """
    test_loader = build_test_loader(clip_preprocess)

    video_embeds = {}
    video_labels = {}

    pbar = tqdm(test_loader, desc="Embedding extraction")
    for imgs, labels, video_ids in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        z = F.normalize(clip_model.encode_image(imgs), dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        for emb, lab, vid in zip(z, labels_np, video_ids):
            if vid not in video_embeds:
                video_embeds[vid] = []
                video_labels[vid] = int(lab)
            video_embeds[vid].append(emb)

    video_ids_list = list(video_embeds.keys())
    Z_list = []
    Y_list = []

    for vid in video_ids_list:
        emb_stack = np.stack(video_embeds[vid], axis=0)
        Z_list.append(emb_stack.mean(axis=0))  
        Y_list.append(video_labels[vid])

    Z = np.stack(Z_list, axis=0)
    Y = np.array(Y_list)

    print(f"✓ Extracted Embeddings: {Z.shape[0]} samples, dim = {Z.shape[1]}")
    return Z, Y

def project_to_3d_with_pca(Z):
    """
    Projecting 3D embeddings with PCA
    """
    pca = PCA(n_components=3)
    Z3 = pca.fit_transform(Z)
    return Z3, pca

def plot_3d_embeddings(Z3, Y, title="CLIP Deepfake - Embedding 3D (PCA on Celeb-test)"):
    """
    Creatin the graph 3D:
      - REAL (label=0) = blue
      - FAKE (label=1) = red
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    mask_real = (Y == 0)
    mask_fake = (Y == 1)

    # REAL = blue
    ax.scatter(
        Z3[mask_real, 0],
        Z3[mask_real, 1],
        Z3[mask_real, 2],
        s=40,
        alpha=1.0,    
        c="blue",
        marker="o",
        edgecolors="none",
        label="REAL (0)"
    )

    # FAKE = red
    ax.scatter(
        Z3[mask_fake, 0],
        Z3[mask_fake, 1],
        Z3[mask_fake, 2],
        s=40,
        alpha=1.0,    
        c="red",
        marker="o",
        edgecolors="none",
        label="FAKE (1)"
    )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.view_init(elev=20, azim=75)

    plt.tight_layout()

    save_dir = "/home/desireebengalli"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clip_embeddings_3d_2.jpg")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved in: {save_path}")

    plt.show()

# Check models from checkpoint
clip_model_vis, head_vis, clip_preprocess_vis = load_trained_models_for_visualization()

# Extract embeddings and label from test
Z, Y = extract_embeddings_and_labels(
    clip_model_vis,
    clip_preprocess_vis
)

Z3, pca = project_to_3d_with_pca(Z)

plot_3d_embeddings(Z3, Y)