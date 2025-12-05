# Plotting logit distributions for CLIP video embeddings (REAL vs FAKE)

import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import gaussian_kde 

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/16"

# Checkpoint of the model
SAVE_PATH = "/home/giadapoloni/results2/CLIP5_ln_bias_linear_notext/clip5_linear_ln_bias_notext.pt"

# Celeb folders
TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

IMG_EXTS = {".jpg"}

# Output plot
OUTPUT_PATH = "/home/desireebengalli/clip_logit_distributions.jpg"

# Model
class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x):
        return self.fc(x)

# Dataset
def collect_test_items(real_dir, fake_dir):
    items = []

    # REAL videos
    real_root = Path(real_dir)
    real_videos = [d for d in real_root.iterdir() if d.is_dir()]
    if len(real_videos) == 0:
        raise ValueError("No real video directories found.")
    real_videos = np.random.choice(real_videos,
                                   size=min(500, len(real_videos)),
                                   replace=False)

    # FAKE videos
    fake_root = Path(fake_dir)
    fake_videos = [d for d in fake_root.iterdir() if d.is_dir()]
    if len(fake_videos) == 0:
        raise ValueError("No fake video directories found.")
    fake_videos = np.random.choice(fake_videos,
                                   size=min(500, len(fake_videos)),
                                   replace=False)

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
    items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    assert len(items) > 0, "No image found"

    ds = TestFrameDataset(items, preprocess)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

@torch.no_grad()
def load_trained_models_for_visualization():
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

    embed_dim = clip_model.text_projection.shape[1]

    head = LinearHead(embed_dim, n_classes=2).to(DEVICE)

    assert os.path.isfile(SAVE_PATH), f"Checkpoint not found: {SAVE_PATH}"
    print(f"Loading checkpoint from: {SAVE_PATH}")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)

    _ = clip_model.visual.load_state_dict(ckpt["visual"], strict=False)
    head.load_state_dict(ckpt["head"])

    clip_model.eval()
    head.eval()

    for p in clip_model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = False

    print("✓ CLIP model and head loaded from checkpoint.")
    return clip_model, head, clip_preprocess



@torch.no_grad()
def extract_video_embeddings_and_labels(clip_model, clip_preprocess):
    test_loader = build_test_loader(clip_preprocess)

    video_embeds = {}
    video_labels = {}

    from tqdm import tqdm
    pbar = tqdm(test_loader, desc="Embedding extraction (video-level)")
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

    print(f"Extracted video embeddings: {Z.shape[0]} videos, dim = {Z.shape[1]}")
    num_real = (Y == 0).sum()
    num_fake = (Y == 1).sum()
    print(f"Real videos: {num_real}, Fake videos: {num_fake}")
    return Z, Y

@torch.no_grad()
def plot_logit_distributions(Z, Y, head, output_path):
    """
    Logit distributions (REAL vs FAKE) using the head classifier.
    KDE computed using SciPy
    """  

    z_t = torch.from_numpy(Z).float().to(DEVICE)
    logits = head(z_t).cpu().numpy()     

    scores_fake = logits[:, 1]

    real_scores = scores_fake[Y == 0]
    fake_scores = scores_fake[Y == 1]

    print(f"Real scores: {real_scores.shape}")
    print(f"Fake scores: {fake_scores.shape}")

    plt.figure(figsize=(8, 6))

    plt.hist(real_scores, bins=30, alpha=0.3, density=True,
             label="REAL (0)", color="blue")
    plt.hist(fake_scores, bins=30, alpha=0.3, density=True,
             label="FAKE (1)", color="red")

    kde_real = gaussian_kde(real_scores)
    x_min = min(real_scores.min(), fake_scores.min())
    x_max = max(real_scores.max(), fake_scores.max())
    xs = np.linspace(x_min, x_max, 300)
    plt.plot(xs, kde_real(xs), color="blue", linewidth=2)

    # 3. KDE for FAKE
    kde_fake = gaussian_kde(fake_scores)
    plt.plot(xs, kde_fake(xs), color="red", linewidth=2)

    plt.xlabel("Logit score for class FAKE")
    plt.ylabel("Density")
    plt.title("Logit distributions (video-level)")
    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Logit distribution plot saved to: {output_path}")
    plt.close()

def main():
    clip_model, head, clip_preprocess = load_trained_models_for_visualization()
    Z, Y = extract_video_embeddings_and_labels(clip_model, clip_preprocess)
    plot_logit_distributions(Z, Y, head, OUTPUT_PATH)


if __name__ == "__main__":
    main()