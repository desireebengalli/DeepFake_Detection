# ===========================================
# VISUALIZZAZIONE 3D DEGLI EMBEDDING CLIP (REAL vs FAKE)
# Usando checkpoint già salvato
# ===========================================

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import clip
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # per la proiezione '3d'
from sklearn.decomposition import PCA

# -------------------------------
# CONFIG MINIMA (ADATTA SOLO SE NECESSARIO)
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/16"

# Path al checkpoint salvato durante il training
SAVE_PATH = "/home/giadapoloni/results2/CLIP5_linear_ln_bias_notext/clip5_linear_ln_bias_notext.pt"

# Cartelle Celeb-test (come nel tuo script originale)
TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

# Estensioni immagini
IMG_EXTS = {".jpg"}

# -------------------------------
# CLASSI E FUNZIONI DI SUPPORTO
# -------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x):
        return self.fc(x)

def collect_test_items(real_dir, fake_dir):
    """
    Raccoglie tutte le immagini real (label=0) e fake (label=1)
    dalle cartelle Celeb-test.
    """
    items = []
    rroot = Path(real_dir)
    if rroot.exists():
        for p in sorted(rroot.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": 0})
    froot = Path(fake_dir)
    if froot.exists():
        for p in sorted(froot.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": 1})
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
        return img, rec["label"], str(rec["path"])

def build_test_loader(preprocess, batch_size=64, num_workers=0):
    """
    Costruisce un DataLoader per il test set (Celeb-test).
    """
    items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    assert len(items) > 0, "Nessuna immagine trovata nelle cartelle di test."
    ds = TestFrameDataset(items, preprocess)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

@torch.no_grad()
def load_trained_models_for_visualization():
    """
    Carica CLIP + head dal checkpoint salvato.
    Ritorna: clip_model, head, clip_preprocess
    """
    # Carica modello CLIP e preprocess
    clip_model, clip_preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

    # Dimensione embedding (512 per ViT-B/32)
    embed_dim = clip_model.text_projection.shape[1]

    # Inizializza la head
    head = LinearHead(embed_dim, n_classes=2).to(DEVICE)

    # Carica stato dal checkpoint
    assert os.path.isfile(SAVE_PATH), f"Checkpoint non trovato: {SAVE_PATH}"
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)

    # Carica solo la parte visual di CLIP e la head
    _ = clip_model.visual.load_state_dict(ckpt["visual"], strict=False)
    head.load_state_dict(ckpt["head"])

    clip_model.eval()
    head.eval()

    for p in clip_model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = False

    print("✓ Modello CLIP e head caricati dal checkpoint.")
    return clip_model, head, clip_preprocess

@torch.no_grad()
def extract_embeddings_and_labels(clip_model, clip_preprocess, max_samples=None):
    """
    Estrae gli embedding dal vision encoder CLIP per il set di test (Celeb-test).
    Ritorna:
        Z : np.array [N, D] embedding normalizzati
        Y : np.array [N]    labels (0=real, 1=fake)
    max_samples: se non None, sottocampiona casualmente a max_samples punti.
    """
    test_loader = build_test_loader(clip_preprocess)

    all_z = []
    all_y = []

    pbar = tqdm(test_loader, desc="Estrazione embedding (test set)")
    for imgs, labels, paths in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        # embedding dal vision encoder CLIP
        z = F.normalize(clip_model.encode_image(imgs), dim=-1)  # [B, 512]
        all_z.append(z.cpu())
        all_y.append(labels.clone().cpu())

    Z = torch.cat(all_z, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()

    if max_samples is not None and Z.shape[0] > max_samples:
        idx = np.random.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]
        Y = Y[idx]

    print(f"✓ Embedding estratti: {Z.shape[0]} campioni, dim = {Z.shape[1]}")
    return Z, Y

def project_to_3d_with_pca(Z):
    """
    Proietta gli embedding Z (N, D) in 3D usando PCA.
    Restituisce Z3 (N, 3) e l'oggetto pca.
    """
    pca = PCA(n_components=3)
    Z3 = pca.fit_transform(Z)
    print("Varianza spiegata dalle 3 componenti PCA:",
          pca.explained_variance_ratio_.sum())
    return Z3, pca

def plot_3d_embeddings(Z3, Y, title="CLIP Deepfake - Embedding 3D (PCA su Celeb-test)"):
    """
    Crea un grafico 3D:
      - REAL (label=0) = blu
      - FAKE (label=1) = rosso
    Usa marker 'o' (sfere) per i punti.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    mask_real = (Y == 0)
    mask_fake = (Y == 1)

    # REAL = blu
    ax.scatter(
        Z3[mask_real, 0],
        Z3[mask_real, 1],
        Z3[mask_real, 2],
        s=30,
        alpha=0.8,
        c="blue",
        marker="o",
        edgecolors="k",
        linewidths=0.2,
        label="REAL (0)"
    )

    # FAKE = rosso
    ax.scatter(
        Z3[mask_fake, 0],
        Z3[mask_fake, 1],
        Z3[mask_fake, 2],
        s=30,
        alpha=0.8,
        c="red",
        marker="o",
        edgecolors="k",
        linewidths=0.2,
        label="FAKE (1)"
    )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

# -------------------------------
# ESECUZIONE COMPLETA
# -------------------------------

# 1. Carica modelli dal checkpoint
clip_model_vis, head_vis, clip_preprocess_vis = load_trained_models_for_visualization()

# 2. Estrai embedding e labels dal set di test
#    Puoi cambiare max_samples=None per usare TUTTI i campioni
Z, Y = extract_embeddings_and_labels(
    clip_model_vis,
    clip_preprocess_vis,
    max_samples=3000  # ad es. 3000 punti per un grafico leggibile
)

# 3. Proietta in 3D con PCA
Z3, pca = project_to_3d_with_pca(Z)

# 4. Plot 3D (REAL=blu, FAKE=rosso)
plot_3d_embeddings(Z3, Y)