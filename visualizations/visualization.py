import os, math, random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt


# CONFIG

MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

CLIP_MODEL = "/home/giadapoloni/results2/CLIP5_ln_bias_linear_notext"
SAVE_PATH = os.path.join(CLIP_MODEL, "clip5_linear_ln_bias_notext.pt")

RESULTS_DIR = "/home/giadapoloni/visualizations"
VIS_DIR = os.path.join(RESULTS_DIR, "attention maps")
os.makedirs(VIS_DIR, exist_ok=True)

path_real = "/home/giadapoloni/C_test/C_real/id1_0000/C_real_id1_0000_001_ctx.jpg"
path_fake = "/home/giadapoloni/C_test/C_fake/id1_id4_0000/C_fake_id1_id4_0000_001_ctx.jpg"
path_fake2 = "/home/giadapoloni/C_test/C_fake/id1_id9_0000/C_fake_id1_id9_0000_001_ctx.jpg"
path_fake3 = "/home/giadapoloni/C_test/C_fake/id1_id16_0000/C_fake_id1_id16_0000_001_ctx.jpg"

IMG_EXTS = {".jpg"}



class LinearHead(nn.Module):
    """Linear classifier"""
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
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append({"path": p, "label": label})
    return items


# CLIP + best checkpoint
def load_best_model():
    print(f"Loading CLIP model {MODEL_NAME} on {DEVICE}...")
    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

    clip_model.float()  

    embed_dim = clip_model.visual.output_dim
    head = LinearHead(embed_dim, 2).to(DEVICE)

    print(f"Loading checkpoint from {SAVE_PATH} ...")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])
    clip_model.visual.load_state_dict(ckpt["visual"])

    clip_model.eval()
    head.eval()

    return clip_model, head, preprocess



# Residual block with attention
def block_forward_with_attn(block, x):

    attn_mask = block.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

    # LayerNorm + MultiHeadAttention
    x_ln = block.ln_1(x)          

    attn_output, attn_weights = block.attn(
        x_ln, x_ln, x_ln,
        need_weights=True,
        attn_mask=attn_mask,
        average_attn_weights=False
    )


    x = x + attn_output
    x = x + block.mlp(block.ln_2(x))

    return x, attn_weights


def normalize_attn_weights(attn_weights, num_heads, batch_size):

    w = attn_weights

    if w.dim() == 4:
        w = w.mean(dim=1)[0]          
    elif w.dim() == 3:
        if w.shape[0] == num_heads:
            w = w.mean(dim=0)          # [tgt, src]
        else:
            w = w.view(batch_size, num_heads, w.shape[1], w.shape[2])
            w = w.mean(dim=1)[0]       # [tgt, src]
    else:
        raise ValueError(f"Unexpected attn_weights shape: {w.shape}")

    return w  # [seq, seq]


# Forward visual + attention rollout

def forward_visual_with_rollout(clip_model, head, img_tensor):
    visual = clip_model.visual

    with torch.no_grad():
        # 1) patch embedding
        x = visual.conv1(img_tensor)               # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid*grid]
        x = x.permute(0, 2, 1)                     # [B, tokens, width]

        # 2) CLS and pos embedding
        class_embedding = visual.class_embedding.to(x.dtype)
        class_tokens = class_embedding + torch.zeros(
            x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_tokens, x], dim=1)    

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # projection
        x = x.permute(1, 0, 2)                    # [seq, B, D]

        attn_mats = []
        num_heads = visual.transformer.resblocks[0].attn.num_heads
        batch_size = img_tensor.shape[0]

        for block in visual.transformer.resblocks:
            x, attn_weights = block_forward_with_attn(block, x)
            w = normalize_attn_weights(attn_weights, num_heads, batch_size)
            attn_mats.append(w.cpu())             # [seq, seq]


        x = x.permute(1, 0, 2)

        # CLS pooling + proj 
        x_cls = x[:, 0, :]
        x_cls = visual.ln_post(x_cls)
        if visual.proj is not None:
            x_cls = x_cls @ visual.proj

        embedding = F.normalize(x_cls, dim=-1)
        logits = head(embedding)                  # [B, 2]

    # attention rollout
    rollout = None
    for w in attn_mats:
        # w: [seq, seq]
        n = w.shape[0]
        w = w + torch.eye(n)
        w = w / w.sum(dim=-1, keepdim=True)

        rollout = w if rollout is None else w @ rollout

    # CLS -> patch
    cls_attn = rollout[0]         # [seq]
    cls_attn = cls_attn[1:]       # excluding CLS

    # normalizing [0,1]
    cls_attn = cls_attn - cls_attn.min()
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)

    cls_attn_np = cls_attn.numpy()

    num_tokens = cls_attn_np.shape[0]
    side = int(round(math.sqrt(num_tokens)))
    cls_attn_np = cls_attn_np[:side * side]
    heatmap_patches = cls_attn_np.reshape(side, side)   

    return logits, heatmap_patches



# Utils per heatmap e overlay
def resize_heatmap_to_image(heatmap, image_size):
    """heatmap [Hp, Wp] -> [H, W] in [0,1]"""
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_img = heatmap_img.resize(image_size, resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0
    return heatmap_arr


def save_overlay(pil_img, heatmap, out_path, alpha=0.5):
    plt.figure(figsize=(4, 4))
    plt.imshow(pil_img)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# Visualization
# def visualize_triplet(path_real, fake_list):
#     clip_model, head, preprocess = load_best_model()

#     # ---------- REAL ----------
#     pil_real = Image.open(path_real).convert("RGB")
#     img_real = preprocess(pil_real).unsqueeze(0).to(DEVICE)
#     logits_r, heatmap_r = forward_visual_with_rollout(clip_model, head, img_real)
#     probs_r = torch.softmax(logits_r, dim=-1)[0].cpu().numpy()
#     pred_r = int(probs_r.argmax())

#     print(f"REAL img: {path_real}")
#     print(f"  Predicted: {'fake' if pred_r == 1 else 'real'} "
#           f"(p_fake={probs_r[1]:.3f}, p_real={probs_r[0]:.3f})")

#     heatmap_r_full = resize_heatmap_to_image(heatmap_r, pil_real.size)

#     # ---------- FAKE IMAGES ----------
#     fake_images = []
#     fake_heatmaps = []
#     fake_preds = []

#     for pf in fake_list:
#         pil_f = Image.open(pf).convert("RGB")
#         img_f = preprocess(pil_f).unsqueeze(0).to(DEVICE)
#         logits_f, heatmap_f = forward_visual_with_rollout(clip_model, head, img_f)
#         probs_f = torch.softmax(logits_f, dim=-1)[0].cpu().numpy()
#         pred_f = int(probs_f.argmax())

#         print(f"FAKE img: {pf}")
#         print(f"  Predicted: {'fake' if pred_f == 1 else 'real'} "
#               f"(p_fake={probs_f[1]:.3f}, p_real={probs_f[0]:.3f})")

#         fake_images.append(pil_f)
#         fake_heatmaps.append(resize_heatmap_to_image(heatmap_f, pil_f.size))
#         fake_preds.append(pred_f)

#     # ---------- VISUALIZATION ----------
#     n_cols = 2 + len(fake_list)   # REAL, REAL+attn, FAKE1, FAKE2, FAKE3

#     plt.figure(figsize=(4 * n_cols, 4))

#     # Real without heatmap
#     plt.subplot(1, n_cols, 1)
#     plt.title("REAL")
#     plt.imshow(pil_real)
#     plt.axis("off")

#     # Real with heatmap
#     plt.subplot(1, n_cols, 2)
#     plt.title("REAL + attn")
#     plt.imshow(pil_real)
#     plt.imshow(heatmap_r_full, cmap="jet", alpha=0.5)
#     plt.axis("off")

#     # Fake + heatmap images
#     for i, (img_f, hm_f) in enumerate(zip(fake_images, fake_heatmaps)):
#         plt.subplot(1, n_cols, 3 + i)
#         plt.title(f"FAKE {i+1} + attn")
#         plt.imshow(img_f)
#         plt.imshow(hm_f, cmap="jet", alpha=0.5)
#         plt.axis("off")

#     plt.tight_layout()

#     out_name = f"triplet_rollout_{os.path.basename(path_real)}.png"
#     out_path = os.path.join(VIS_DIR, out_name)
#     plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
#     plt.close()

#     print(f"  -> Saved triplet visualization to: {out_path}")

def visualize_triplet(path_real, fake_list):
    clip_model, head, preprocess = load_best_model()

    # ---------- REAL ----------
    pil_real = Image.open(path_real).convert("RGB")
    img_real = preprocess(pil_real).unsqueeze(0).to(DEVICE)
    logits_r, heatmap_r = forward_visual_with_rollout(clip_model, head, img_real)
    probs_r = torch.softmax(logits_r, dim=-1)[0].cpu().numpy()
    pred_r = int(probs_r.argmax())

    print(f"REAL img: {path_real}")
    print(
        f"  Predicted: {'fake' if pred_r == 1 else 'real'} "
        f"(p_fake={probs_r[1]:.3f}, p_real={probs_r[0]:.3f})"
    )

    heatmap_r_full = resize_heatmap_to_image(heatmap_r, pil_real.size)

    # ---------- FAKE IMAGES ----------
    fake_images = []
    fake_heatmaps = []
    fake_preds = []

    for pf in fake_list:
        pil_f = Image.open(pf).convert("RGB")
        img_f = preprocess(pil_f).unsqueeze(0).to(DEVICE)
        logits_f, heatmap_f = forward_visual_with_rollout(clip_model, head, img_f)
        probs_f = torch.softmax(logits_f, dim=-1)[0].cpu().numpy()
        pred_f = int(probs_f.argmax())

        print(f"FAKE img: {pf}")
        print(
            f"  Predicted: {'fake' if pred_f == 1 else 'real'} "
            f"(p_fake={probs_f[1]:.3f}, p_real={probs_f[0]:.3f})"
        )

        fake_images.append(pil_f)
        fake_heatmaps.append(resize_heatmap_to_image(heatmap_f, pil_f.size))
        fake_preds.append(pred_f)

    # ---------- VISUALIZZAZIONE 2x4 ----------
    n_cols = 1 + len(fake_list)   # colonna 0 = REAL, poi FAKE 1..N

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # --- Riga superiore: immagini senza heatmap ---

    # Colonna 0: REAL senza heatmap
    axes[0, 0].set_title("REAL")
    axes[0, 0].imshow(pil_real)
    axes[0, 0].axis("off")

    # Colonne 1..: FAKE senza heatmap
    for i, img_f in enumerate(fake_images):
        col = 1 + i
        axes[0, col].set_title(f"FAKE {i+1}")
        axes[0, col].imshow(img_f)
        axes[0, col].axis("off")

    # --- Riga inferiore: immagini con heatmap ---

    # Colonna 0: REAL con heatmap
    axes[1, 0].set_title("REAL + attn")
    axes[1, 0].imshow(pil_real)
    axes[1, 0].imshow(heatmap_r_full, cmap="jet", alpha=0.5)
    axes[1, 0].axis("off")

    # Colonne 1..: FAKE con heatmap
    for i, (img_f, hm_f) in enumerate(zip(fake_images, fake_heatmaps)):
        col = 1 + i
        axes[1, col].set_title(f"FAKE {i+1} + attn")
        axes[1, col].imshow(img_f)
        axes[1, col].imshow(hm_f, cmap="jet", alpha=0.5)
        axes[1, col].axis("off")

    plt.tight_layout()

    out_name = f"triplet_rollout_{os.path.basename(path_real)}.png"
    out_path = os.path.join(VIS_DIR, out_name)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"  -> Saved triplet visualization to: {out_path}")

    

if __name__ == "__main__":
    fake_list = [path_fake, path_fake2, path_fake3]
    visualize_triplet(path_real, fake_list)

