import os
import numpy as np
import pandas as pd

from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import clip


# CONFIG

MODEL_NAME = "ViT-B/16"

# checkpoint saved by the cosine+LN training script
CKPT_PATH = "/home/giadapoloni/results2/CLIP3_baseline_cosine/clip_baseline_cosine.pt"

# test directories (frames)
TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

# where to save metrics
RESULTS_DIR = "/home/giadapoloni/results_TEST/CLIP3_cosine_notext/test_cosine_ln_learnscale"
RESULTS_CSV_FRAME = os.path.join(
    RESULTS_DIR, "clip_test_metrics_global_cosine_ln_learnscale_frame.csv"
)
RESULTS_CSV_VIDEO = os.path.join(
    RESULTS_DIR, "clip_test_metrics_global_cosine_ln_learnscale_video.csv"
)

VIDEO_DECISION_THRESHOLD = 0.5
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = {".jpg"}


# MODEL

class CosineHead(nn.Module):
    """
    Cosine classifier: normalize weights and use a learnable scale.
    Input is expected to be already L2-normalized.
    """
    def __init__(self, in_dim, n_classes=2, init_scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, n_classes))  # [D, C]
        nn.init.normal_(self.W, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)), requires_grad=True)

    def forward(self, z):  # z: [B, D] (already normalized)
        Wn = F.normalize(self.W, dim=0)      # [D, C]
        logits = self.scale * (z @ Wn)       # [B, C]
        return logits


# DATASET LOADER

def collect_test_items(real_dir, fake_dir):
    """
    Collect all frame paths from real and fake test directories.
    """
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if not rroot.exists():
            continue
        for p in sorted(rroot.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": label})
    return items


class TestFrameDataset(Dataset):
    """
    Dataset for test frames.
    Returns: (image_tensor, label, path_str)
    """
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.preprocess(img)
        return img, rec["label"], str(rec["path"])


def build_test_loader(preprocess, real_dir, fake_dir):
    items = collect_test_items(real_dir, fake_dir)
    if len(items) == 0:
        raise RuntimeError(f"No frames found in {real_dir} and {fake_dir}")
    ds = TestFrameDataset(items, preprocess)
    persistent = (NUM_WORKERS > 0)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None),
    )
    return loader


# PREDICTION

@torch.no_grad()
def extract_features(clip_model, imgs):
    """
    Encode images with CLIP visual encoder and apply L2 normalization.
    """
    z = clip_model.encode_image(imgs)
    z = F.normalize(z, dim=-1)
    return z


@torch.no_grad()
def predict_batch(clip_model, head, imgs):
    """
    Forward pass for a batch: CLIP visual encoder -> L2 norm -> Cosine head.
    """
    z = extract_features(clip_model, imgs)
    logits = head(z)
    return logits


# EVALUATION

@torch.no_grad()
def evaluate(clip_model, head, data_loader, device, video_threshold=0.5):
    """
    Evaluate on test set:
      - frame-level metrics
      - video-level metrics (avg prob_fake over up to 32 frames per video)
    Video id is defined as the parent directory of each frame.
    """
    clip_model.eval()
    head.eval()

    softmax = nn.Softmax(dim=-1)

    y_true = []
    prob_fake = []

    per_video_probs = defaultdict(list)
    per_video_labels = {}

    for imgs, labels, paths in tqdm(data_loader, desc="Testing (frames)"):
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels, device=device)

        logits = predict_batch(clip_model, head, imgs)
        probs = softmax(logits)
        batch_prob_fake = probs[:, 1].detach().cpu().numpy()

        # frame-level stats
        y_true.extend(labels.cpu().numpy().tolist())
        prob_fake.extend(batch_prob_fake.tolist())

        # video-level: group by parent directory, keep at most 32 frames per video
        for pth, lab, pr in zip(paths, labels.cpu().numpy(), batch_prob_fake):
            p = Path(pth)
            video_id = str(p.parent)
            if len(per_video_probs[video_id]) < 32:
                per_video_probs[video_id].append(float(pr))
            if video_id not in per_video_labels:
                per_video_labels[video_id] = int(lab)

    # frame-level metrics
    y_true_arr = np.array(y_true, dtype=int)
    prob_fake_arr = np.array(prob_fake, dtype=float)
    y_pred_arr = (prob_fake_arr >= 0.5).astype(int)

    TP = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    TN = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
    FP = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    FN = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())

    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (TP + TN) / max(1, len(y_true_arr))
    auc_roc = (
        roc_auc_score(y_true_arr, prob_fake_arr)
        if len(np.unique(y_true_arr)) > 1
        else float("nan")
    )

    frame_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "N": int(len(y_true_arr)),
        "threshold": 0.5,
    }

    # video-level metrics
    video_ids = sorted(per_video_probs.keys())
    if len(video_ids) == 0:
        video_metrics = None
    else:
        y_true_vid = []
        prob_fake_vid = []

        for vid in video_ids:
            probs_v = per_video_probs[vid]
            avg_prob = float(np.mean(probs_v)) if len(probs_v) > 0 else 0.0
            prob_fake_vid.append(avg_prob)
            y_true_vid.append(per_video_labels.get(vid, 0))

        y_true_vid = np.array(y_true_vid, dtype=int)
        prob_fake_vid = np.array(prob_fake_vid, dtype=float)
        y_pred_vid = (prob_fake_vid >= video_threshold).astype(int)

        TPv = int(((y_pred_vid == 1) & (y_true_vid == 1)).sum())
        TNv = int(((y_pred_vid == 0) & (y_true_vid == 0)).sum())
        FPv = int(((y_pred_vid == 1) & (y_true_vid == 0)).sum())
        FNv = int(((y_pred_vid == 0) & (y_true_vid == 1)).sum())

        precision_v = TPv / (TPv + FPv + eps)
        recall_v = TPv / (TPv + FNv + eps)
        f1_v = 2 * precision_v * recall_v / (precision_v + recall_v + eps)
        accuracy_v = (TPv + TNv) / max(1, len(y_true_vid))
        auc_roc_v = (
            roc_auc_score(y_true_vid, prob_fake_vid)
            if len(np.unique(y_true_vid)) > 1
            else float("nan")
        )

        video_metrics = {
            "accuracy": accuracy_v,
            "precision": precision_v,
            "recall": recall_v,
            "f1": f1_v,
            "auc_roc": auc_roc_v,
            "TP": TPv,
            "TN": TNv,
            "FP": FPv,
            "FN": FNv,
            "N_videos": int(len(y_true_vid)),
            "threshold": video_threshold,
            "frames_per_video_avg": 32,
        }

    print("===== GLOBAL METRICS (frame-level) =====")
    print(f"Accuracy : {frame_metrics['accuracy']:.4f}")
    print(f"Precision: {frame_metrics['precision']:.4f}")
    print(f"Recall   : {frame_metrics['recall']:.4f}")
    print(f"F1       : {frame_metrics['f1']:.4f}")
    if not np.isnan(frame_metrics["auc_roc"]):
        print(f"AUC-ROC  : {frame_metrics['auc_roc']:.4f}")
    else:
        print("AUC-ROC  : n/a")
    print(
        f"TP={frame_metrics['TP']}  TN={frame_metrics['TN']}  "
        f"FP={frame_metrics['FP']}  FN={frame_metrics['FN']}  N={frame_metrics['N']}"
    )

    if video_metrics is not None:
        print("===== GLOBAL METRICS (video-level, avg 32 frames) =====")
        print(f"Videos   : {video_metrics['N_videos']}")
        print(f"Accuracy : {video_metrics['accuracy']:.4f}")
        print(f"Precision: {video_metrics['precision']:.4f}")
        print(f"Recall   : {video_metrics['recall']:.4f}")
        print(f"F1       : {video_metrics['f1']:.4f}")
        if not np.isnan(video_metrics["auc_roc"]):
            print(f"AUC-ROC  : {video_metrics['auc_roc']:.4f}")
        else:
            print("AUC-ROC  : n/a")
        print(
            f"TP={video_metrics['TP']}  TN={video_metrics['TN']}  "
            f"FP={video_metrics['FP']}  FN={video_metrics['FN']}"
        )
    else:
        print("No video groups found for video-level evaluation.")

    return frame_metrics, video_metrics

# MAIN
def main():
    print(f"Using device: {DEVICE}")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_name = ckpt.get("model_name", MODEL_NAME)
    print(f"Loaded checkpoint from: {CKPT_PATH}")
    print(f"Model name in checkpoint: {model_name}")

    # Load CLIP model and preprocess
    clip_model, preprocess = clip.load(model_name, device=DEVICE, jit=False)
    clip_model.float()

    # Load visual encoder weights from checkpoint (LN-tuned visual)
    clip_model.visual.load_state_dict(ckpt["visual"])

    # Freeze everything (inference only)
    for p in clip_model.parameters():
        p.requires_grad = False

    # Build and load CosineHead
    embed_dim = clip_model.visual.output_dim
    head = CosineHead(embed_dim, n_classes=2, init_scale=16.0).to(DEVICE)
    head.load_state_dict(ckpt["head"])

    clip_model.eval()
    head.eval()

    # Build test loader
    test_loader = build_test_loader(preprocess, TEST_REAL_DIR, TEST_FAKE_DIR)

    # Run evaluation
    frame_metrics, video_metrics = evaluate(
        clip_model,
        head,
        test_loader,
        DEVICE,
        video_threshold=VIDEO_DECISION_THRESHOLD,
    )

    # Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    frame_df = pd.DataFrame([frame_metrics])
    frame_df.to_csv(RESULTS_CSV_FRAME, index=False)
    print(f"Saved frame-level metrics CSV to: {RESULTS_CSV_FRAME}")

    if video_metrics is not None:
        video_df = pd.DataFrame([video_metrics])
        video_df.to_csv(RESULTS_CSV_VIDEO, index=False)
        print(f"Saved video-level metrics CSV to: {RESULTS_CSV_VIDEO}")
    else:
        print("Video-level CSV not created (no videos found).")


if __name__ == "__main__":
    main()
