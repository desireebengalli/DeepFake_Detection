import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import clip
from collections import defaultdict
from sklearn.metrics import roc_auc_score


# CONFIG

MODEL_NAME = "ViT-B/16"

CKPT_PATH = "/home/liciabordignion/results2/CLIP4_cosine_ln_with_text/clip4_cosine_ln_with_text.pt"

TEST_REAL_DIR = "/home/giadapoloni/C_test/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_test/C_fake"

RESULTS_DIR = "/home/liciabordignion/results2/CLIP4_cosine_ln_with_text/test"
RESULTS_CSV_METRICS = os.path.join(RESULTS_DIR, "clip4_test_metrics_global_cosine_ln_with_text_frame.csv")
RESULTS_CSV_METRICS_VIDEO = os.path.join(RESULTS_DIR, "clip4_test_metrics_video_cosine_ln_with_text_video.csv")

VIDEO_DECISION_THRESHOLD = 0.5
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = (DEVICE == "cuda")

IMG_EXTS = {".jpg"}

amp_dtype = torch.float16

def autocast_ctx():
    if IS_CUDA:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    class _NullCtx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return _NullCtx()


# COSINE HEAD

class CosineHead(nn.Module):
    """
    Cosine classifier: normalizza i pesi e usa uno scale learnable.
    Input atteso: z già normalizzato (F.normalize).
    """
    def __init__(self, in_dim, n_classes=2, init_scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, n_classes))  # [D, C]
        nn.init.normal_(self.W, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)), requires_grad=True)

    def forward(self, z):  # z: [B, D] (già normalizzato)
        Wn = F.normalize(self.W, dim=0)         # [D, C]
        logits = self.scale * (z @ Wn)          # [B, C]
        return logits


# DATASET

def collect_test_items(real_dir, fake_dir):
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if rroot.exists():
            for p in sorted(rroot.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append({"path": p, "label": label})
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


def build_test_loader(preprocess):
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    if len(test_items) == 0:
        raise RuntimeError(f"No frames found in {TEST_REAL_DIR} and {TEST_FAKE_DIR}")
    ds = TestFrameDataset(test_items, preprocess)
    persistent = (NUM_WORKERS > 0)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None),
    )


# TEXT BANK

@torch.no_grad()
def build_text_bank(clip_model, device):
    real_prompts = ["a real photo of a human face",
                    "authentic portrait, not edited",
                    "unaltered photograph of a person"]
    fake_prompts = ["deepfake, AI-generated face",
                    "synthetic portrait, computer-generated",
                    "manipulated face image, fake"]
    txt_r = F.normalize(
        clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1
    ).mean(0, keepdim=True)
    txt_f = F.normalize(
        clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1
    ).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)


@torch.no_grad()
def predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7):
    z = F.normalize(clip_model.encode_image(imgs), dim=-1)
    logits_sup = head(z)
    logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
    logits = alpha_sup * logits_sup + (1 - alpha_sup) * logits_txt
    return logits


# EVALUATION (frame + video)

@torch.no_grad()
def evaluate(clip_model, head, data_loader, device, text_bank, video_threshold=0.5, verbose=False):
    clip_model.eval()
    head.eval()

    softmax = nn.Softmax(dim=-1)
    y_true, prob_fake = [], []
    per_video_probs = defaultdict(list)
    per_video_labels = {}

    for imgs, labels, paths in tqdm(data_loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        with autocast_ctx():
            logits = predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7)
            probs = softmax(logits)
        batch_prob_fake = probs[:, 1].detach().cpu().numpy()

        # frame-level
        y_true += list(labels)
        prob_fake += batch_prob_fake.tolist()

        # video-level
        for pth, lab, pr in zip(paths, labels, batch_prob_fake):
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
    auc_roc = roc_auc_score(y_true_arr, prob_fake_arr) if len(np.unique(y_true_arr)) > 1 else float("nan")

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
        y_true_vid, prob_fake_vid = [], []
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
        auc_roc_v = roc_auc_score(y_true_vid, prob_fake_vid) if len(np.unique(y_true_vid)) > 1 else float("nan")

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

    if verbose:
        print("===== GLOBAL METRICS (per-FRAME) =====")
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
            print("===== GLOBAL METRICS (per-VIDEO, avg 32 frame) =====")
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
            print("ERROR - NO VIDEO FOUND")

    return frame_metrics, video_metrics


# MAIN

def main():
    print(f"Using device: {DEVICE}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_name = ckpt.get("model_name", MODEL_NAME)
    print(f"Loaded checkpoint from: {CKPT_PATH}")
    print(f"Model name in checkpoint: {model_name}")

    clip_model, preprocess = clip.load(model_name, device=DEVICE, jit=False)
    clip_model.float()

    # load LN-tuned visual encoder
    clip_model.visual.load_state_dict(ckpt["visual"])

    # freeze all params (inference only)
    for p in clip_model.parameters():
        p.requires_grad = False

    # build head and load weights
    embed_dim = clip_model.visual.output_dim
    head = CosineHead(embed_dim, n_classes=2, init_scale=16.0).to(DEVICE)
    head.load_state_dict(ckpt["head"])

    # build text bank
    text_bank = build_text_bank(clip_model, DEVICE)

    # build test loader
    test_loader = build_test_loader(preprocess)

    # evaluate
    frame_metrics, video_metrics = evaluate(
        clip_model, head, test_loader, DEVICE, text_bank,
        video_threshold=VIDEO_DECISION_THRESHOLD, verbose=True
    )

    # save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_df = pd.DataFrame([frame_metrics])
    metrics_df.to_csv(RESULTS_CSV_METRICS, index=False)
    print(f"Saved CSV global metrics (frame) in {RESULTS_CSV_METRICS}")

    if video_metrics is not None:
        metrics_vid_df = pd.DataFrame([video_metrics])
        metrics_vid_df.to_csv(RESULTS_CSV_METRICS_VIDEO, index=False)
        print(f"Saved CSV global metrics (video) in {RESULTS_CSV_METRICS_VIDEO}")
    else:
        print("CSV video not created (no videos found).")


if __name__ == "__main__":
    main()
