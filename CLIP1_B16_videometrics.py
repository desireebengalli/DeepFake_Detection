"""
Valutazione per-VIDEO senza retrain.
- Carica il checkpoint leggero (bias MLP + head) salvato durante il training.
- Esegue l'inferenza sui frame di test e aggrega le probabilità per video.
- Calcola metriche globali per-VIDEO (accuracy/precision/recall/F1/AUC) e salva CSV.

Assunzioni:
- La struttura delle cartelle di test è come nel file di training:
  TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
  TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
- Il checkpoint è in SAVE_PATH.
- Stesso MODEL_NAME (CLIP ViT-B/16) e stessa logica di text-bank usata nel training.
"""

import os
import math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from sklearn.metrics import roc_auc_score

# ---------------- CONFIG ----------------
MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = (DEVICE == "cuda")

# Directory (adatta se necessario)
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
EPOCH_DIR = "/home/giadapoloni/results/CLIP1_B16"
RESULTS_DIR = "/home/giadapoloni/results/CLIP1_B16_videometrics"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# File di output
RESULTS_CSV_METRICS_VIDEO = os.path.join(RESULTS_DIR, "clip_test_metrics_per_video.csv")
RESULTS_CSV_PER_VIDEO_DETAILS = os.path.join(RESULTS_DIR, "clip_test_per_video_details.csv")

# Checkpoint (uguale a quello usato nel training)
SAVE_PATH = os.path.join(EPOCH_DIR, "clip_deepfake_FAST_T4.pt")

# Batch di test
TEST_BATCH_SIZE = 64
NUM_WORKERS = 4

# Soglia decisione a livello VIDEO
VIDEO_THRESHOLD = 0.3

# AMP (solo su CUDA)
amp_dtype = torch.float16

def autocast_ctx():
    if IS_CUDA:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    class _NullCtx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return _NullCtx()

# ---------------- Dataset & Loader ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png"}

class TestFrameDataset(torch.utils.data.Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        return self.preprocess(img), rec["label"], str(rec["path"])

def collect_test_items(real_dir, fake_dir):
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if rroot.exists():
            for p in sorted(rroot.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append({"path": p, "label": label})
    if len(items) == 0:
        raise RuntimeError("Nessun frame trovato nelle cartelle di test.")
    return items

def build_test_loader(preprocess):
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    ds = TestFrameDataset(test_items, preprocess)
    persistent = (NUM_WORKERS > 0)
    return torch.utils.data.DataLoader(
        ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None)
    )

# ---------------- Modello & util ----------------
class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def build_text_bank(clip_model, device):
    real_prompts = [
        "a real photo of a human face",
        "authentic portrait, not edited",
        "unaltered photograph of a person",
    ]
    fake_prompts = [
        "deepfake, AI-generated face",
        "synthetic portrait, computer-generated",
        "manipulated face image, fake",
    ]
    txt_r = F.normalize(clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    txt_f = F.normalize(clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)

@torch.no_grad()
def predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7):
    z = F.normalize(clip_model.encode_image(imgs), dim=-1)
    logits_sup = head(z)
    logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
    logits = alpha_sup * logits_sup + (1 - alpha_sup) * logits_txt
    return logits

def extract_video_id(path_str: str) -> str:
    p = Path(path_str)
    vid = p.parent.name  # assume struttura .../<video_id>/<frame>.jpg
    if vid == "" or vid == p.name:
        stem = p.stem
        for sep in ["__", "_", "-"]:
            if sep in stem:
                return stem.split(sep)[0]
        return stem
    return vid

# ---------------- Valutazione per-VIDEO ----------------

def eval_per_video():
    # 1) Carica CLIP e preprocess
    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    clip_model.float()  # come nel training, manteniamo FP32 per sicurezza

    # 2) Prepara head con la giusta dimensione
    embed_dim = clip_model.text_projection.shape[1]
    head = LinearHead(embed_dim, 2).to(DEVICE)

    # 3) Carica checkpoint (head + bias MLP del visual)
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"Checkpoint non trovato: {SAVE_PATH}")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])
    clip_model.visual.load_state_dict(ckpt["visual_bias"], strict=False)

    # Metti tutto in eval
    clip_model.eval()
    head.eval()

    # 4) Prepara text bank e dataloader
    text_bank = build_text_bank(clip_model, DEVICE)
    test_loader = build_test_loader(preprocess)
    softmax = nn.Softmax(dim=-1)

    # 5) Inferenza e aggregazione per video
    video_probs = defaultdict(list)
    video_labels = defaultdict(list)

    for imgs, labels, paths in tqdm(test_loader, desc="Testing (frames)"):
        imgs = imgs.to(DEVICE)
        with autocast_ctx():
            logits = predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7)
            probs = softmax(logits)  # [B, 2]
        batch_prob_fake = probs[:, 1].detach().cpu().tolist()

        for path_str, lab, pf in zip(paths, labels.tolist(), batch_prob_fake):
            vid = extract_video_id(path_str)
            video_probs[vid].append(pf)
            video_labels[vid].append(int(lab))

    if len(video_probs) == 0:
        raise RuntimeError("Nessun video rilevato. Controlla la struttura delle cartelle di test.")

    # 6) Calcolo metriche per-VIDEO
    video_ids = sorted(video_probs.keys())
    video_scores = np.array([np.mean(video_probs[vid]) for vid in video_ids], dtype=float)
    # etichetta per video: moda delle label dei suoi frame
    video_y = np.array([Counter(video_labels[vid]).most_common(1)[0][0] for vid in video_ids], dtype=int)

    video_pred = (video_scores >= VIDEO_THRESHOLD).astype(int)

    TP = int(((video_pred == 1) & (video_y == 1)).sum())
    TN = int(((video_pred == 0) & (video_y == 0)).sum())
    FP = int(((video_pred == 1) & (video_y == 0)).sum())
    FN = int(((video_pred == 0) & (video_y == 1)).sum())

    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / max(1, len(video_ids))
    auc_roc = roc_auc_score(video_y, video_scores, multi_class="ovr", average="macro")
    # auc_roc   = roc_auc_score(video_y, video_scores) if len(np.unique(video_y)) > 1 else float("nan")

    print("===== METRICHE GLOBALI (per-VIDEO) =====")
    print(f"Videos   : {len(video_ids)}")
    print(f"Threshold: {VIDEO_THRESHOLD:.2f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC-ROC  : {auc_roc:.4f}" if not np.isnan(auc_roc) else "AUC-ROC  : n/a (classi non bilanciate nei video)")
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")

    # 7) Salvataggi CSV
    metrics_video_df = pd.DataFrame([{
        "videos": int(len(video_ids)),
        "threshold": float(VIDEO_THRESHOLD),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc) if not np.isnan(auc_roc) else None,
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }])
    metrics_video_df.to_csv(RESULTS_CSV_METRICS_VIDEO, index=False)
    print(f"\n✓ Salvato CSV metriche globali (per-video) in {RESULTS_CSV_METRICS_VIDEO}")

    details_df = pd.DataFrame({
        "video_id": video_ids,
        "label": video_y,
        "prob_fake_mean": video_scores,
        "pred": video_pred,
    })
    details_df.to_csv(RESULTS_CSV_PER_VIDEO_DETAILS, index=False)
    print(f"✓ Salvato CSV dettagli (per-video) in {RESULTS_CSV_PER_VIDEO_DETAILS}")

    # 8) Metriche per-CLASSE (one-vs-all) a livello VIDEO
    per_class_rows = []
    N = len(video_ids)
    for cls in [0, 1]:
        TPc = int(((video_pred == cls) & (video_y == cls)).sum())
        FPc = int(((video_pred == cls) & (video_y != cls)).sum())
        FNc = int(((video_pred != cls) & (video_y == cls)).sum())
        TNc = int(N - TPc - FPc - FNc)

        precision_c = TPc / (TPc + FPc + eps)
        recall_c    = TPc / (TPc + FNc + eps)  # anche 'class recall' / 'class accuracy (per support)'
        acc_ova_c   = (TPc + TNc) / max(1, N)  # accuratezza one-vs-all per la classe

        per_class_rows.append({
            "class": int(cls),
            "support": int((video_y == cls).sum()),
            "precision": float(precision_c),
            "recall": float(recall_c),
            "accuracy_one_vs_all": float(acc_ova_c),
            "TP": TPc, "TN": TNc, "FP": FPc, "FN": FNc,
        })

    per_class_df = pd.DataFrame(per_class_rows)
    RESULTS_CSV_PER_CLASS = os.path.join(RESULTS_DIR, "clip_test_metrics_per_video_per_class.csv")
    per_class_df.to_csv(RESULTS_CSV_PER_CLASS, index=False)
    print(f"✓ Salvato CSV metriche per-classe (per-video) in {RESULTS_CSV_PER_CLASS}")


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    eval_per_video()

if __name__ == "__main__":
    main()