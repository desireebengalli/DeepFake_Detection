import os, math, random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score

import torch.hub
from torchvision import transforms as T

# CONFIG 

MODEL_NAME = "dino_vitb16"  
EPOCHS = 2
BATCH_SIZE = 16
ACCUM_STEPS = 8             # total = 128
NUM_WORKERS = 4

# LRs e scheduler: LR low + warmup + cosine decay
BASE_LR = 3e-4             # LR max
MIN_LR = 1e-5              # LR min at the end of training
WARMUP_RATIO = 0.05        # 5% warmup ratio

WEIGHT_DECAY = 0           

USE_WEIGHTED_SAMPLER = False
USE_CLASS_WEIGHTS = False   

VIDEO_DECISION_THRESHOLD = 0.5
SEED = 1

# Early stopping
PATIENCE = 6
MIN_DELTA = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = (DEVICE == "cuda")

# Directory
DATA_DIR = "/home/giadapoloni/preprocessed_frames"
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
RESULTS_DIR = "/home/default/resultsDINO"
RESULTS_CSV_METRICS = os.path.join(RESULTS_DIR, "dino_test_metrics_global_linear_ln.csv")
RESULTS_CSV_METRICS_VIDEO = os.path.join(RESULTS_DIR, "dino_test_metrics_video_linear_ln.csv")
SAVE_PATH = os.path.join(RESULTS_DIR, "dino_linear_ln.pt")


amp_dtype = torch.float16
scaler = torch.amp.GradScaler(device='cuda', enabled=IS_CUDA)

def autocast_ctx():
    if IS_CUDA:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    class _NullCtx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return _NullCtx()




# Utility & dataset

IMG_EXTS = {".jpg"}  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LinearHead(nn.Module):
    """
    Linear classifier
    """
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

def enable_ln_tuning_on_backbone(backbone: nn.Module):
    """
    LN-tuning
    """
    for p in backbone.parameters():
        p.requires_grad = False

    tuned_params = []
    for m in backbone.modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                m.weight.requires_grad = True
                tuned_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad = True
                tuned_params.append(m.bias)
    return tuned_params

def collect_frames(root_dir):
    root = Path(root_dir)
    items = []
    real_dir = root / "real_ff"
    if real_dir.exists():
        for vid in sorted([d for d in real_dir.iterdir() if d.is_dir()]):
            for fp in sorted([p for p in vid.iterdir() if p.suffix.lower() in IMG_EXTS]):
                items.append({"path": fp, "label": 0})
    fake_dir = root / "fake_ff"
    for m in ["deepfakes", "face2face", "faceswap", "neuraltextures"]:
        mdir = fake_dir / m
        if not mdir.exists():
            continue
        for vid in sorted([d for d in mdir.iterdir() if d.is_dir()]):
            for fp in sorted([p for p in vid.iterdir() if p.suffix.lower() in IMG_EXTS]):
                items.append({"path": fp, "label": 1})
    return items

def collect_test_items(real_dir, fake_dir):
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if rroot.exists():
            for p in sorted(rroot.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items.append({"path": p, "label": label})
    return items

class FrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(self.items[i]["path"]).convert("RGB")
        img = self.preprocess(img)
        return img, self.items[i]["label"]

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

def build_train_loader(preprocess):
    items = collect_frames(DATA_DIR)
    assert len(items) > 0, "No frame in DATA_DIR."
    ds = FrameDataset(items, preprocess)
    persistent = (NUM_WORKERS > 0)
    if USE_WEIGHTED_SAMPLER:
        labels = torch.tensor([it["label"] for it in items])
        binc = torch.bincount(labels)
        class_w = 1.0 / binc.float()
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=(2 if persistent else None),
        )
    else:
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=persistent,
            prefetch_factor=(2 if persistent else None),
        )

def build_test_loader(preprocess):
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    ds = TestFrameDataset(test_items, preprocess)
    persistent = (NUM_WORKERS > 0)
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=(2 if persistent else None),
    )

def build_optimizer_and_scheduler(head, tuned_params, total_steps):
    params = list(head.parameters()) + [p for p in tuned_params if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )

    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    def lr_mult(step):
        if step < warmup_steps:
            warmup_factor = float(step + 1) / float(warmup_steps)
            return (MIN_LR / BASE_LR) + (1.0 - MIN_LR / BASE_LR) * warmup_factor

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
        lr_factor = (MIN_LR / BASE_LR) + (1.0 - MIN_LR / BASE_LR) * cos_factor
        return lr_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_mult(s),
    )
    return optimizer, scheduler

# DINO: backbone + preprocess 

def get_dino_backbone(device):
    """
    DINO ViT-B/16 (facebookresearch/dino).
    """
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    backbone.to(device)
    return backbone

def get_dino_preprocess():
    """
    Preprocess standard ImageNet (comparible with DINO ViT-B/16).
    """
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

def dino_extract_features(backbone, imgs):
    feats = backbone.get_intermediate_layers(imgs, n=1)[0][:, 0]
    feats = F.normalize(feats, dim=-1)
    return feats

@torch.no_grad()
def dino_predict_batch(backbone, head, imgs):
    z = dino_extract_features(backbone, imgs)
    logits = head(z)
    return logits

# EVALUATION (frame + video) 

@torch.no_grad()
def evaluate(backbone, head, data_loader, device, video_threshold=0.5, verbose=False):
    backbone.eval()
    head.eval()

    softmax = nn.Softmax(dim=-1)
    y_true, prob_fake = [], []
    per_video_probs = defaultdict(list)
    per_video_labels = {}

    for imgs, labels, paths in tqdm(data_loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        with autocast_ctx():
            logits = dino_predict_batch(backbone, head, imgs)
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
        print(f"TP={frame_metrics['TP']}  TN={frame_metrics['TN']}  FP={frame_metrics['FP']}  FN={frame_metrics['FN']}  N={frame_metrics['N']}")

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
            print("No video found.")

    return frame_metrics, video_metrics

# Training + Test (early stopping, if applied)

def train_and_eval():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #loading DINO
    backbone = get_dino_backbone(DEVICE)
    preprocess = get_dino_preprocess()
    backbone.float()

    # LN-tuning 
    tuned_params = enable_ln_tuning_on_backbone(backbone)

    # Linear head
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
        dummy_feats = dino_extract_features(backbone, dummy)
    embed_dim = dummy_feats.shape[-1]
    backbone.train()

    head = LinearHead(embed_dim, 2).to(DEVICE)

    # Dataloader
    train_loader = build_train_loader(preprocess)
    test_loader = build_test_loader(preprocess)

    # Loss (cross-entropy)
    if USE_CLASS_WEIGHTS:
        counts = Counter([it["label"] for it in collect_frames(DATA_DIR)])
        total = sum(counts.values())
        class_weights = torch.tensor(
            [total / (2 * counts[0]), total / (2 * counts[1])],
            device=DEVICE,
            dtype=torch.float32,
        )
    else:
        class_weights = None
    ce = nn.CrossEntropyLoss(weight=class_weights)

    # Scheduler (warmup + cosine decay)
    steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    optimizer, scheduler = build_optimizer_and_scheduler(head, tuned_params, total_steps)

    # TRAIN
    backbone.train()
    head.train()

    best_auc_v = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        # Train Loop
        for step, (imgs, y) in enumerate(pbar, 1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            y = torch.as_tensor(y, device=DEVICE)

            with autocast_ctx():
                z = dino_extract_features(backbone, imgs)
                logits = head(z)
                loss = ce(logits, y)

            # gradient accumulation + scaler
            if IS_CUDA:
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            if step % ACCUM_STEPS == 0:
                did_step = True
                if IS_CUDA:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_step = scaler.get_scale() >= prev_scale
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if did_step:
                    scheduler.step()
                    global_step += 1

            running += loss.item()
            pbar.set_postfix(loss=running / max(1, step))

        # VALIDATION (+early stopping if applied
        print(f"\nValidation after epoch {epoch}...")
        frame_m, video_m = evaluate(
            backbone, head, test_loader, DEVICE,
            video_threshold=VIDEO_DECISION_THRESHOLD,
            verbose=False
        )

        if video_m is None or np.isnan(video_m["auc_roc"]):
            print("No video")
            current_auc_v = float("nan")
            improved = False
        else:
            current_auc_v = video_m["auc_roc"]
            print(f"[Val] Epoch {epoch} - video AUC: {current_auc_v:.4f}")

            improved = current_auc_v > (best_auc_v + MIN_DELTA)

        if improved:
            print("New best epoch, saving checkpoint...")
            best_auc_v = current_auc_v
            best_epoch = epoch
            epochs_no_improve = 0

            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            ckpt = {
                "epoch": epoch,
                "model_name": MODEL_NAME,
                "head": head.state_dict(),
                "backbone": backbone.state_dict(),
            }
            torch.save(ckpt, SAVE_PATH)
        else:
            epochs_no_improve += 1
            print(f"No improvement (AUC). epochs_no_improve = {epochs_no_improve}/{PATIENCE}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

        backbone.train()
        head.train()

    print(f"\nTrain ended. Best epoch = {best_epoch}, best video AUC = {best_auc_v:.4f}")

    # TEST final on best checkpoint
    print("Reloading best checkpoint and computing metrics...")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])
    backbone.load_state_dict(ckpt["backbone"])

    frame_metrics, video_metrics = evaluate(
        backbone, head, test_loader, DEVICE,
        video_threshold=VIDEO_DECISION_THRESHOLD,
        verbose=True
    )

    # CSV (frame-level)
    os.makedirs(os.path.dirname(RESULTS_CSV_METRICS), exist_ok=True)
    metrics_df = pd.DataFrame([frame_metrics])
    metrics_df.to_csv(RESULTS_CSV_METRICS, index=False)
    print(f"Saved CSV (frame) in {RESULTS_CSV_METRICS}")

    # CSV (video-level)
    if video_metrics is not None:
        os.makedirs(os.path.dirname(RESULTS_CSV_METRICS_VIDEO), exist_ok=True)
        metrics_vid_df = pd.DataFrame([video_metrics])
        metrics_vid_df.to_csv(RESULTS_CSV_METRICS_VIDEO, index=False)
        print(f"Saved CSV (video) in {RESULTS_CSV_METRICS_VIDEO}")
    else:
        print("No video.")

def main():
    train_and_eval()

if __name__ == "__main__":
    main()
