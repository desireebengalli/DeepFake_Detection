# ============================
# FAST TRAIN (frames preprocessati) - ViT-B/16
# Cosine head + PEFT (bias-tuning su MLP del vision encoder)
# Cosine LR con warmup, gradient accumulation, loss ibrida (label + testo + contrastiva)
# TEST:
#   - per-FRAME -> metriche globali salvate in CSV
#   - per-VIDEO -> media su 32 frame/video, metriche globali salvate in secondo CSV
# Configurato per GPU T4 (FP16 + GradScaler)
# ============================

import os, math, random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import clip
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score

# ---------- CONFIG per T4 ----------
MODEL_NAME = "ViT-B/16"     # cambiato da ViT-L/14 a ViT-B/16
EPOCHS = 10
BATCH_SIZE = 16
ACCUM_STEPS = 8            # effettivo 128
NUM_WORKERS = 4

LR_HEAD = 8e-4
LR_BIAS = 3e-4
WEIGHT_DECAY_HEAD = 1e-5
WEIGHT_DECAY_BIAS = 0.0
WARMUP_RATIO = 0.06

USE_WEIGHTED_SAMPLER = False
ALPHA_SUP_EPOCH_SCHEDULE = [(1, 0.60), (4, 0.65), (7, 0.70)]
VIDEO_DECISION_THRESHOLD = 0.5  # threshold per label video

SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = (DEVICE == "cuda")

# Directory (aggiustale sulla VM)
DATA_DIR = "/home/giadapoloni/preprocessed_frames"
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
RESULTS_DIR = "/home/giadapoloni/results/CLIP1_B16_cosine"
RESULTS_CSV_METRICS = os.path.join(RESULTS_DIR, "clip_test_metrics_global.csv")
RESULTS_CSV_METRICS_VIDEO = os.path.join(RESULTS_DIR, "clip_test_metrics_video.csv")
SAVE_PATH = os.path.join(RESULTS_DIR, "clip_cosine.pt")

# AMP: FP16 per T4 (solo se CUDA)
amp_dtype = torch.float16
# scaler = torch.cuda.amp.GradScaler(enabled=IS_CUDA)
scaler = torch.amp.GradScaler(device='cuda', enabled=IS_CUDA)

# wrapper per autocast che si disattiva automaticamente su CPU
def autocast_ctx():
    if IS_CUDA:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    # contesto nullo su CPU
    class _NullCtx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return _NullCtx()

# ---------- Funzioni generiche ----------
IMG_EXTS = {".jpg"}

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

def enable_bias_tuning_on_mlp(visual_module):
    # disabilita tutto
    for p in visual_module.parameters():
        p.requires_grad = False
    tuned = []
    # abilita SOLO i bias dei due Linear dell'MLP in ogni residual block
    if hasattr(visual_module, "transformer") and hasattr(visual_module.transformer, "resblocks"):
        for blk in visual_module.transformer.resblocks:
            if hasattr(blk, "mlp"):
                if isinstance(blk.mlp[0], nn.Linear) and blk.mlp[0].bias is not None:
                    blk.mlp[0].bias.requires_grad = True; tuned.append(blk.mlp[0].bias)
                if isinstance(blk.mlp[2], nn.Linear) and blk.mlp[2].bias is not None:
                    blk.mlp[2].bias.requires_grad = True; tuned.append(blk.mlp[2].bias)
    return tuned

def supervised_contrastive_loss(z, y, tau=0.07):
    sim = (z @ z.t()) / tau
    mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float()
    exp_sim = torch.exp(sim) * (1 - torch.eye(len(y), device=y.device))
    pos = exp_sim * mask
    neg = exp_sim * (1 - mask)
    pos_sum, neg_sum = pos.sum(1), neg.sum(1)
    loss_i = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)
    return loss_i.mean()

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
        if not mdir.exists(): continue
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
        self.items = items; self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img = Image.open(self.items[i]["path"]).convert("RGB")
        return self.preprocess(img), self.items[i]["label"]

class TestFrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items; self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        return self.preprocess(img), rec["label"], str(rec["path"])

def build_train_loader(preprocess):
    items = collect_frames(DATA_DIR)
    assert len(items) > 0, "Nessun frame trovato in DATA_DIR."
    ds = FrameDataset(items, preprocess)
    persistent = (NUM_WORKERS > 0)
    if USE_WEIGHTED_SAMPLER:
        labels = torch.tensor([it["label"] for it in items])
        binc = torch.bincount(labels)
        class_w = 1.0 / binc.float()
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=persistent, prefetch_factor=(2 if persistent else None))
    else:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=persistent, prefetch_factor=(2 if persistent else None))

def build_test_loader(preprocess):
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    ds = TestFrameDataset(test_items, preprocess)
    persistent = (NUM_WORKERS > 0)
    return DataLoader(ds, batch_size=64, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=persistent, prefetch_factor=(2 if persistent else None))

def build_optimizer_and_scheduler(head, tuned_bias_params, total_steps):
    optimizer = torch.optim.Adam(
        [
            {"params": head.parameters(), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY_HEAD},
            {"params": [p for p in tuned_bias_params if p.requires_grad], "lr": LR_BIAS, "weight_decay": WEIGHT_DECAY_BIAS},
        ], betas=(0.9, 0.999)
    )
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    def lr_mult(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_mult(s))
    return optimizer, scheduler

@torch.no_grad()
def build_text_bank(clip_model, device):
    real_prompts = ["a real photo of a human face","authentic portrait, not edited","unaltered photograph of a person"]
    fake_prompts = ["deepfake, AI-generated face","synthetic portrait, computer-generated","manipulated face image, fake"]
    txt_r = F.normalize(clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    txt_f = F.normalize(clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)

def alpha_for_epoch(epoch):
    a = ALPHA_SUP_EPOCH_SCHEDULE[0][1]
    for e, val in ALPHA_SUP_EPOCH_SCHEDULE:
        if epoch >= e: a = val
    return a

@torch.no_grad()
def predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7):
    z = F.normalize(clip_model.encode_image(imgs), dim=-1)
    logits_sup = head(z)
    logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
    logits = alpha_sup * logits_sup + (1 - alpha_sup) * logits_txt
    return logits

# ---------- Training + Test ----------
def train_and_eval():
    set_seed = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s))
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1) Carica CLIP
    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

    # 2) Porta tutto in FP32 per evitare "Attempting to unscale FP16 gradients"
    clip_model.float()

    # 3) Congela tutto e lasciare liberi solo i bias MLP del visual
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.logit_scale.requires_grad = False

    embed_dim = clip_model.text_projection.shape[1]
    text_bank = build_text_bank(clip_model, DEVICE)
    tuned_bias = enable_bias_tuning_on_mlp(clip_model.visual)

    train_loader = build_train_loader(preprocess)
    test_loader = build_test_loader(preprocess)

    # class weights per sbilanciamento 80/20 (adatta ai tuoi dati)
    counts = Counter([it["label"] for it in collect_frames(DATA_DIR)])
    total = sum(counts.values())
    class_weights = torch.tensor([total/(2*counts[0]), total/(2*counts[1])], device=DEVICE, dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=class_weights if not USE_WEIGHTED_SAMPLER else None)

    # --- Cosine head ---
    head = CosineHead(embed_dim, 2, init_scale=16.0).to(DEVICE)

    steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1)//ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    optimizer, scheduler = build_optimizer_and_scheduler(head, tuned_bias, total_steps)

    # come richiesto: non mettiamo il modello in train, ma manteniamo il visual in eval
    clip_model.visual.eval()
    head.train()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        alpha_sup = alpha_for_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, y) in enumerate(pbar, 1):
            imgs, y = imgs.to(DEVICE, non_blocking=True), torch.as_tensor(y, device=DEVICE)

            with autocast_ctx():
                z = F.normalize(clip_model.encode_image(imgs), dim=-1)
                logits_sup = head(z)
                logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
                loss = 0.4*ce(logits_sup, y) + 0.4*ce(logits_txt, y) + 0.2*supervised_contrastive_loss(z, y)

            # gradient accumulation + scaler
            if IS_CUDA:
                scaler.scale(loss/ACCUM_STEPS).backward()
            else:
                (loss/ACCUM_STEPS).backward()

            # if step % ACCUM_STEPS == 0:
            #     if IS_CUDA:
            #         scaler.step(optimizer); scaler.update()
            #     else:
            #         optimizer.step()
            #     optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1

            if step % ACCUM_STEPS == 0:
                did_step = True
                if IS_CUDA:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)         # può essere skippato in caso di overflow
                    scaler.update()
                    # se c'è stato overflow: new_scale < prev_scale -> lo step è stato SKIPPATO
                    did_step = scaler.get_scale() >= prev_scale
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # Chiama lo scheduler SOLO se l'optimizer ha fatto davvero lo step
                if did_step:
                    scheduler.step()
                    global_step += 1

            running += loss.item()
            pbar.set_postfix(loss=running/max(1, step))

        # ---------- CHECKPOINT LEGGERO (solo head + bias MLP tunati) ----------
        tuned_bias_state = {
            name: p.detach().cpu()
            for name, p in clip_model.visual.named_parameters()
            if p.requires_grad and name.endswith("bias")
        }
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(
            {
                "head":       head.state_dict(),
                "visual_bias": tuned_bias_state,
                "epoch":      epoch,
                "model_name": MODEL_NAME,
            },
            SAVE_PATH
        )
        print(f"✓ Saved lightweight checkpoint epoch {epoch} -> {SAVE_PATH}")

    print("Training completato. Avvio test...")

    # ----- TEST -----
    clip_model.eval(); head.eval()
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])

    # Copia diretta dei bias salvati nei parametri del visual
    with torch.no_grad():
        for name, p in clip_model.visual.named_parameters():
            if name in ckpt["visual_bias"]:
                p.copy_(ckpt["visual_bias"][name].to(device=p.device, dtype=p.dtype))

    softmax = nn.Softmax(dim=-1)
    y_true, prob_fake = [], []
    # Per-video: accumuliamo prob per path->video
    per_video_probs = defaultdict(list)
    per_video_labels = {}

    for imgs, labels, paths in tqdm(test_loader, desc="Testing (frames)"):
        imgs = imgs.to(DEVICE)
        with autocast_ctx():
            logits = predict_batch(clip_model, head, text_bank, imgs, alpha_sup=0.7)
            probs = softmax(logits)  # [B, 2]
        batch_prob_fake = probs[:, 1].detach().cpu().numpy()

        # frame-level accumulo
        y_true += list(labels)
        prob_fake += batch_prob_fake.tolist()

        # video key (gruppo) -> uso la cartella padre come id video
        for pth, lab, pr in zip(paths, labels, batch_prob_fake):
            p = Path(pth)
            # id video = directory padre del frame
            video_id = str(p.parent)
            # tieni massimo 32 frame per video (come richiesto)
            if len(per_video_probs[video_id]) < 32:
                per_video_probs[video_id].append(float(pr))
            # cattura l'etichetta (assumo coerente dentro il video)
            if video_id not in per_video_labels:
                per_video_labels[video_id] = int(lab)

    # ---------- METRICHE GLOBALI (per-FRAME) ----------
    y_true = np.array(y_true, dtype=int)
    prob_fake_arr = np.array(prob_fake, dtype=float)
    y_pred = (prob_fake_arr >= 0.5).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / max(1, len(y_true))
    auc_roc   = roc_auc_score(y_true, prob_fake_arr) if len(np.unique(y_true)) > 1 else float("nan")

    print("===== METRICHE GLOBALI (per-FRAME) =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC-ROC  : {auc_roc:.4f}" if not np.isnan(auc_roc) else "AUC-ROC  : n/a (classi non bilanciate nel test)")
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}  N={len(y_true)}")

    # Salva CSV frame-level
    os.makedirs(os.path.dirname(RESULTS_CSV_METRICS), exist_ok=True)
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "N": int(len(y_true)),
        "threshold": 0.5
    }])
    metrics_df.to_csv(RESULTS_CSV_METRICS, index=False)
    print(f"✓ Salvato CSV metriche globali (frame) in {RESULTS_CSV_METRICS}")

    # ---------- METRICHE GLOBALI (per-VIDEO, media su 32 frame) ----------
    video_ids = sorted(per_video_probs.keys())
    if len(video_ids) == 0:
        print("Nessun gruppo video trovato per il test per-VIDEO.")
    else:
        y_true_vid, prob_fake_vid = [], []
        for vid in video_ids:
            probs_v = per_video_probs[vid]
            # media (su fino a 32 frame già limitati sopra)
            avg_prob = float(np.mean(probs_v)) if len(probs_v) > 0 else 0.0
            prob_fake_vid.append(avg_prob)
            y_true_vid.append(per_video_labels.get(vid, 0))  # default 0 se mancante (non dovrebbe)

        y_true_vid = np.array(y_true_vid, dtype=int)
        prob_fake_vid = np.array(prob_fake_vid, dtype=float)
        y_pred_vid = (prob_fake_vid >= VIDEO_DECISION_THRESHOLD).astype(int)

        TPv = int(((y_pred_vid == 1) & (y_true_vid == 1)).sum())
        TNv = int(((y_pred_vid == 0) & (y_true_vid == 0)).sum())
        FPv = int(((y_pred_vid == 1) & (y_true_vid == 0)).sum())
        FNv = int(((y_pred_vid == 0) & (y_true_vid == 1)).sum())

        precision_v = TPv / (TPv + FPv + eps)
        recall_v    = TPv / (TPv + FNv + eps)
        f1_v        = 2 * precision_v * recall_v / (precision_v + recall_v + eps)
        accuracy_v  = (TPv + TNv) / max(1, len(y_true_vid))
        auc_roc_v   = roc_auc_score(y_true_vid, prob_fake_vid) if len(np.unique(y_true_vid)) > 1 else float("nan")

        print("===== METRICHE GLOBALI (per-VIDEO, avg 32 frame) =====")
        print(f"Videos   : {len(video_ids)}")
        print(f"Accuracy : {accuracy_v:.4f}")
        print(f"Precision: {precision_v:.4f}")
        print(f"Recall   : {recall_v:.4f}")
        print(f"F1       : {f1_v:.4f}")
        print(f"AUC-ROC  : {auc_roc_v:.4f}" if not np.isnan(auc_roc_v) else "AUC-ROC  : n/a (classi non bilanciate nel test)")
        print(f"TP={TPv}  TN={TNv}  FP={FPv}  FN={FNv}")

        # Salva CSV video-level
        os.makedirs(os.path.dirname(RESULTS_CSV_METRICS_VIDEO), exist_ok=True)
        metrics_vid_df = pd.DataFrame([{
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
            "threshold": VIDEO_DECISION_THRESHOLD,
            "frames_per_video_avg": 32
        }])
        metrics_vid_df.to_csv(RESULTS_CSV_METRICS_VIDEO, index=False)
        print(f"✓ Salvato CSV metriche globali (video) in {RESULTS_CSV_METRICS_VIDEO}")

def main():
    train_and_eval()

if __name__ == "__main__":
    main()
