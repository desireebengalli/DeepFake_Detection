# ============================
# FAST TRAIN (frames preprocessati) - ViT-B/16
# Cosine head + DFF-Adapter "tipo DINO" *dentro CLIP* + CLIP text logits
# Cosine LR con warmup, gradient accumulation, loss ibrida
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
MODEL_NAME = "ViT-B/16"
EPOCHS = 1
BATCH_SIZE = 16
ACCUM_STEPS = 8
NUM_WORKERS = 7

LR_HEAD = 8e-4         # head autenticità
LR_ADAPTER = 4e-4      # DFF-Adapter
LR_FORGERY = 4e-4      # forgery-type head
WEIGHT_DECAY_HEAD = 1e-5
WEIGHT_DECAY_ADAPTER = 0.0
WEIGHT_DECAY_FORGERY = 1e-5
WARMUP_RATIO = 0.06

USE_WEIGHTED_SAMPLER = False
ALPHA_SUP_EPOCH_SCHEDULE = [(1, 0.60), (4, 0.65), (7, 0.70)]
VIDEO_DECISION_THRESHOLD = 0.5

# pesi per multitask authenticity / forgery-type
LAMBDA_AUTH = 1.0
LAMBDA_FORGERY = 0

SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CUDA = (DEVICE == "cuda")

DATA_DIR = "/home/giadapoloni/preprocessed_frames"
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
RESULTS_DIR = "/home/default/results/CLIP1_B16_cosine_DFF"
RESULTS_CSV_METRICS = os.path.join(RESULTS_DIR, "clip_test_metrics_global_v2.csv")
RESULTS_CSV_METRICS_VIDEO = os.path.join(RESULTS_DIR, "clip_test_metrics_video_v2.csv")
SAVE_PATH = os.path.join(RESULTS_DIR, "clip_cosine_dff_v2.pt")
print("USO RESULTS_DIR =", RESULTS_DIR, flush=True)

amp_dtype = torch.float16
scaler = torch.amp.GradScaler(device='cuda', enabled=IS_CUDA)

def autocast_ctx():
    if IS_CUDA:
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    class _NullCtx:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    return _NullCtx()

IMG_EXTS = {".jpg"}

# ==== DFF-ADAPTER: definizione metodi di forgery ====
FORGERY_METHODS = ["deepfakes", "face2face", "faceswap", "neuraltextures"]
FORGERY_TO_ID = {m: i for i, m in enumerate(FORGERY_METHODS)}
NUM_FORGERY_TYPES = len(FORGERY_METHODS)

# ---------- Modelli ----------

class CosineHead(nn.Module):
    def __init__(self, in_dim, n_classes=2, init_scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, n_classes))
        nn.init.normal_(self.W, std=0.02)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)), requires_grad=True)

    def forward(self, z):
        Wn = F.normalize(self.W, dim=0)        # [D, C]
        logits = self.scale * (z @ Wn)         # [B, C]
        return logits


# ===== DFF-ADAPTER: LAYER + TRASFORMER + VISUAL WRAPPER =====

class DFFAdapterLayer(nn.Module):
    """
    Singolo layer DFF-Adapter:
    - LoRA multi-head + pool di esperti condiviso
    - branche: 'share', 'auth', 'ftc'
    Opera su un embedding [B, D] (tipicamente CLS di un blocco ViT).
    """
    def __init__(
        self,
        dim: int,
        rank: int = 16,
        num_heads: int = 4,
        num_experts: int = 6,
        beta: float = 0.5
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim deve essere divisibile per num_heads"
        self.dim = dim
        self.h = num_heads
        self.d_h = dim // num_heads
        self.r_h = max(1, rank // num_heads)
        self.num_experts = num_experts
        self.beta = beta

        # pool di LoRA experts condiviso tra share/auth/ftc
        self.A = nn.Parameter(torch.randn(num_experts, self.d_h, self.r_h) * 0.02)
        self.B = nn.Parameter(torch.randn(num_experts, self.r_h, self.d_h) * 0.02)

        # router logits: [head, num_experts] per share/auth/ftc
        self.Z_share = nn.Parameter(torch.zeros(self.h, num_experts))
        self.Z_auth  = nn.Parameter(torch.zeros(self.h, num_experts))
        self.Z_ftc   = nn.Parameter(torch.zeros(self.h, num_experts))

        nn.init.normal_(self.Z_share, std=0.02)
        nn.init.normal_(self.Z_auth,  std=0.02)
        nn.init.normal_(self.Z_ftc,   std=0.02)

    def _route(self, z_heads, Z_router, topk: int = 3):
        """
        z_heads: [B, h, d_h]
        Z_router: [h, num_experts]
        """
        B, h, d_h = z_heads.shape
        device = z_heads.device
        out = torch.zeros_like(z_heads)

        probs = F.softmax(Z_router, dim=-1)  # [h, num_experts]
        top_vals, top_idx = torch.topk(probs, k=min(topk, self.num_experts), dim=-1)  # [h, k]

        for head in range(h):
            z_h = z_heads[:, head, :]  # [B, d_h]
            idx = top_idx[head]        # [k]
            w = top_vals[head]         # [k]
            w = w / (w.sum() + 1e-8)

            delta = torch.zeros(B, d_h, device=device, dtype=z_heads.dtype)
            for eid, wj in zip(idx, w):
                A_j = self.A[eid]      # [d_h, r_h]
                B_j = self.B[eid]      # [r_h, d_h]
                delta = delta + wj * (z_h @ A_j @ B_j)
            out[:, head, :] = delta

        return out

    def forward(self, z, task: str = "auth"):
        """
        z: [B, D] normalizzato
        task: 'auth' o 'ftc'
        """
        B, D = z.shape
        z_heads = z.view(B, self.h, self.d_h)

        delta_share = self._route(z_heads, self.Z_share)
        if task == "auth":
            delta_task = self._route(z_heads, self.Z_auth)
        elif task == "ftc":
            delta_task = self._route(z_heads, self.Z_ftc)
        else:
            raise ValueError(f"Task sconosciuto per DFFAdapterLayer: {task}")

        delta = delta_share + delta_task
        delta = delta.view(B, D)
        return z + self.beta * delta


class TransformerWithDFF(nn.Module):
    """
    Wrap del transformer di CLIP, con un DFFAdapterLayer per ogni resblock.
    Opera su sequenza [seq, B, D]. Applica l'adapter sul token CLS (posizione 0).
    """
    def __init__(self, base_transformer: nn.Module,
                 dim: int,
                 rank: int = 16,
                 num_heads: int = 4,
                 num_experts: int = 6,
                 beta: float = 0.5):
        super().__init__()
        # riutilizziamo i resblocks originali (con pesi congelati)
        self.resblocks = base_transformer.resblocks
        num_layers = len(self.resblocks)
        self.adapters = nn.ModuleList([
            DFFAdapterLayer(dim, rank, num_heads, num_experts, beta)
            for _ in range(num_layers)
        ])

    def forward(self, x, task: str = "auth"):
        """
        x: [seq, B, D]
        """
        for i, block in enumerate(self.resblocks):
            x = block(x)  # forward standard CLIP
            # CLS token = posizione 0 lungo seq
            cls = x[0]   # [B, D]
            cls = self.adapters[i](cls, task=task)
            x = x.clone()
            x[0] = cls
        return x


class VisualWithDFF(nn.Module):
    """
    Wrapper del VisionTransformer di CLIP:
    - riusa conv1, class_embedding, positional_embedding, ln_pre, ln_post, proj
    - sostituisce il transformer con TransformerWithDFF(task-aware)
    - forward(x, task) -> embedding nello stesso spazio di CLIP.encode_image
    """
    def __init__(self, base_visual: nn.Module,
                 rank: int = 16,
                 num_heads: int = 4,
                 num_experts: int = 6,
                 beta: float = 0.5):
        super().__init__()
        # copiamo i moduli "strutturali" del visual originale
        self.conv1 = base_visual.conv1
        self.class_embedding = base_visual.class_embedding
        self.positional_embedding = base_visual.positional_embedding
        self.ln_pre = base_visual.ln_pre
        self.ln_post = base_visual.ln_post
        self.proj = base_visual.proj
        # attributi utility
        self.input_resolution = getattr(base_visual, "input_resolution", None)
        # patch size ricavata da conv1 (kernel quadrato)
        self.patch_size = base_visual.conv1.kernel_size[0]

        # dimensione canale (width) = dim delle pos embedding
        dim = self.positional_embedding.shape[-1]

        # trasformer con DFF
        self.transformer_dff = TransformerWithDFF(
            base_transformer=base_visual.transformer,
            dim=dim,
            rank=rank,
            num_heads=num_heads,
            num_experts=num_experts,
            beta=beta
        )

    def forward(self, x, task: str = "auth"):
        # forward copiato da VisionTransformer di CLIP, ma con transformer_dff(x, task)
        x = self.conv1(x)  # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)                     # [B, HW, C]

        cls_emb = self.class_embedding.to(x.dtype)
        cls_tokens = cls_emb + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)      # [B, 1+HW, C]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [seq, B, C]
        x = self.transformer_dff(x, task=task)
        x = x.permute(1, 0, 2)  # [B, seq, C]

        x = self.ln_post(x[:, 0, :])  # CLS

        if self.proj is not None:
            x = x @ self.proj

        return x  # [B, D_clip]


# ---------- Funzioni generiche ----------

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
    # real
    real_dir = root / "real_ff"
    if real_dir.exists():
        for vid in sorted([d for d in real_dir.iterdir() if d.is_dir()]):
            for fp in sorted([p for p in vid.iterdir() if p.suffix.lower() in IMG_EXTS]):
                items.append({"path": fp, "label": 0, "forgery_type": -1})

    # fake: sottocartelle per metodo
    fake_dir = root / "fake_ff"
    for m in FORGERY_METHODS:
        mdir = fake_dir / m
        if not mdir.exists():
            continue
        forgery_id = FORGERY_TO_ID[m]
        for vid in sorted([d for d in mdir.iterdir() if d.is_dir()]):
            for fp in sorted([p for p in vid.iterdir() if p.suffix.lower() in IMG_EXTS]):
                items.append({"path": fp, "label": 1, "forgery_type": forgery_id})
    return items


def collect_test_items(real_dir, fake_dir):
    items = []
    for root, label in [(real_dir, 0), (fake_dir, 1)]:
        rroot = Path(root)
        if rroot.exists():
            for p in sorted(rroot.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    # forgery_type non usato nel test -> metto -1
                    items.append({"path": p, "label": label, "forgery_type": -1})
    return items


class FrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        x = self.preprocess(img)
        y = rec["label"]
        ft = rec["forgery_type"]
        return x, y, ft


class TestFrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        x = self.preprocess(img)
        return x, rec["label"], str(rec["path"])


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


def build_optimizer_and_scheduler(head, visual_dff, forgery_head, total_steps):
    params = [
        {"params": head.parameters(),        "lr": LR_HEAD,    "weight_decay": WEIGHT_DECAY_HEAD},
        {"params": visual_dff.parameters(),  "lr": LR_ADAPTER, "weight_decay": WEIGHT_DECAY_ADAPTER},
        {"params": forgery_head.parameters(),"lr": LR_FORGERY, "weight_decay": WEIGHT_DECAY_FORGERY},
    ]
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
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
    real_prompts = ["a real photo of a human face",
                    "authentic portrait, not edited",
                    "unaltered photograph of a person"]
    fake_prompts = ["deepfake, AI-generated face",
                    "synthetic portrait, computer-generated",
                    "manipulated face image, fake"]
    txt_r = F.normalize(clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    txt_f = F.normalize(clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)


def alpha_for_epoch(epoch):
    a = ALPHA_SUP_EPOCH_SCHEDULE[0][1]
    for e, val in ALPHA_SUP_EPOCH_SCHEDULE:
        if epoch >= e: a = val
    return a


@torch.no_grad()
def predict_batch(clip_model, visual_dff, head, text_bank, imgs, alpha_sup=0.7):
    # ramo supervisionato: CLIP+adapter (auth)
    z_auth = F.normalize(visual_dff(imgs, task="auth"), dim=-1)
    logits_sup = head(z_auth)

    # ramo CLIP testo: embedding "pulito" di CLIP
    z_clip = F.normalize(clip_model.encode_image(imgs), dim=-1)
    logits_txt = clip_model.logit_scale.exp() * (z_clip @ text_bank.t())

    logits = alpha_sup * logits_sup + (1 - alpha_sup) * logits_txt
    return logits


# ---------- Training + Test ----------

def train_and_eval():
    def set_seed(s):
        random.seed(s); np.random.seed(s)
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    clip_model.float()

    # congela completamente CLIP (visivo + testo + logit_scale)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.logit_scale.requires_grad = False

    embed_dim = clip_model.text_projection.shape[1]
    text_bank = build_text_bank(clip_model, DEVICE)

    # VisualWithDFF: usa il visual di CLIP come base congelata
    visual_dff = VisualWithDFF(
        base_visual=clip_model.visual,
        rank=16,
        num_heads=4,
        num_experts=6,
        beta=0.5
    ).to(DEVICE)

    train_loader = build_train_loader(preprocess)
    test_loader = build_test_loader(preprocess)

    counts = Counter([it["label"] for it in collect_frames(DATA_DIR)])
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total/(2*counts[0]), total/(2*counts[1])],
        device=DEVICE, dtype=torch.float32
    )
    ce_auth = nn.CrossEntropyLoss(weight=class_weights if not USE_WEIGHTED_SAMPLER else None)
    ce_forg = nn.CrossEntropyLoss()  # forgery-type: assumiamo classi ok

    head = CosineHead(embed_dim, 2, init_scale=16.0).to(DEVICE)
    forgery_head = nn.Linear(embed_dim, NUM_FORGERY_TYPES).to(DEVICE)

    steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1)//ACCUM_STEPS
    total_steps = EPOCHS * steps_per_epoch
    optimizer, scheduler = build_optimizer_and_scheduler(head, visual_dff, forgery_head, total_steps)

    clip_model.eval()
    visual_dff.train()
    head.train()
    forgery_head.train()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        alpha_sup = alpha_for_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, y, ft) in enumerate(pbar, 1):
            imgs  = imgs.to(DEVICE, non_blocking=True)
            y     = torch.as_tensor(y,  device=DEVICE)
            ft_id = torch.as_tensor(ft, device=DEVICE)

            with autocast_ctx():
                # z per ramo CLIP "pulito" (testo)
                z_clip = F.normalize(clip_model.encode_image(imgs), dim=-1)

                # ramo autenticità + DFFAdapter (auth)
                z_auth = F.normalize(visual_dff(imgs, task="auth"), dim=-1)
                logits_sup = head(z_auth)
                logits_txt = clip_model.logit_scale.exp() * (z_clip @ text_bank.t())

                loss_auth_sup = ce_auth(logits_sup, y)
                loss_auth_txt = ce_auth(logits_txt, y)
                loss_contrast = supervised_contrastive_loss(z_auth, y)

                L_auth = 0.4 * loss_auth_sup + 0.4 * loss_auth_txt + 0.2 * loss_contrast

                # ramo forgery-type: solo frame fake, con task="ftc"
                fake_mask = (y == 1)
                # if fake_mask.any():
                #     imgs_fake = imgs[fake_mask]
                #     z_ftc = F.normalize(visual_dff(imgs_fake, task="ftc"), dim=-1)
                #     logits_forg = forgery_head(z_ftc)
                #     ft_targets = ft_id[fake_mask]  # 0..NUM_FORGERY_TYPES-1
                #     L_forg = ce_forg(logits_forg, ft_targets)
                # else:
                #     L_forg = torch.tensor(0.0, device=DEVICE, dtype=z_auth.dtype)

                loss = LAMBDA_AUTH * L_auth

            # backward con grad scaling + accumulation
            if IS_CUDA:
                scaler.scale(loss/ACCUM_STEPS).backward()
            else:
                (loss/ACCUM_STEPS).backward()

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
            pbar.set_postfix(loss=running/max(1, step))

        # checkpoint: head + visual_dff + forgery_head
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(
            {
                "head": head.state_dict(),
                "visual_dff": visual_dff.state_dict(),
                "forgery_head": forgery_head.state_dict(),
                "epoch": epoch,
                "model_name": MODEL_NAME,
            },
            SAVE_PATH
        )
        print(f"✓ Saved checkpoint epoch {epoch} -> {SAVE_PATH}")

    print("Training completato. Avvio test...")

    # ----- TEST -----
    clip_model.eval()
    visual_dff.eval()
    head.eval()
    forgery_head.eval()

    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    head.load_state_dict(ckpt["head"])
    visual_dff.load_state_dict(ckpt["visual_dff"])
    forgery_head.load_state_dict(ckpt["forgery_head"])

    softmax = nn.Softmax(dim=-1)
    y_true, prob_fake = [], []
    per_video_probs = defaultdict(list)
    per_video_labels = {}

    for imgs, labels, paths in tqdm(test_loader, desc="Testing (frames)"):
        imgs = imgs.to(DEVICE)
        labels = torch.as_tensor(labels)
        with autocast_ctx():
            logits = predict_batch(clip_model, visual_dff, head, text_bank, imgs, alpha_sup=0.7)
            probs = softmax(logits)
        batch_prob_fake = probs[:, 1].detach().cpu().numpy()

        y_true += list(labels.numpy())
        prob_fake += batch_prob_fake.tolist()

        for pth, lab, pr in zip(paths, labels, batch_prob_fake):
            p = Path(pth)
            video_id = str(p.parent)
            if len(per_video_probs[video_id]) < 32:
                per_video_probs[video_id].append(float(pr))
            if video_id not in per_video_labels:
                per_video_labels[video_id] = int(lab)

    # ----- metriche per-FRAME -----
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
    print(f"AUC-ROC  : {auc_roc:.4f}" if not np.isnan(auc_roc) else "AUC-ROC  : n/a")
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}  N={len(y_true)}")

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

    # ----- metriche per-VIDEO -----
    video_ids = sorted(per_video_probs.keys())
    if len(video_ids) == 0:
        print("Nessun gruppo video trovato per il test per-VIDEO.")
    else:
        y_true_vid, prob_fake_vid = [], []
        for vid in video_ids:
            probs_v = per_video_probs[vid]
            avg_prob = float(np.mean(probs_v)) if len(probs_v) > 0 else 0.0
            prob_fake_vid.append(avg_prob)
            y_true_vid.append(per_video_labels.get(vid, 0))

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
        print(f"AUC-ROC  : {auc_roc_v:.4f}" if not np.isnan(auc_roc_v) else "AUC-ROC  : n/a")
        print(f"TP={TPv}  TN={TNv}  FP={FPv}  FN={FNv}")

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
