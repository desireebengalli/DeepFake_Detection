# eval_from_ckpt_limited_macroF1.py
import os, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import clip
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef

# ================== CONFIG ==================
MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"

# Allinea questo percorso al file realmente salvato in train
# (se in train hai salvato in /home/giadapoloni/results/clip_deepfake_FAST_T4.pt,
#  copia il file nella cartella TAKE_EPOCH, oppure cambia CKPT_PATH direttamente)
TAKE_EPOCH   = "/home/giadapoloni/results/CLIP1_B16"
RESULTS_DIR  = "/home/giadapoloni/results/PROVA"

CKPT_PATH = os.path.join(TAKE_EPOCH, "clip_deepfake_FAST_T4.pt")
OUT_CSV_SWEEP  = os.path.join(RESULTS_DIR, "eval_macroF1_metrics.csv")

BATCH_TEST  = 64
NUM_WORKERS = 4
ALPHA_SUP = 0.70   # fusione sup/testo

# ================== UTILITIES ==================
IMG_EXTS = {".jpg", ".jpeg", ".png"}  # estensioni comuni

class TestFrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items; self.preprocess = preprocess
    def __len__(self): return len(self.items)
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
    return items

@torch.no_grad()
def build_text_bank(clip_model, device):
    real_prompts = [
        "a real photo of a human face",
        "authentic portrait, not edited",
        "unaltered photograph of a person"]
    fake_prompts = [
        "deepfake, AI-generated face",
        "synthetic portrait, computer-generated",
        "manipulated face image, fake"]
    txt_r = F.normalize(clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    txt_f = F.normalize(clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)

class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x): return self.fc(x)

@torch.no_grad()
def predict_proba_frames(clip_model, head, text_bank, loader, alpha_sup=0.7):
    softmax = nn.Softmax(dim=-1)
    y_true, prob_fake = [], []
    clip_model.eval(); head.eval()
    for imgs, labels, _paths in tqdm(loader, desc="Inference"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        z = F.normalize(clip_model.encode_image(imgs), dim=-1)
        logits_sup = head(z)
        logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
        logits = alpha_sup * logits_sup + (1 - alpha_sup) * logits_txt
        probs = softmax(logits)[:, 1]
        y_true.extend(labels.tolist())
        prob_fake.extend(probs.detach().cpu().tolist())
    return np.array(y_true, int), np.array(prob_fake, float)

def metrics_from_probs(y_true, prob_fake, thr):
    y_pred = (prob_fake >= thr).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    eps = 1e-12
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / max(1, len(y_true))
    try:
        auc_roc = roc_auc_score(y_true, prob_fake) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc_roc = float("nan")
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN, "N": int(len(y_true))
    }

# ========== MACRO-F1 ==========
def classwise_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn = ((y_pred==0)&(y_true==0)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()
    tp = ((y_pred==1)&(y_true==1)).sum()
    tpr = tp / max(1, tp+fn)
    tnr = tn / max(1, tn+fp)
    bacc = 0.5*(tpr+tnr)
    gmean = (tpr*tnr) ** 0.5
    return {
        "prec_real": float(p[0]), "rec_real": float(r[0]), "f1_real": float(f1[0]),
        "prec_fake": float(p[1]), "rec_fake": float(r[1]), "f1_fake": float(f1[1]),
        "macro_f1": float((f1[0]+f1[1])/2),
        "bacc": float(bacc), "gmean": float(gmean), "mcc": float(mcc)
    }

def pick_threshold_macro_f1(y_true, prob_fake, grid=None):
    if grid is None: grid = np.linspace(0.01, 0.99, 197)
    best = (-1.0, 0.5)
    for t in grid:
        y_pred = (prob_fake >= t).astype(int)
        m = classwise_metrics(y_true, y_pred)
        if m["macro_f1"] > best[0]:
            best = (m["macro_f1"], t)
    return best[1], best[0]

# ========== REMAP CHIAVI BIAS ==========
def _remap_visual_bias_keys(bias_sd, visual_module):
    """
    Rinomina/normalizza le chiavi dei bias MLP per adattarle al CLIP corrente.
    Converte tra 'mlp.0.bias'/'mlp.2.bias', 'mlp.c_fc.bias'/'mlp.c_proj.bias',
    ed eventuali 'mlp.fc1.bias'/'mlp.fc2.bias'. Filtra sulle chiavi esistenti.
    """
    if not isinstance(bias_sd, dict) or len(bias_sd) == 0:
        return {}

    remapped = {}
    for k, v in bias_sd.items():
        k2 = k
        # indici -> nomi
        k2 = k2.replace(".mlp.0.bias", ".mlp.c_fc.bias")
        k2 = k2.replace(".mlp.2.bias", ".mlp.c_proj.bias")
        # alias comuni
        k2 = k2.replace(".mlp.fc1.bias", ".mlp.c_fc.bias")
        k2 = k2.replace(".mlp.fc2.bias", ".mlp.c_proj.bias")
        remapped[k2] = v

    target_keys = set(visual_module.state_dict().keys())
    # se il modello usa indici, mappiamo anche nomi -> indici
    if not any(k.endswith(".mlp.c_fc.bias") or k.endswith(".mlp.c_proj.bias") for k in target_keys):
        tmp = {}
        for k, v in remapped.items():
            k3 = k.replace(".mlp.c_fc.bias", ".mlp.0.bias").replace(".mlp.c_proj.bias", ".mlp.2.bias")
            tmp[k3] = v
        remapped = tmp

    # filtra alle sole chiavi presenti
    remapped = {k: v for k, v in remapped.items() if k in target_keys}
    return remapped

# ================== MAIN ==================
def main():
    # 1) Carica CLIP base
    clip_model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    clip_model.float()  # FP32 coerente con training

    embed_dim = clip_model.text_projection.shape[1]
    head = LinearHead(embed_dim, 2).to(DEVICE)

    # 2) Carica checkpoint (head + bias visual)
    assert os.path.exists(CKPT_PATH), f"Checkpoint non trovato: {CKPT_PATH}"
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Head
    head.load_state_dict(ckpt["head"])

    # Visual bias con remap chiavi
    vb_raw = ckpt.get("visual_bias", {})
    vb = _remap_visual_bias_keys(vb_raw, clip_model.visual)
    loaded = clip_model.visual.load_state_dict(vb, strict=False)

    print("=== CKPT INFO ===")
    print("ckpt epoch:", ckpt.get("epoch"))
    print("bias nel ckpt (raw):", len(vb_raw), " | bias usati dopo remap:", len(vb))
    print("missing_keys (prime 10):", loaded.missing_keys[:10])
    print("unexpected_keys (prime 10):", loaded.unexpected_keys[:10])

    if len(vb) == 0:
        print("(!) Nessun bias del visual caricato: stai usando i pesi CLIP base per il visual. "
              "Controlla che il file ckpt sia quello giusto e che il training salvi i bias con chiavi compatibili.")

    # 3) Loader test
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    assert len(test_items) > 0, f"Nessun frame trovato nel set di test: {TEST_REAL_DIR} / {TEST_FAKE_DIR}"
    test_ds = TestFrameDataset(test_items, preprocess)
    loader = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=(NUM_WORKERS > 0),
                        prefetch_factor=(2 if NUM_WORKERS > 0 else None))

    # 4) Text bank e inference
    text_bank = build_text_bank(clip_model, DEVICE)
    y_true, prob_fake = predict_proba_frames(clip_model, head, text_bank, loader, alpha_sup=ALPHA_SUP)

    # diagnostica sugli score
    pf = np.array(prob_fake, dtype=float)
    print(f"prob_fake min/mean/max/std: {pf.min():.4f} / {pf.mean():.4f} / {pf.max():.4f} / {pf.std():.6f}")

    # 5) Scelta soglia per macro-F1 + metriche
    thr_mf1, mf1 = pick_threshold_macro_f1(y_true, prob_fake)
    met = metrics_from_probs(y_true, prob_fake, thr_mf1)
    y_pred = (prob_fake >= thr_mf1).astype(int)
    met.update(classwise_metrics(y_true, y_pred))
    met["which_threshold"] = "MacroF1_opt"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([met]).to_csv(OUT_CSV_SWEEP, index=False)

    print(f"MacroF1_opt: thr={thr_mf1:.3f} Acc={met['accuracy']:.4f} "
          f"BAcc={met['bacc']:.4f} MacroF1={met['macro_f1']:.4f} MCC={met['mcc']:.4f} "
          f"FN={met['FN']} FP={met['FP']}  "
          f"Prec_fake={met['prec_fake']:.3f} Rec_fake={met['rec_fake']:.3f}")
    print(f"✓ Salvato CSV in: {OUT_CSV_SWEEP}")

if __name__ == "__main__":
    main()
