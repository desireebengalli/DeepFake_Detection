# =========================================================
# Valutazione per-VIDEO con Cosine Classifier da checkpoint CLIP+Linear (robusto al tuo formato ckpt)
# =========================================================
import os, glob, math, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm  # <-- progress bar

# ------------- Config (adatta i percorsi alle tue cartelle) -------------
MODEL_NAME = "ViT-B/16"  # il tuo checkpoint sembra B/16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint singolo (quello che sovrascrivi a ogni epoch)
SAVE_PATH = "/home/giadapoloni/results/CLIP1_B16/clip_deepfake_FAST_T4.pt"

# (Opzionale) cartella con checkpoint per-epoch (*.pt)
CHECKPOINTS_DIR = ""  # lascia "" o None se non la usi

# Test set (come nel tuo script)
TEST_REAL_DIR = "/home/giadapoloni/C_validation/C_real"
TEST_FAKE_DIR = "/home/giadapoloni/C_validation/C_fake"
IMG_EXTS = {".jpg"}
TEST_BATCH_SIZE = 64

# (Opzionale) dove salvare i risultati per-VIDEO (CSV) quando si valuta SAVE_PATH
SAVE_CSV_PER_VIDEO = "/home/giadapoloni/results/CLIP2_B16_cosine/clip_test_results_celeb_videos_cosine.csv"

# Parametri Cosine Classifier / mixing testo
COSINE_SCALE = 20.0           # "temperature" dei logit coseno
USE_TEXT_MIXING = True       # True per combinare con banca di testo (come nel tuo script)
ALPHA_SUP = 0.7               # peso del ramo supervisionato se mixing attivo


# ======================= Dataset test =======================
class TestFrameDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items; self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        img = self.preprocess(img)
        return img, rec["label"], str(rec["path"])

def collect_test_items(real_dir, fake_dir):
    items = []
    for root, lab in [(real_dir, 0), (fake_dir, 1)]:
        root = Path(root)
        if not root.exists(): continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append({"path": p, "label": lab})
    return items

def build_test_loader(preprocess):
    test_items = collect_test_items(TEST_REAL_DIR, TEST_FAKE_DIR)
    assert len(test_items) > 0, "Nessuna immagine trovata nelle cartelle di test."
    ds = TestFrameDataset(test_items, preprocess)
    return DataLoader(ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True, drop_last=False)


# ================== Cosine Classifier & utils ==================
# ---- Helpers per caricare checkpoint in formati diversi e avvisi di compatibilità ----
def _strip_prefix(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Torch < 2.4: non supporta weights_only
        return torch.load(path, map_location=map_location)

def _first_2d_tensor(d):
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return k, v
    return None, None

def _suggest_model_from_visual(visual_sd, model_name):
    convw = visual_sd.get("conv1.weight") if isinstance(visual_sd, dict) else None
    if isinstance(convw, torch.Tensor) and convw.ndim == 4:
        patch = int(convw.shape[-1])
        expected = "ViT-B/16" if patch == 16 else ("ViT-B/32" if patch == 32 else None)
        if expected and expected != model_name:
            print(f"⚠️ Il checkpoint sembra {expected} (patch={patch}) ma MODEL_NAME è {model_name}. Valuta di impostare MODEL_NAME='{expected}'.")


def load_visual_and_head_from_ckpt(clip_model, ckpt_path, device):
    """
    Supporta il tuo formato ckpt:
    - root dict con chiavi: 'head' (dict), 'visual_bias' (dict), 'epoch'
    E anche i formati comuni:
    - { 'clip_visual': {...}, 'head': {...} }
    - { 'state_dict': {...} } con chiavi 'visual.*' e 'head.*' (anche 'module.*')
    Ritorna: W (tensor [C, D])
    """
    ckpt = _safe_torch_load(ckpt_path, map_location=device)

    # Scegli lo state_dict principale, se presente
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else {}
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # --- VISUAL ---
    visual_sd = None
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("clip_visual"), dict):
            visual_sd = ckpt["clip_visual"]
        elif isinstance(ckpt.get("visual_bias"), dict):
            # checkpoint di bias-tuning: aggiorna solo le chiavi presenti
            visual_sd = ckpt["visual_bias"]
        else:
            # Prova a estrarre da uno state_dict piatto
            tmp = _strip_prefix(sd, "visual.") if isinstance(sd, dict) else {}
            if tmp:
                visual_sd = tmp
    if visual_sd:
        _suggest_model_from_visual(visual_sd, MODEL_NAME)
        _ = clip_model.visual.load_state_dict(visual_sd, strict=False)
    else:
        print("(!) Nessun peso 'visual' nel ckpt: uso i pesi CLIP base.")

    # --- HEAD / LINEAR ---
    head_sd = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("head"), dict):
        head_sd = ckpt["head"]
    else:
        head_sd = _strip_prefix(sd, "head.") if isinstance(sd, dict) else None

    assert isinstance(head_sd, dict), "Checkpoint privo di 'head' con pesi della testa."

    # prova nomi comuni
    W = None
    for name in ["fc.weight", "classifier.weight", "linear.weight", "weight"]:
        w = head_sd.get(name)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            W = w
            break
    if W is None:
        k, w = _first_2d_tensor(head_sd)  # primo tensore 2D plausibile
        assert w is not None, "Impossibile trovare i pesi 2D della testa nel checkpoint."
        print(f"(i) Uso '{k}' come pesi della testa lineare.")
        W = w

    return W


class CosineClassifier(nn.Module):
    """
    Classifier a prototipi: logits = scale * cos(x, W)
    - Niente bias (standard nei cosine classifier)
    - W: [num_classes, D] normalizzato L2 (per riga)
    """
    def __init__(self, weight_init, scale=20.0, learnable_scale=False):
        super().__init__()
        w = F.normalize(weight_init.float(), dim=1)
        self.weight = nn.Parameter(w, requires_grad=False)  # inferenza
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        x = F.normalize(x.float(), dim=1)          # [B, D]
        logits = self.scale * (x @ self.weight.t())  # [B, C]
        return logits

@torch.no_grad()
def predict_batch_cosine(clip_model, cos_cls, imgs, text_bank=None, alpha_sup=1.0):
    """
    - alpha_sup=1.0 -> SOLO cosine classifier
    - alpha_sup in (0,1) -> mixing con banca di testo (come nel tuo script)
    """
    z = F.normalize(clip_model.encode_image(imgs), dim=-1)        # [B, D]
    logits_sup = cos_cls(z)                                       # [B, 2]
    if (text_bank is None) or (alpha_sup >= 0.999):
        return logits_sup
    logits_txt = clip_model.logit_scale.exp() * (z @ text_bank.t())
    logits = alpha_sup * logits_sup + (1.0 - alpha_sup) * logits_txt
    return logits

def build_text_bank(clip_model, device):
    import clip
    real_prompts = [
        "a real photo of a human face",
        "authentic portrait, not edited",
        "unaltered photograph of a person"
    ]
    fake_prompts = [
        "deepfake, AI-generated face",
        "synthetic portrait, computer-generated",
        "manipulated face image, fake"
    ]
    with torch.no_grad():
        txt_r = F.normalize(clip_model.encode_text(clip.tokenize(real_prompts).to(device)), dim=-1).mean(0, keepdim=True)
        txt_f = F.normalize(clip_model.encode_text(clip.tokenize(fake_prompts).to(device)), dim=-1).mean(0, keepdim=True)
    return torch.cat([txt_r, txt_f], 0)  # [2, D]


# ===================== Metriche & helper =====================
def _confusion_2x2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def _prec_rec_f1(tp, tn, fp, fn):
    # classe positiva = 1 (FAKE)
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos   = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0
    # classe 0 (REAL)
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg   = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0
    macro_p  = 0.5 * (prec_pos + prec_neg)
    macro_r  = 0.5 * (rec_pos + rec_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return {
        "pos": (prec_pos, rec_pos, f1_pos),
        "neg": (prec_neg, rec_neg, f1_neg),
        "macro": (macro_p, macro_r, macro_f1)
    }

def _roc_auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        P = (y_true == 1).sum()
        N = (y_true == 0).sum()
        if P == 0 or N == 0:
            return float("nan")
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == 0)
        tpr = tps / P
        fpr = fps / N
        return float(np.trapz(tpr, fpr))

def _safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ===================== Valutazione per-VIDEO =====================
@torch.no_grad()
def eval_with_checkpoint(ckpt_path, use_text_mixing=True, alpha_sup=0.7, scale=20.0,
                         save_csv_per_video=None):
    """
    Valutazione per-VIDEO:
      1) calcolo prob_fake per frame con cosine-classifier (opzionale mixing testo)
      2) aggregazione per-VIDEO a LOGIT (mean logit) -> prob (expit)
      3) metriche su video: accuracy, precision/recall/F1 (per classe e macro), AUROC
    """
    print(f"\n==> Carico checkpoint: {ckpt_path}")
    import clip
    clip_model, clip_preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad = False
    if hasattr(clip_model, "logit_scale"): clip_model.logit_scale.requires_grad = False

    # Carico i pesi del visual (bias tunati) e costruisco il cosine-classifier dai pesi della LinearHead
    W = load_visual_and_head_from_ckpt(clip_model, ckpt_path, DEVICE)  # [C, D]
    cos_cls = CosineClassifier(W.to(DEVICE), scale=scale, learnable_scale=False).to(DEVICE)

    # Banca di testo opzionale per mixing
    text_bank = build_text_bank(clip_model, DEVICE) if use_text_mixing else None

    # Loader test (teniamo i path per risalire al video)
    test_loader = build_test_loader(clip_preprocess)
    softmax = nn.Softmax(dim=-1)

    paths, y_true_frames, prob_fake_frames = [], [], []

    # --- Progress bar qui ---
    for imgs, labels, fpaths in tqdm(test_loader, desc="Inferenza (frames)", unit="batch"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = predict_batch_cosine(clip_model, cos_cls, imgs,
                                          text_bank=text_bank,
                                          alpha_sup=alpha_sup if use_text_mixing else 1.0)
            probs = softmax(logits)  # [:, 1] = prob fake
        prob_fake_frames.extend(probs[:, 1].detach().cpu().tolist())
        y_true_frames.extend(np.asarray(labels).tolist())
        paths.extend(list(fpaths))

    # DataFrame frame-level
    df_frames = pd.DataFrame({
        "path": paths,
        "gt_label": np.array(y_true_frames, dtype=int),
        "prob_fake": np.array(prob_fake_frames, dtype=float),
    })

    # video_id = nome della cartella padre del frame (come nel tuo script)
    df_frames["video_id"] = df_frames["path"].apply(lambda p: Path(p).parent.name)

    # Aggregazione per-VIDEO a LOGIT
    df_frames["logit_fake"] = _safe_logit(df_frames["prob_fake"].values)
    def _agg_video(g):
        mean_logit = float(np.mean(np.clip(g["logit_fake"].to_numpy(), -8.0, 8.0)))
        prob_mean = 1.0 / (1.0 + np.exp(-mean_logit))
        pred = int(prob_mean >= 0.5)
        gt = int(round(g["gt_label"].mean()))
        return pd.Series({
            "gt_label": gt,
            "pred_label": pred,
            "prob_fake_mean": prob_mean,
            "n_frames": int(len(g))
        })

    df_videos = df_frames.groupby("video_id", as_index=False).apply(_agg_video).reset_index(drop=True)

    # Metriche per-VIDEO
    y_true_v = df_videos["gt_label"].to_numpy(dtype=int)
    y_pred_v = df_videos["pred_label"].to_numpy(dtype=int)
    prob_mean_v = df_videos["prob_fake_mean"].to_numpy(dtype=float)

    acc_video = float((y_true_v == y_pred_v).mean())
    tp_v, tn_v, fp_v, fn_v = _confusion_2x2(y_true_v, y_pred_v)
    pr_v = _prec_rec_f1(tp_v, tn_v, fp_v, fn_v)
    auroc_v = _roc_auc(y_true_v, prob_mean_v)

    print("\n=== Metriche per-VIDEO ===")
    print(f"Accuracy           : {acc_video:.4f}  (N_videos={len(df_videos)})")
    print(f"Classe REAL (0)    : Precision={pr_v['neg'][0]:.4f}  Recall={pr_v['neg'][1]:.4f}  F1={pr_v['neg'][2]:.4f}")
    print(f"Classe FAKE (1)    : Precision={pr_v['pos'][0]:.4f}  Recall={pr_v['pos'][1]:.4f}  F1={pr_v['pos'][2]:.4f}")
    print(f"Macro (0/1)        : Precision={pr_v['macro'][0]:.4f}  Recall={pr_v['macro'][1]:.4f}  F1={pr_v['macro'][2]:.4f}")
    print(f"AUROC (pos=FAKE)   : {auroc_v:.4f}")
    print(f"Confusion [[TN, FP],[FN, TP]] = [[{tn_v}, {fp_v}], [{fn_v}, {tp_v}]]")

    # (Opzionale) salva CSV per-VIDEO
    if save_csv_per_video:
        Path(save_csv_per_video).parent.mkdir(parents=True, exist_ok=True)
        df_videos.to_csv(save_csv_per_video, index=False)
        print(f"✓ Salvato CSV per-VIDEO: {save_csv_per_video}")

    return {
        "video": {
            "n_videos": int(len(df_videos)),
            "accuracy": acc_video,
            "precision_real": pr_v['neg'][0], "recall_real": pr_v['neg'][1], "f1_real": pr_v['neg'][2],
            "precision_fake": pr_v['pos'][0], "recall_fake": pr_v['pos'][1], "f1_fake": pr_v['pos'][2],
            "precision_macro": pr_v['macro'][0], "recall_macro": pr_v['macro'][1], "f1_macro": pr_v['macro'][2],
            "auroc": auroc_v,
            "confusion": {"tn": tn_v, "fp": fp_v, "fn": fn_v, "tp": tp_v},
        },
        "df_videos": df_videos
    }


# ===================== Runner =====================
def run_all():
    results = {}

    # 1) Valuta il checkpoint singolo SAVE_PATH (e salva CSV se vuoi)
    if SAVE_PATH and os.path.isfile(SAVE_PATH):
        metrics = eval_with_checkpoint(
            SAVE_PATH,
            use_text_mixing=USE_TEXT_MIXING,
            alpha_sup=ALPHA_SUP,
            scale=COSINE_SCALE,
            save_csv_per_video=SAVE_CSV_PER_VIDEO
        )
        results["single_ckpt"] = {"path": SAVE_PATH, "metrics": metrics["video"]}

    # 2) (Opzionale) valuta tutti i checkpoint in CHECKPOINTS_DIR
    if CHECKPOINTS_DIR and os.path.isdir(CHECKPOINTS_DIR):
        ckpts = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pt")))
        if len(ckpts) > 0:
            results["dir_ckpts"] = []
            for ck in ckpts:
                metrics = eval_with_checkpoint(
                    ck,
                    use_text_mixing=USE_TEXT_MIXING,
                    alpha_sup=ALPHA_SUP,
                    scale=COSINE_SCALE,
                    save_csv_per_video=None  # evita di sovrascrivere; gestisci tu se vuoi
                )
                results["dir_ckpts"].append({"path": ck, "metrics": metrics["video"]})
        else:
            print(f"(Nessun .pt trovato in {CHECKPOINTS_DIR})")

    if not results:
        print("Nessun checkpoint trovato. Controlla SAVE_PATH / CHECKPOINTS_DIR.")
    return results


if __name__ == "__main__":
    run_all()
