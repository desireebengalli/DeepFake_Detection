#!/usr/bin/env python3
import os, glob, json, cv2, numpy as np
from pathlib import Path

# -----------------------
# Config
# -----------------------
BASE_FRAMES = "/home/giadapoloni/extracted_frames"
IN_ROOTS = [
    os.path.join(BASE_FRAMES, "real_ff"),
    os.path.join(BASE_FRAMES, "fake_ff", "deepfakes"),
    os.path.join(BASE_FRAMES, "fake_ff", "face2face"),
    os.path.join(BASE_FRAMES, "fake_ff", "faceswap"),
    os.path.join(BASE_FRAMES, "fake_ff", "neuraltextures"),
]
OUT_ROOT = Path("/home/giadapoloni/preprocessed_frames")
OUTPUT_SIZE = 512
MARGIN = 1.3
DET_THRESH = 0.3
JPEG_QUALITY = 92

# Threading “gentile” per decodifica/salvataggio
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# -----------------------
# Imports modello
# -----------------------
from insightface.app import FaceAnalysis
import onnxruntime as ort

# -----------------------
# Helpers
# -----------------------
REF_5PTS_112 = np.array([
    [38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
    [41.5493,92.3655],[70.7299,92.2041]
], dtype=np.float32)
REF_5PTS = REF_5PTS_112 * (OUTPUT_SIZE / 112.0)

JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
IMG_PATTERNS = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG","*.webp"]

def expand_square(bbox, margin, w, h):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    nx1, ny1 = int(cx - side / 2), int(cy - side / 2)
    nx2, ny2 = int(cx + side / 2), int(cy + side / 2)
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(w - 1, nx2), min(h - 1, ny2)
    return [nx1, ny1, nx2, ny2]

def align_by_5pts(img, kps, out_size=OUTPUT_SIZE):
    src = np.array(kps, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(src, REF_5PTS, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR)

def quality_scores(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean())
    }

def list_video_dirs(root):
    if not os.path.isdir(root):
        print(f"[WARN] Root non trovata: {root}")
        return []
    out = []
    for name in sorted(os.listdir(root)):
        vdir = os.path.join(root, name)
        if not os.path.isdir(vdir):
            continue
        has_imgs = any(glob.glob(os.path.join(vdir, p)) for p in IMG_PATTERNS)
        if has_imgs:
            out.append(vdir)
    return out

def _derive_root_tag(matched_root):
    parts = os.path.normpath(matched_root).split(os.sep)
    if parts[-1].lower() in ("real", "real_ff"):
        return "real_ff"
    if len(parts) >= 2 and parts[-2].lower() in ("fake", "fake_ff"):
        return f"fake_ff/{parts[-1].lower()}"
    return parts[-1].lower()

def out_dir_for(video_dir):
    matched_root = None
    for r in IN_ROOTS:
        rp = os.path.commonpath([os.path.abspath(video_dir), os.path.abspath(r)])
        if rp == os.path.abspath(r):
            matched_root = r
            break
    if matched_root is None:
        rel = os.path.basename(video_dir.rstrip('/'))
        root_tag = "misc"
    else:
        rel = os.path.relpath(video_dir, start=matched_root)
        root_tag = _derive_root_tag(matched_root)
    out_dir = OUT_ROOT / root_tag / rel
    return out_dir

# -----------------------
# Core: processa una cartella video
# -----------------------
def process_video_dir_with_app(video_dir, app):
    out_dir = out_dir_for(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    # Se esiste già il meta → skip intera cartella (idempotente)
    if meta_path.exists():
        print(f"[SKIP] {video_dir} (meta.json trovato)")
        return 0, 0

    frames = sorted([p for ext in IMG_PATTERNS for p in glob.glob(os.path.join(video_dir, ext))])
    print(f"[INFO] {video_dir} -> {len(frames)} frame")

    meta = {"video_id": os.path.basename(video_dir.rstrip('/')), "frames": []}
    ok = fail = 0

    for fp in frames:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            meta["frames"].append({"frame": os.path.basename(fp), "status": "imread_failed"})
            fail += 1
            continue

        faces = app.get(img)
        if not faces:
            meta["frames"].append({"frame": os.path.basename(fp), "status": "no_face"})
            fail += 1
            continue

        # volto più grande
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        x1,y1,x2,y2 = [int(v) for v in f.bbox]
        h,w = img.shape[:2]

        try:
            face_aligned = align_by_5pts(img, f.kps, out_size=OUTPUT_SIZE)
            bx1,by1,bx2,by2 = expand_square([x1,y1,x2,y2], MARGIN, w, h)
            context = cv2.resize(img[by1:by2, bx1:bx2],
                                 (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            bx1,by1,bx2,by2 = expand_square([x1,y1,x2,y2], MARGIN, w, h)
            face_aligned = cv2.resize(img[by1:by2, bx1:bx2],
                                      (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
            context = face_aligned

        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(str(out_dir / f"{base}_ctx.jpg"), context, JPEG_PARAMS)

        meta["frames"].append({
            "frame": os.path.basename(fp),
            "status": "ok",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "kps": np.asarray(f.kps).astype(float).round(2).tolist(),
            "files": {"face_ctx": f"{base}_ctx.jpg"},
            "quality": quality_scores(face_aligned)
        })
        ok += 1

    meta["summary"] = {"ok": ok, "fail": fail, "total": len(frames)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] {video_dir} -> ok:{ok} fail:{fail}")
    return ok, fail

# -----------------------
# Main
# -----------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Inizializza ONNXRuntime su GPU se disponibile
    providers = ort.get_available_providers()
    use_gpu = "CUDAExecutionProvider" in providers
    if not use_gpu:
        print("[WARN] CUDAExecutionProvider NON disponibile; userò CPUExecutionProvider.")
    else:
        print("[INFO] ONNX providers:", providers)

    # 2) Inizializza il detector (solo detection; stessa qualità)
    app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
    app.prepare(ctx_id=(0 if use_gpu else -1), det_thresh=DET_THRESH, det_size=(640, 640))
    _ = app.get(np.zeros((640,640,3), dtype=np.uint8))  # warmup

    # 3) Scansione cartelle e skip di quelle già processate
    all_vdirs = []
    for root in IN_ROOTS:
        vdirs = list_video_dirs(root)
        print(f"[ROOT] {root} -> {len(vdirs)} video")
        for vd in vdirs:
            out_dir = out_dir_for(vd)
            meta_path = out_dir / "meta.json"
            if meta_path.exists():
                print(f"[SKIP] {vd} (meta.json trovato)")
                continue
            all_vdirs.append(vd)

    if not all_vdirs:
        print("Niente da processare (tutto già fatto).")
        return

    # 4) Processo singolo (ottimo per GPU; evita contention sulla VRAM)
    total_ok = total_fail = total_vids = 0
    for vd in all_vdirs:
        ok, fail = process_video_dir_with_app(vd, app)
        total_ok += ok; total_fail += fail; total_vids += 1
        print(f"[ACCUM] video={total_vids} ok={total_ok} fail={total_fail}")

    print(f"=== FINITO ===  Video: {total_vids} | Frames OK: {total_ok} | Frames FAIL: {total_fail}")

if __name__ == "__main__":
    main()