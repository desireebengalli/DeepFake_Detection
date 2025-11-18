import os, glob, json, cv2, numpy as np
os.environ["OMP_NUM_THREADS"] = "2"        # meno contesa CPU quando la GPU lavora
os.environ["MKL_NUM_THREADS"] = "2"
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

from insightface.app import FaceAnalysis
import onnxruntime as ort

# ==== CONFIG ====
IN_ROOTS = [
    "/home/default/DeepFake_Detection/extracted_frames/FaceSwap",
    "/home/default/DeepFake_Detection/extracted_frames/NeuralTextures",
]
OUT_ROOT    = "/home/default/DeepFake_Detection/preprocessed_512jpg"
OUTPUT_SIZE = 512
MARGIN = 1.3
DET_THRESH = 0.3
JPEG_QUALITY = 92
# ===============

os.makedirs(OUT_ROOT, exist_ok=True)
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
REF_5PTS_112 = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                         [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
REF_5PTS = REF_5PTS_112 * (OUTPUT_SIZE / 112.0)

# ---- GPU check & FaceAnalysis on CUDA ----
providers = ort.get_available_providers()
use_gpu = "CUDAExecutionProvider" in providers
print("[INFO] ONNX providers:", providers)
if not use_gpu:
    print("[WARN] CUDAExecutionProvider NON disponibile; userò CPUExecutionProvider.")

app = FaceAnalysis(name="buffalo_l")
ctx = 0 if use_gpu else -1        # forza GPU se disponibile
# det_size=640 è il default “giusto”; non lo abbassiamo per non alterare i risultati
app.prepare(ctx_id=ctx, det_thresh=DET_THRESH, det_size=(640, 640))

# warmup: carica pesi in VRAM per evitare il primo frame lento
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
_ = app.get(_dummy)
print("[INFO] InsightFace pronto su", ("GPU" if use_gpu else "CPU"))

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
    return {"sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(gray.mean())}

def list_video_dirs(root):
    """Sottocartelle immediate che contengono immagini (un video per cartella)."""
    if not os.path.isdir(root): return []
    out = []
    for name in sorted(os.listdir(root)):
        vdir = os.path.join(root, name)
        if not os.path.isdir(vdir): continue
        has_imgs = any(glob.glob(os.path.join(vdir, p)) for p in IMG_PATTERNS)
        if has_imgs: out.append(vdir)
    return out

def process_video_dir(video_dir, out_root):
    # preserva la gerarchia: out_root/<root_name>/<subpath>
    matched_root = None
    for r in IN_ROOTS:
        rp = os.path.commonpath([os.path.abspath(video_dir), os.path.abspath(r)])
        if rp == os.path.abspath(r):
            matched_root = r; break
    if matched_root is None:
        rel = os.path.basename(video_dir.rstrip('/'))
        root_tag = "misc"
    else:
        rel = os.path.relpath(video_dir, start=matched_root)
        root_tag = os.path.basename(matched_root.rstrip("/"))

    out_dir = os.path.join(out_root, root_tag, rel)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted([p for ext in IMG_PATTERNS for p in glob.glob(os.path.join(video_dir, ext))])
    print(f"[INFO] {video_dir} -> {len(frames)} frame")

    meta = {"video_id": os.path.basename(video_dir.rstrip('/')), "frames":[]}
    ok = fail = 0

    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            meta["frames"].append({"frame": os.path.basename(fp), "status":"imread_failed"}); fail += 1; continue

        faces = app.get(img)  # GPU se disponibile
        if not faces:
            meta["frames"].append({"frame": os.path.basename(fp), "status":"no_face"}); fail += 1; continue

        # volto più grande
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        x1,y1,x2,y2 = [int(v) for v in f.bbox]
        h, w = img.shape[:2]

        try:
            face_aligned = align_by_5pts(img, f.kps, out_size=OUTPUT_SIZE)
            bx1,by1,bx2,by2 = expand_square([x1,y1,x2,y2], MARGIN, w, h)
            context = cv2.resize(img[by1:by2, bx1:bx2], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            bx1,by1,bx2,by2 = expand_square([x1,y1,x2,y2], MARGIN, w, h)
            face_aligned = cv2.resize(img[by1:by2, bx1:bx2], (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
            context = face_aligned

        base = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}.jpg"),     face_aligned, JPEG_PARAMS)
        cv2.imwrite(os.path.join(out_dir, f"{base}_ctx.jpg"), context,      JPEG_PARAMS)

        meta["frames"].append({
            "frame": os.path.basename(fp),
            "status":"ok",
            "bbox":[int(x1),int(y1),int(x2),int(y2)],
            "kps": np.asarray(f.kps).astype(float).round(2).tolist(),
            "files":{"face": f"{base}.jpg", "context": f"{base}_ctx.jpg"},
            "quality": quality_scores(face_aligned)
        })
        ok += 1

    meta["summary"] = {"ok":ok,"fail":fail,"total":len(frames)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] {video_dir} -> ok:{ok} fail:{fail}")
    return ok, fail

def main():
    total_ok = total_fail = total_videos = 0
    for root in IN_ROOTS:
        vdirs = list_video_dirs(root)
        print(f"[ROOT] {root} -> {len(vdirs)} video")
        for vd in vdirs:
            ok, fail = process_video_dir(vd, OUT_ROOT)
            total_ok += ok; total_fail += fail; total_videos += 1
    print(f"=== FINITO ===  Video: {total_videos} | Frames OK: {total_ok} | Frames FAIL: {total_fail}")

if __name__ == "__main__":
    main()
