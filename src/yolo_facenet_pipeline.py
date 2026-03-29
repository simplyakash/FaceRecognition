import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from numpy.linalg import norm
from ultralytics import YOLO


# ---------------------------------
# Configuration
# ---------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE_SIZE = 160
SIMILARITY_THRESHOLD = 0.6


# ---------------------------------
# Load Models
# ---------------------------------

print("Loading YOLO face detector...")
face_detector = YOLO("yolov8n-face.pt")

print("Loading FaceNet model...")
facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)


# ---------------------------------
# Face Detection
# ---------------------------------

def detect_faces(image):

    results = face_detector(image)

    boxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2))

    return boxes


# ---------------------------------
# Face Crop
# ---------------------------------

def crop_face(image, box):

    x1, y1, x2, y2 = box

    face = image[y1:y2, x1:x2]

    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

    return face


# ---------------------------------
# Embedding Extraction
# ---------------------------------

def get_embedding(face):

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0

    face = np.transpose(face, (2, 0, 1))

    face = torch.tensor(face).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = facenet(face)

    return embedding.cpu().numpy()[0]


# ---------------------------------
# Similarity
# ---------------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (norm(a) * norm(b))


def _l2_normalize(vec):

    n = norm(vec)
    if n < 1e-8:
        return vec
    return vec / n


def load_gallery(npz_path):

    data = np.load(str(npz_path), allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    paths = data["image_paths"]
    boxes = data["boxes_xyxy"].astype(np.int32)
    if "face_index" in data.files:
        gallery_face_idx = data["face_index"].astype(np.int32)
    else:
        gallery_face_idx = np.zeros(len(paths), dtype=np.int32)
    embeddings = embeddings / (
        norm(embeddings, axis=1, keepdims=True) + 1e-8
    )
    return embeddings, paths, boxes, gallery_face_idx


def best_match_index(query_embedding, gallery_embeddings):

    q = _l2_normalize(query_embedding.astype(np.float32))
    sims = gallery_embeddings @ q
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def top_k_gallery_matches(query_embedding, gallery_embeddings, k=1):

    q = _l2_normalize(query_embedding.astype(np.float32))
    sims = gallery_embeddings @ q
    k = max(1, min(int(k), len(sims)))
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def gallery_face_thumbnail(paths, boxes, row_index, size):

    path = str(paths[row_index])
    img = cv2.imread(path)
    if img is None:
        panel = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(
            panel,
            "missing",
            (10, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        return panel

    x1, y1, x2, y2 = map(int, boxes[row_index])
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))


def _box_area(box):

    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _clamp_box(box, width, height):

    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


# ---------------------------------
# Match one image against gallery (.npz)
# ---------------------------------


def match_image_to_gallery(
    image_path,
    embeddings_npz,
    faces="largest",
    top_k=1,
):

    """
    Compare faces in a query image to all rows in embeddings.npz.

    faces:
      "largest" — only the biggest face box in the query image
      "first"   — only the first detected box (YOLO order)
      "all"     — one best (or top_k) gallery hit per detected face

    Returns a list of dicts with query_face_index, query_box, matches (list of
    (gallery_row, score)), and gallery metadata for the best hit.
    """

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gallery_emb, gallery_paths, gallery_boxes, gallery_face_idx = load_gallery(
        embeddings_npz
    )
    if gallery_emb.shape[0] == 0:
        return []

    h, w = img.shape[:2]
    boxes = detect_faces(img)
    if not boxes:
        return []

    if faces == "largest":
        bi = max(range(len(boxes)), key=lambda i: _box_area(boxes[i]))
        face_boxes = [boxes[bi]]
        query_indices = [bi]
    elif faces == "first":
        face_boxes = [boxes[0]]
        query_indices = [0]
    elif faces == "all":
        face_boxes = boxes
        query_indices = list(range(len(boxes)))
    else:
        raise ValueError('faces must be "largest", "first", or "all"')

    results = []
    for qfi, box in zip(query_indices, face_boxes):
        x1, y1, x2, y2 = _clamp_box(box, w, h)
        crop = crop_face(img, (x1, y1, x2, y2))
        qemb = get_embedding(crop)
        ranked = top_k_gallery_matches(qemb, gallery_emb, k=top_k)
        match_records = []
        for row, score in ranked:
            match_records.append(
                {
                    "gallery_row": row,
                    "score": score,
                    "image_path": str(gallery_paths[row]),
                    "face_index": int(gallery_face_idx[row]),
                    "box_xyxy": tuple(map(int, gallery_boxes[row])),
                }
            )
        best = match_records[0]
        results.append(
            {
                "query_face_index": qfi,
                "query_box_xyxy": (x1, y1, x2, y2),
                "matches": match_records,
                "best_gallery_row": best["gallery_row"],
                "best_score": best["score"],
                "gallery_image_path": best["image_path"],
                "gallery_face_index": best["face_index"],
                "gallery_box_xyxy": best["box_xyxy"],
            }
        )
    return results


def print_match_results(image_path, results):

    if not results:
        print(f"No faces in query image: {image_path}")
        return

    print(f"Query: {image_path}")
    for r in results:
        print(f"  face #{r['query_face_index']} box={r['query_box_xyxy']}")
        for rank, m in enumerate(r["matches"], start=1):
            flag = "✓" if m["score"] >= SIMILARITY_THRESHOLD else "·"
            print(
                f"    {flag} rank {rank}: row {m['gallery_row']}  "
                f"score={m['score']:.4f}  face_index={m['face_index']}\n"
                f"         path={m['image_path']}"
            )


def gallery_crop_square(image_path, box_xyxy, side):

    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((side, side, 3), dtype=np.uint8)

    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((side, side, 3), dtype=np.uint8)

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (side, side))


def build_match_collage(
    query_bgr,
    query_box_xyxy,
    matches,
    cell_size=200,
    gap=8,
):

    """
    Top: full query image scaled to width W (W = k * cell + gaps).
    Bottom: one row of k gallery face crops with rank/score labels.
    """

    k = len(matches)
    if k == 0:
        return query_bgr.copy()

    W = k * cell_size + (k - 1) * gap

    qdisp = query_bgr.copy()
    x1, y1, x2, y2 = query_box_xyxy
    t = max(2, min(qdisp.shape[0], qdisp.shape[1]) // 200)
    cv2.rectangle(qdisp, (x1, y1), (x2, y2), (0, 255, 0), t)

    qh, qw = qdisp.shape[:2]
    new_h = max(1, int(round(qh * W / qw)))
    qrow = cv2.resize(qdisp, (W, new_h))

    tiles = []
    for i, m in enumerate(matches):
        tile = gallery_crop_square(m["image_path"], m["box_xyxy"], cell_size)
        rank = i + 1
        ok = m["score"] >= SIMILARITY_THRESHOLD
        color = (0, 200, 0) if ok else (0, 140, 255)
        cv2.rectangle(tile, (0, 0), (cell_size - 1, cell_size - 1), color, 2)
        label = f"#{rank} {m['score']:.3f}"
        cv2.putText(
            tile,
            label,
            (4, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        tiles.append(tile)
        if i < k - 1:
            tiles.append(np.full((cell_size, gap, 3), 255, dtype=np.uint8))

    bottom = np.hstack(tiles)
    spacer = np.full((gap, W, 3), 255, dtype=np.uint8)
    return np.vstack([qrow, spacer, bottom])


def save_match_collages(query_bgr, results, out_path, cell_size=200):

    """
    Writes one image per query face. If multiple faces, names are
    stem_face{idx}.suffix next to out_path stem.
    """

    out_path = Path(out_path)
    multi = len(results) > 1
    written = []

    for r in results:
        if multi:
            path = out_path.parent / f"{out_path.stem}_face{r['query_face_index']}{out_path.suffix}"
        else:
            path = out_path

        collage = build_match_collage(
            query_bgr,
            r["query_box_xyxy"],
            r["matches"],
            cell_size=cell_size,
        )
        ok = cv2.imwrite(str(path), collage)
        if ok:
            written.append(str(path))

    return written


# ---------------------------------
# Webcam + gallery match (realtime)
# ---------------------------------


def _video_capture_backends():

    order = [None]
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        order.append(cv2.CAP_V4L2)
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        order.append(cv2.CAP_DSHOW)
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        order.append(cv2.CAP_AVFOUNDATION)
    return order


def open_video_capture(camera_id=None, max_index=10):

    """
    Open a device that yields at least one non-empty frame.

    camera_id None: try indices 0..max_index for each backend (auto-detect).
    camera_id int: try only that index (each backend).
    Returns (VideoCapture, index) or (None, None).
    """

    backends = _video_capture_backends()

    def try_open(index, backend):
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.release()
            return None
        return cap

    if camera_id is not None:
        indices = [int(camera_id)]
    else:
        indices = list(range(max_index + 1))

    for backend in backends:
        for idx in indices:
            cap = try_open(idx, backend)
            if cap is not None:
                label = "default" if backend is None else repr(backend)
                print(f"Using camera index {idx} (OpenCV API {label})")
                return cap, idx

    return None, None


def _linux_has_display_env():

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


_IMWRITE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def _ensure_image_extension_for_imwrite(path):

    p = Path(path)
    if p.suffix.lower() in _IMWRITE_EXTS:
        return p
    out = p.with_suffix(".jpg")
    print(
        f"Note: imwrite needs .jpg/.png/…; using {out} "
        f"(was missing or unsupported suffix {p.suffix!r})"
    )
    return out


def run_webcam_recognition(
    embeddings_npz,
    camera_id=None,
    match_thumbnail_size=None,
    max_camera_scan=10,
    headless=False,
    out_video=None,
    dump_dir=None,
    dump_file=None,
    dump_every=1,
    max_frames=None,
    target_fps=None,
):

    gallery_emb, gallery_paths, gallery_boxes, _ = load_gallery(embeddings_npz)
    if gallery_emb.shape[0] == 0:
        print("Gallery is empty.")
        return

    cap, _used = open_video_capture(camera_id, max_index=max_camera_scan)
    if cap is None:
        print(
            "No working camera found. Try: plug the device, close other apps using it, "
            "check permissions (Linux: video group), or pass an explicit index, e.g. "
            f"--camera 1 or --camera 2 (scanned 0..{max_camera_scan})."
        )
        return

    dump_path = Path(dump_dir) if dump_dir is not None else None
    dump_file_path = (
        _ensure_image_extension_for_imwrite(Path(dump_file))
        if dump_file is not None
        else None
    )
    use_gui = (
        not headless
        and out_video is None
        and dump_path is None
        and dump_file_path is None
    )
    if sys.platform.startswith("linux") and not _linux_has_display_env():
        if use_gui:
            print(
                "No DISPLAY or WAYLAND_DISPLAY: opening a window will fail. "
                "Use --dump-file F.jpg, --dump-dir DIR, --out-video FILE.mp4, or --headless."
            )
        use_gui = False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 20.0
    fps = min(30.0, float(fps))

    target_period = None
    if target_fps is not None and float(target_fps) > 0:
        target_period = 1.0 / float(target_fps)
        print(f"Loop cap ~{float(target_fps):.2f} fps (sleep after each frame)")

    record_fps = float(target_fps) if target_period is not None else fps

    win = "Camera | matched gallery face (q to quit)"
    last_thumb = None
    last_idx = -1
    writer = None
    frame_i = 0
    dumped = 0
    if dump_path is not None:
        dump_path.mkdir(parents=True, exist_ok=True)
        print(
            f"Saving combined frames to {dump_path.resolve()} "
            f"(every {max(1, dump_every)} frame(s); Ctrl+C to stop)"
        )
    if dump_file_path is not None:
        dump_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Overwriting {dump_file_path.resolve()} "
            f"(every {max(1, dump_every)} frame(s); Ctrl+C to stop)"
        )

    while True:
        t_loop = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and frame_i >= max_frames:
            break

        display = frame.copy()
        h, w = display.shape[:2]
        boxes = detect_faces(frame)

        for box in boxes:
            x1, y1, x2, y2 = _clamp_box(box, w, h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        thumb_h = h if match_thumbnail_size is None else int(match_thumbnail_size)
        panel_w = thumb_h
        dbg_idx, dbg_score = None, None

        if boxes:
            primary = max(boxes, key=_box_area)
            px1, py1, px2, py2 = _clamp_box(primary, w, h)
            cv2.rectangle(display, (px1, py1), (px2, py2), (0, 200, 255), 3)

            crop = crop_face(frame, (px1, py1, px2, py2))
            qemb = get_embedding(crop)
            idx, score = best_match_index(qemb, gallery_emb)

            label = f"{score:.2f}"
            if score >= SIMILARITY_THRESHOLD:
                label += " match"
            else:
                label += " ?"

            cv2.putText(
                display,
                label,
                (px1, max(0, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255) if score >= SIMILARITY_THRESHOLD else (0, 165, 255),
                2,
            )

            if idx != last_idx:
                last_thumb = gallery_face_thumbnail(
                    gallery_paths, gallery_boxes, idx, thumb_h
                )
                last_idx = idx
            elif last_thumb is None:
                last_thumb = gallery_face_thumbnail(
                    gallery_paths, gallery_boxes, idx, thumb_h
                )

            if last_thumb is not None and last_thumb.shape[0] != thumb_h:
                last_thumb = cv2.resize(last_thumb, (thumb_h, thumb_h))

            side = last_thumb.copy()
            cv2.putText(
                side,
                f"id {idx} {score:.2f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            dbg_idx, dbg_score = idx, score
        else:
            side = np.zeros((thumb_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(
                side,
                "No face",
                (20, thumb_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )
            last_idx = -1

        combined = np.hstack([display, side])

        if out_video is not None:
            if writer is None:
                oh, ow = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video), fourcc, record_fps, (ow, oh))
                if not writer.isOpened():
                    print(f"Could not open video writer: {out_video}")
                    cap.release()
                    return
                print(
                    f"Writing video to {out_video} at {record_fps:.1f} fps "
                    f"(Ctrl+C to stop)"
                )
            writer.write(combined)

        if dump_path is not None and frame_i % max(1, dump_every) == 0:
            out_img = dump_path / f"frame_{frame_i:06d}.jpg"
            cv2.imwrite(
                str(out_img),
                combined,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            dumped += 1

        if dump_file_path is not None and frame_i % max(1, dump_every) == 0:
            cv2.imwrite(
                str(dump_file_path),
                combined,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            dumped += 1

        if use_gui:
            cv2.imshow(win, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        elif out_video is None and dump_path is None and dump_file_path is None:
            if frame_i % 30 == 0:
                if dbg_idx is not None:
                    print(
                        f"frame {frame_i}  match_id={dbg_idx}  score={dbg_score:.3f}  "
                        f"faces={len(boxes)}"
                    )
                else:
                    print(f"frame {frame_i}  no face")

        frame_i += 1

        if target_period is not None:
            elapsed = time.perf_counter() - t_loop
            time.sleep(max(0.0, target_period - elapsed))

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Finished writing {out_video}")
    if dump_path is not None:
        print(f"Saved {dumped} image(s) under {dump_path.resolve()}")
    if dump_file_path is not None:
        print(f"Last write: {dump_file_path.resolve()} ({dumped} overwrite(s))")
    if use_gui:
        cv2.destroyAllWindows()


# ---------------------------------
# Face Verification
# ---------------------------------

def verify_faces(image1_path, image2_path):

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error loading images")
        return

    boxes1 = detect_faces(img1)
    boxes2 = detect_faces(img2)

    if len(boxes1) == 0 or len(boxes2) == 0:
        print("No face detected")
        return

    face1 = crop_face(img1, boxes1[0])
    face2 = crop_face(img2, boxes2[0])

    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)

    similarity = cosine_similarity(emb1, emb2)

    print("Similarity Score:", similarity)

    if similarity > SIMILARITY_THRESHOLD:
        print("✅ Same Person")
    else:
        print("❌ Different Person")


# ---------------------------------
# Main
# ---------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Verify two images, match one image to a gallery .npz, or live webcam matching."
        ),
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Live camera: draw face boxes and show best-matching gallery crop",
    )
    parser.add_argument(
        "--match-image",
        type=Path,
        default=None,
        metavar="PATH",
        help="Find closest gallery embedding(s) for this image (needs --embeddings)",
    )
    parser.add_argument(
        "--faces",
        choices=["largest", "first", "all"],
        default="largest",
        help="Which face(s) in the query image to match (default: largest box)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        metavar="K",
        help="Number of top gallery rows to show per query face (default: 1)",
    )
    parser.add_argument(
        "--match-out",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save a collage: query on top (matched face boxed), top-k gallery "
            "crops in one row below"
        ),
    )
    parser.add_argument(
        "--match-cell",
        type=int,
        default=200,
        metavar="PX",
        help="Pixel size of each gallery square in --match-out (default: 200)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="embeddings.npz from batch_extract_embeddings.py",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Camera index. If omitted, probes indices 0..--max-camera-scan with "
            "platform backends (V4L2 on Linux, etc.)"
        ),
    )
    parser.add_argument(
        "--max-camera-scan",
        type=int,
        default=10,
        metavar="N",
        help="Highest index to try when --camera is omitted (default: 10)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No GUI window: print match scores (for SSH or when Qt/X11 is unavailable)",
    )
    parser.add_argument(
        "--out-video",
        type=Path,
        default=None,
        metavar="FILE.mp4",
        help="Record the combined view to a video file (implies no GUI window)",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save each combined frame as numbered JPGs in this folder (no GUI)",
    )
    parser.add_argument(
        "--dump-file",
        type=Path,
        default=None,
        metavar="FILE.jpg",
        help="Save the combined view to this path only, overwriting each time (no GUI)",
    )
    parser.add_argument(
        "--dump-every",
        type=int,
        default=1,
        metavar="N",
        help="With --dump-dir or --dump-file, write every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after this many frames (useful with --dump-dir)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        metavar="FPS",
        help=(
            "Cap processing loop near this frame rate using sleep (default: no cap). "
            "Also sets --out-video writer fps when recording."
        ),
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=[],
        help="Two image paths for offline verify (ignored with --webcam)",
    )
    args = parser.parse_args()

    if args.webcam and args.match_image is not None:
        print("Error: use either --webcam or --match-image, not both.")
        sys.exit(1)

    if args.webcam:
        if args.embeddings is None:
            print("Error: --webcam requires --embeddings /path/to/embeddings.npz")
            sys.exit(1)
        if not args.embeddings.is_file():
            print(f"Not found: {args.embeddings}")
            sys.exit(1)
        outv = args.out_video.resolve() if args.out_video is not None else None
        dumpd = args.dump_dir.resolve() if args.dump_dir is not None else None
        dumpf = args.dump_file.resolve() if args.dump_file is not None else None
        if dumpd is not None and outv is not None:
            print("Error: use either --dump-dir or --out-video, not both.")
            sys.exit(1)
        if dumpf is not None and outv is not None:
            print("Error: use either --dump-file or --out-video, not both.")
            sys.exit(1)
        if dumpd is not None and dumpf is not None:
            print("Error: use either --dump-dir or --dump-file, not both.")
            sys.exit(1)
        run_webcam_recognition(
            args.embeddings.resolve(),
            camera_id=args.camera,
            max_camera_scan=max(0, args.max_camera_scan),
            headless=args.headless,
            out_video=outv,
            dump_dir=dumpd,
            dump_file=dumpf,
            dump_every=max(1, args.dump_every),
            max_frames=args.max_frames,
            target_fps=args.target_fps,
        )
    elif args.match_image is not None:
        if args.embeddings is None:
            print("Error: --match-image requires --embeddings")
            sys.exit(1)
        if not args.match_image.is_file():
            print(f"Not found: {args.match_image}")
            sys.exit(1)
        if not args.embeddings.is_file():
            print(f"Not found: {args.embeddings}")
            sys.exit(1)
        try:
            out = match_image_to_gallery(
                args.match_image.resolve(),
                args.embeddings.resolve(),
                faces=args.faces,
                top_k=max(1, args.top_k),
            )
            print_match_results(str(args.match_image), out)
            if args.match_out is not None and out:
                qimg = cv2.imread(str(args.match_image.resolve()))
                if qimg is None:
                    print("Could not reload query image for --match-out")
                else:
                    saved = save_match_collages(
                        qimg,
                        out,
                        args.match_out.resolve(),
                        cell_size=max(32, args.match_cell),
                    )
                    for p in saved:
                        print(f"Wrote collage: {p}")
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
    elif len(args.images) == 2:
        verify_faces(args.images[0], args.images[1])
    else:
        verify_faces("face1.jpeg", "face2.jpeg")