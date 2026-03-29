from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from yolo_facenet_pipeline import crop_face, detect_faces, get_embedding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_image_files(folder: Path, recursive: bool):
    if recursive:
        paths = folder.rglob("*")
    else:
        paths = folder.iterdir()
    for p in sorted(paths):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def extract_from_folder(
    input_dir: Path,
    output_npz: Path,
    crops_dir: Optional[Path],
    recursive: bool,
):
    """Each detected face in each image becomes one row in the output arrays."""
    embeddings_list = []
    paths_list = []
    face_idx_list = []
    boxes_list = []

    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_image_files(input_dir, recursive=recursive):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"skip (unreadable): {img_path}")
            continue

        boxes = detect_faces(image)
        if not boxes:
            print(f"skip (no face): {img_path}")
            continue

        stem = img_path.stem
        for fi, box in enumerate(boxes):
            crop = crop_face(image, box)
            emb = get_embedding(crop)

            embeddings_list.append(emb)
            paths_list.append(str(img_path.resolve()))
            face_idx_list.append(fi)
            boxes_list.append(box)

            if crops_dir is not None:
                out_crop = crops_dir / f"{stem}_face{fi}.png"
                cv2.imwrite(str(out_crop), crop)

    if not embeddings_list:
        print("No embeddings extracted.")
        sys.exit(1)

    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
    face_indices = np.array(face_idx_list, dtype=np.int32)
    boxes_arr = np.array(boxes_list, dtype=np.int32)
    path_arr = np.array(paths_list, dtype=object)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        embeddings=embeddings,
        image_paths=path_arr,
        face_index=face_indices,
        boxes_xyxy=boxes_arr,
    )
    print(
        f"Saved {len(embeddings_list)} face embedding(s) to {output_npz} "
        f"(shape {embeddings.shape})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Crop faces from images in a folder and save FaceNet embeddings. "
            "Images may contain one or more faces; each face gets one embedding row. "
            "Join rows with the same image_paths value using face_index (0, 1, …)."
        ),
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("sample_images"),
        help="Directory containing images (default: sample_images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("embeddings.npz"),
        help="Output .npz path (default: embeddings.npz)",
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=Path("crops"),
        help="Write each 160x160 face crop as {stem}_face0.png, {stem}_face1.png, …",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include images in subfolders",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}")
        sys.exit(1)

    crops = args.crops_dir.resolve() if args.crops_dir is not None else None
    extract_from_folder(
        input_dir,
        args.output.resolve(),
        crops,
        recursive=args.recursive,
    )
