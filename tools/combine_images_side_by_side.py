import os
import argparse
from typing import Tuple

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def read_image(path: str) -> np.ndarray:
    img = mpimg.imread(path)
    # Convert floating [0,1] to uint8 if needed
    if img.dtype.kind == "f":
        img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # Ensure 3-channel RGB (drop alpha if present)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def trim_whitespace(img: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255), tolerance: int = 5) -> np.ndarray:
    """Trim uniform-color borders (default: near-white) to make the image compact."""
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    diff = np.abs(img.astype(np.int16) - np.array(bg_color, dtype=np.int16))
    # content mask: any channel differs from background beyond tolerance
    content_mask = (diff > tolerance).any(axis=2)
    if not content_mask.any():
        return img
    rows = np.where(content_mask.any(axis=1))[0]
    cols = np.where(content_mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return img[top:bottom + 1, left:right + 1]


def pad_to_height(img: np.ndarray, target_h: int, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h, w, c = img.shape
    if h == target_h:
        return img
    pad = target_h - h
    top = pad // 2
    bottom = pad - top
    pad_block = np.full((top, w, c), bg_color, dtype=img.dtype)
    pad_block2 = np.full((bottom, w, c), bg_color, dtype=img.dtype)
    return np.vstack([pad_block, img, pad_block2])


def combine_side_by_side(left_path: str, right_path: str, out_path: str, align: str = "height", gap_pixels: int = 6) -> None:
    left = trim_whitespace(read_image(left_path))
    right = trim_whitespace(read_image(right_path))

    if align == "height":
        target_h = max(left.shape[0], right.shape[0])
        left = pad_to_height(left, target_h)
        right = pad_to_height(right, target_h)

    # small white gap between images for visual separation
    gap = np.full((left.shape[0], gap_pixels, 3), 255, dtype=left.dtype)
    combined = np.concatenate([left, gap, right], axis=1)
    # Save with plt.imsave to avoid extra deps
    plt.imsave(out_path, combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine two images side by side.")
    parser.add_argument("left", nargs="?", default="/data/zhantianyu/Project/GRID/logs/train/runs/beauty/gr/aggregates/compare_NDCG_at_5.png", help="Left image path")
    parser.add_argument("right", nargs="?", default="/data/zhantianyu/Project/GRID/logs/train/runs/beauty/gr/aggregates/test_bars.png", help="Right image path")
    parser.add_argument("output", nargs="?", default="/data/zhantianyu/Project/GRID/logs/train/runs/beauty/gr/aggregates/combined_ndcg5_test.png", help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combine_side_by_side(args.left, args.right, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


