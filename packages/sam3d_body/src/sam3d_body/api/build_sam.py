# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
import torch


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("########### Using human segmentor: SAM2...")
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2

        else:
            raise NotImplementedError

    def run_sam(self, img, boxes, **kwargs):
        return self.sam_func(self.sam, img, boxes)


def _pick_ckpt_and_cfg(path: Path) -> tuple[Path, str]:
    """Return the checkpoint file and config name (relative path) for the given location."""
    path = Path(path)

    # If the user points directly to a .pt file, use it.
    if path.is_file() and path.suffix == ".pt":
        ckpt = path
    else:
        # Otherwise search common filenames inside the directory.
        candidates = [
            path / "sam2.1_hiera_large.pt",
            path / "sam2.1_hiera_base_plus.pt",
            path / "sam2.1_hiera_base.pt",
            path / "sam2.1_hiera_small.pt",
            path / "sam2.1_hiera_tiny.pt",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            pt_files = list(path.glob("*.pt"))
            if pt_files:
                ckpt = pt_files[0]

    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {path}. "
            "Pass --segmentor-path to the checkpoint file or a directory containing it."
        )

    ckpt_name = ckpt.name
    if "hiera_large" in ckpt_name:
        cfg_name = "sam2.1_hiera_l.yaml"
    elif "hiera_base_plus" in ckpt_name or "hiera_b+" in ckpt_name:
        cfg_name = "sam2.1_hiera_b+.yaml"
    elif "hiera_small" in ckpt_name:
        cfg_name = "sam2.1_hiera_s.yaml"
    elif "hiera_tiny" in ckpt_name:
        cfg_name = "sam2.1_hiera_t.yaml"
    else:
        # Default to large if we cannot infer; it exists in the package.
        cfg_name = "sam2.1_hiera_l.yaml"

    cfg_relpath = f"configs/sam2.1/{cfg_name}"
    return ckpt, cfg_relpath


def load_sam2(device, path):
    ckpt_path, cfg_relpath = _pick_ckpt_and_cfg(Path(path))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(cfg_relpath, str(ckpt_path), device=device))
    predictor.model.eval()

    return predictor


def run_sam2(sam_predictor, img, boxes):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img)
        all_masks, all_scores = [], []
        for i in range(boxes.shape[0]):
            # First prediction: bbox only
            masks, scores, logits = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[[i]],
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            mask_1 = masks[0]
            score_1 = scores[0]
            all_masks.append(mask_1)
            all_scores.append(score_1)

            # cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)[:-4]}_mask_{i}.jpg"), (mask_1 * 255).astype(np.uint8))
        all_masks = np.stack(all_masks)
        all_scores = np.stack(all_scores)

    return all_masks, all_scores
