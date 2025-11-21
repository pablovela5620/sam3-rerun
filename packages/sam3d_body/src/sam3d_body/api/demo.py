# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam3d_body.api.vis_utils import visualize_sample_together
from sam3d_body.build_models import load_sam_3d_body
from sam3d_body.sam_3d_body_estimator import PosePrediction, SAM3DBodyEstimator


@dataclass
class Sam3DBodyConfig:
    """
    Configure the demo entrypoint.

    Attributes:
        mhr_path: Path to the MHR mesh/pose asset file required by the head network.
        detector_path: Optional local checkpoint for the human detector; empty to use default weights.
        segmentor_path: Optional local checkpoint for the SAM-based human segmentor; empty to use default weights.
        image_folder: Directory containing input images to process.
        output_folder: Directory where rendered visualizations will be saved.
        checkpoint_path: Core SAM 3D Body model checkpoint (.ckpt).
        detector_name: Human detector backbone to load (None disables detection and uses full-image box).
        segmentor_name: Segmentor name used when `segmentor_path` is provided.
        fov_name: FOV estimator name (None disables FOV estimation and uses default intrinsics).
        bbox_thresh: Confidence threshold for detector boxes.
        use_mask: Whether to request mask-conditioned inference (requires `segmentor_path` / SAM).
    """

    mhr_path: Path = Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    detector_path: str = ""
    segmentor_path: str = ""
    image_folder: Path = Path("test-input/")
    output_folder: Path = Path("test-outputs/")
    checkpoint_path: Path = Path("checkpoints/sam-3d-body-dinov3/model.ckpt")
    detector_name: Literal["vitdet"] | None = "vitdet"
    segmentor_name: Literal["sam2"] | None = "sam2"
    fov_name: Literal["moge2"] | None = "moge2"
    bbox_thresh: float = 0.8
    use_mask: bool = False


def main(cfg: Sam3DBodyConfig):
    if cfg.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(cfg.image_folder))
    else:
        output_folder = cfg.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = cfg.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = cfg.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = cfg.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(cfg.checkpoint_path, device=device, mhr_path=mhr_path)
    human_detector, human_segmentor, fov_estimator = None, None, None
    if cfg.detector_name:
        from sam3d_body.api.build_detector import HumanDetector

        human_detector = HumanDetector(name=cfg.detector_name, device=device, path=detector_path)
    if len(segmentor_path):
        from sam3d_body.api.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(name=cfg.segmentor_name, device=device, path=segmentor_path)
    if cfg.fov_name:
        from sam3d_body.api.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=cfg.fov_name, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list: list[str] = sorted(
        [image for ext in image_extensions for image in glob(os.path.join(cfg.image_folder, ext))]
    )

    for image_path in cast(Iterable[str], tqdm(images_list)):
        outputs: list[PosePrediction] = estimator.process_one_image(
            image_path,
            bbox_thr=cfg.bbox_thresh,
            use_mask=cfg.use_mask,
        )

        if len(outputs) == 0:
            # Detector/FOV failed on this frame; avoid crashing the visualization step.
            print(f"[warn] No detections for {image_path}; skipping.")
            continue

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )
