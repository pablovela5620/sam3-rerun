# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm
from yacs.config import CfgNode

from sam3d_body.build_models import load_sam_3d_body
from sam3d_body.metadata.mhr70 import MHR70_ID2NAME, MHR70_IDS, MHR70_LINKS
from sam3d_body.models.meta_arch import SAM3DBody
from sam3d_body.sam_3d_body_estimator import PosePredictionDict, SAM3DBodyEstimator

BOX_PALETTE: UInt8[np.ndarray, "n_colors 4"] = np.array(
    [
        [255, 99, 71, 255],  # tomato
        [65, 105, 225, 255],  # royal blue
        [60, 179, 113, 255],  # medium sea green
        [255, 215, 0, 255],  # gold
        [138, 43, 226, 255],  # blue violet
        [255, 140, 0, 255],  # dark orange
        [220, 20, 60, 255],  # crimson
        [70, 130, 180, 255],  # steel blue
    ],
    dtype=np.uint8,
)

# Use a separate id range for segmentation classes to avoid clobbering the person class (id=0).
SEG_CLASS_OFFSET = 1000  # background = 1000, persons start at 1001


def compute_vertex_normals(
    verts: Float32[ndarray, "n_verts 3"],
    faces: Int[ndarray, "n_faces 3"],
    eps: float = 1e-12,
) -> Float32[ndarray, "n_verts 3"]:
    """Compute per-vertex normals for a single mesh.

    Args:
        verts: Float32 array of vertex positions with shape ``(n_verts, 3)``.
        faces: Int array of triangle indices with shape ``(n_faces, 3)``.
        eps: Small epsilon to avoid division by zero when normalizing.

    Returns:
        Float32 array of unit vertex normals with shape ``(n_verts, 3)``; zeros for degenerate vertices.
    """

    # Expand faces to vertex triplets and fetch their positions.
    faces_i: Int[ndarray, "n_faces 3"] = faces.astype(np.int64)
    v0: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 0]]
    v1: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 1]]
    v2: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 2]]

    # Face normal = cross(edge1, edge2).
    e1: Float32[ndarray, "n_faces 3"] = v1 - v0
    e2: Float32[ndarray, "n_faces 3"] = v2 - v0
    face_normals: Float32[ndarray, "n_faces 3"] = np.cross(e1, e2)

    # Accumulate each face normal into its three vertices with a vectorized scatter-add.
    vertex_normals: Float32[ndarray, "n_verts 3"] = np.zeros_like(verts, dtype=np.float32)
    flat_indices: Int[ndarray, "n_faces3"] = faces_i.reshape(-1)
    face_normals_repeated: Float32[ndarray, "n_faces3 3"] = np.repeat(face_normals, 3, axis=0)
    np.add.at(vertex_normals, flat_indices, face_normals_repeated)

    norms: Float32[ndarray, "n_verts 1"] = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    denom: Float32[ndarray, "n_verts 1"] = np.maximum(norms, eps).astype(np.float32)
    vn_unit: Float32[ndarray, "n_verts 3"] = (vertex_normals / denom).astype(np.float32)
    mask: ndarray = norms > eps
    vn_unit = np.where(mask, vn_unit, np.float32(0.0))
    return vn_unit


@dataclass(slots=True)
class Sam3DBodyConfig:
    """Configuration for the standalone demo runner."""

    rr_config: RerunTyroConfig
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""

    mhr_path: Path = Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    """Path to the MHR mesh/pose asset file required by the head network."""

    detector_path: str = ""
    """Optional local checkpoint for the human detector; empty uses default weights."""

    segmentor_path: Path = Path("checkpoints/sam2.1")
    """Optional local checkpoint for the SAM-based human segmentor; empty uses default weights."""

    image_folder: Path = Path("data/test-input/")
    """Directory containing input images to process."""

    checkpoint_path: Path = Path("checkpoints/sam-3d-body-dinov3/model.ckpt")
    """Core SAM 3D Body model checkpoint (.ckpt)."""

    detector_name: Literal["vitdet"] | None = "vitdet"
    """Detector backbone to load; ``None`` disables detection and uses full-image box."""

    segmentor_name: Literal["sam2"] | None = "sam2"
    """Segmentor name used when ``segmentor_path`` is provided; ``None`` disables segmentation."""

    fov_name: Literal["moge2"] | None = "moge2"
    """FOV estimator name; ``None`` disables FOV estimation and falls back to default intrinsics."""

    bbox_thresh: float = 0.8
    """Confidence threshold for detector boxes."""

    use_mask: bool = True
    """Whether to request mask-conditioned inference (requires ``segmentor_path`` / SAM)."""

    max_frames: int | None = None
    """Optional limit on the number of images to process; ``None`` processes all images."""


def set_annotation_context() -> None:
    """Register MHR-70 semantic metadata so subsequent logs show names/edges and mask colors."""
    # Base person class (for keypoints / boxes) uses id=0 (original), segmentation uses 1000+ to avoid clashes.
    person_class = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="Person", color=(0, 0, 255)),
        keypoint_annotations=[rr.AnnotationInfo(id=idx, label=name) for idx, name in MHR70_ID2NAME.items()],
        keypoint_connections=MHR70_LINKS,
    )

    # Segmentation classes: id=SEG_CLASS_OFFSET background, ids SEG_CLASS_OFFSET+1..n for each instance color.
    seg_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64))),
    ]
    for idx, color in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
        seg_classes.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(
                    id=SEG_CLASS_OFFSET + idx, label=f"Person-{idx}", color=tuple(int(c) for c in color)
                ),
            )
        )

    rr.log(
        "/",
        rr.AnnotationContext([person_class, *seg_classes]),
        static=True,
    )


def visualize_sample(
    outputs: list[PosePredictionDict], image_path: str, parent_log_path: Path, faces: Int[ndarray, "n_faces 3"]
) -> None:
    bgr_image: UInt8[ndarray, "h w 3"] = cv2.imread(image_path)
    cam_log_path: Path = parent_log_path / "cam"
    pinhole_log_path: Path = cam_log_path / "pinhole"
    image_log_path: Path = pinhole_log_path / "image"
    pred_log_path: Path = pinhole_log_path / "pred"
    # clear the previous pred logs
    rr.log(f"{pred_log_path}", rr.Clear(recursive=True))
    rr.log(f"{image_log_path}", rr.Image(bgr_image, color_model=rr.ColorModel.BGR))

    # Build per-pixel maps (SEG_CLASS_OFFSET = background). Also build RGBA overlay with transparent background.
    h, w = bgr_image.shape[:2]
    seg_map: Int[ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.int32)
    seg_overlay: UInt8[ndarray, "h w 4"] = np.zeros((h, w, 4), dtype=np.uint8)

    mesh_root_path: Path = parent_log_path / "pred"
    rr.log(str(mesh_root_path), rr.Clear(recursive=True))

    for i, output in enumerate(outputs):
        box_color: UInt8[ndarray, "1 4"] = BOX_PALETTE[i % len(BOX_PALETTE)].reshape(1, 4)
        rr.log(
            f"{pred_log_path}/bbox_{i}",
            rr.Boxes2D(
                array=output["bbox"],
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                colors=box_color,
                show_labels=True,
            ),
        )
        rr.log(
            f"{pred_log_path}/uv_{i}",
            rr.Points2D(
                positions=output["pred_keypoints_2d"],
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # Accumulate segmentation masks (if present) into a single segmentation image.
        mask = output.get("mask")
        if mask is not None:
            mask_arr: ndarray = np.asarray(mask).squeeze()
            if mask_arr.shape != seg_map.shape:
                mask_arr = cv2.resize(
                    mask_arr.astype(np.uint8), (seg_map.shape[1], seg_map.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            mask_bool = mask_arr.astype(bool)
            seg_id = SEG_CLASS_OFFSET + i + 1  # keep person class (0) separate from seg classes
            seg_map = np.where(mask_bool, np.uint16(seg_id), seg_map)

            # Color overlay for this instance, background stays transparent.
            color = BOX_PALETTE[i % len(BOX_PALETTE), :3]
            seg_overlay[mask_bool] = np.array([color[0], color[1], color[2], 120], dtype=np.uint8)

        # Log 3D keypoints in world coordinates
        kpts_cam: Float32[ndarray, "n_kpts 3"] = np.ascontiguousarray(output["pred_keypoints_3d"], dtype=np.float32)
        cam_t: Float32[ndarray, "3"] = np.ascontiguousarray(output["pred_cam_t"], dtype=np.float32)
        kpts_world: Float32[ndarray, "n_kpts 3"] = np.ascontiguousarray(kpts_cam + cam_t, dtype=np.float32)
        rr.log(
            f"{parent_log_path}/pred/kpts3d_{i}",
            rr.Points3D(
                positions=kpts_world,
                keypoint_ids=MHR70_IDS,
                class_ids=0,
                colors=(0, 255, 0),
            ),
        )

        # Log the full-body mesh in world coordinates so it shows in 3D
        verts_cam: Float32[ndarray, "n_verts 3"] = np.ascontiguousarray(output["pred_vertices"], dtype=np.float32)
        verts_world: Float32[ndarray, "n_verts 3"] = np.ascontiguousarray(verts_cam + cam_t, dtype=np.float32)
        faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(faces, dtype=np.int32)
        vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)
        rr.log(
            f"{parent_log_path}/pred/mesh_{i}",
            rr.Mesh3D(
                vertex_positions=verts_world,
                triangle_indices=faces_int,
                vertex_normals=vertex_normals,
                albedo_factor=(
                    float(box_color[0, 0]) / 255.0,
                    float(box_color[0, 1]) / 255.0,
                    float(box_color[0, 2]) / 255.0,
                    0.35,
                ),
            ),
        )

    # Log segmentation ids (full map) and an RGBA overlay with transparent background.
    if np.any(seg_map != SEG_CLASS_OFFSET):
        rr.log(f"{pred_log_path}/segmentation_ids", rr.SegmentationImage(seg_map))
        rr.log(f"{pred_log_path}/segmentation_overlay", rr.Image(seg_overlay, color_model=rr.ColorModel.RGBA))


def create_view() -> rrb.ContainerLike:
    view_2d = rrb.Vertical(
        contents=[
            # Top: people-only overlay on the RGB image.
            rrb.Spatial2DView(
                name="image",
                contents=[
                    "/world/cam/pinhole/image",
                    "/world/cam/pinhole/pred/segmentation_overlay",
                ],
            ),
            # Bottom: 2D boxes + keypoints; segmentation hidden.
            rrb.Spatial2DView(
                name="mhr",
                contents=[
                    "/world/cam/pinhole/image",
                    "/world/cam/pinhole/pred/**",
                    "- /world/cam/pinhole/pred/segmentation_overlay/**",
                    "- /world/cam/pinhole/pred/segmentation_ids/**",
                ],
            ),
        ],
    )
    view_3d = rrb.Spatial3DView(name="mhr_3d", contents=["/world/pred/**"], line_grid=rrb.LineGrid3D(visible=False))
    main_view = rrb.Horizontal(contents=[view_2d, view_3d], column_shares=[2, 3])
    view = rrb.Tabs(contents=[main_view], name="sam-3d-body-demo")
    return view


def main(cfg: Sam3DBodyConfig):
    # rerun setup
    parent_log_path = Path("/world")
    set_annotation_context()
    # blueprint
    view = create_view()
    blueprint = rrb.Blueprint(view, collapse_panels=True)
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # Use command-line args or environment variables
    mhr_path = cfg.mhr_path
    detector_path = cfg.detector_path
    segmentor_path = cfg.segmentor_path

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    load_output: tuple[SAM3DBody, CfgNode] = load_sam_3d_body(cfg.checkpoint_path, device=device, mhr_path=mhr_path)
    model: SAM3DBody = load_output[0]
    model_cfg: CfgNode = load_output[1]

    human_detector, human_segmentor, fov_estimator = None, None, None
    if cfg.detector_name:
        from sam3d_body.api.build_detector import HumanDetector

        human_detector = HumanDetector(name=cfg.detector_name, device=device, path=detector_path)
    if segmentor_path:
        from sam3d_body.api.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(name=cfg.segmentor_name, device=device, path=str(segmentor_path))
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

    image_extensions: list[str] = [
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

    for idx, image_path in enumerate(tqdm(images_list)):
        rr.set_time(timeline="image_sequence", sequence=idx)
        bgr_hw3: UInt8[ndarray, "h w 3"] = cv2.imread(image_path)
        rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)
        outputs: list[PosePredictionDict] = estimator.process_one_image(
            rgb_hw3,
            bbox_thr=cfg.bbox_thresh,
            use_mask=cfg.use_mask,
        )

        if len(outputs) == 0:
            # Detector/FOV failed on this frame; avoid crashing the visualization step.
            print(f"[warn] No detections for {image_path}; skipping.")
            continue

        visualize_sample(outputs, image_path, parent_log_path=parent_log_path, faces=estimator.faces)
