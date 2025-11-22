# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast

import cv2
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from serde import from_dict, serde
from torchvision.transforms import ToTensor
from yacs.config import CfgNode

from sam3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam3d_body.data.utils.io import load_image
from sam3d_body.data.utils.prepare_batch import NoCollate, prepare_batch
from sam3d_body.models.meta_arch import SAM3DBody
from sam3d_body.models.meta_arch.sam3d_body import BodyPredContainer
from sam3d_body.utils import recursive_to


@serde
class PosePrediction:
    bbox: Float[ndarray, "4"]


class PosePredictionDict(TypedDict, total=False):
    bbox: Float[ndarray, "4"]
    focal_length: Float[ndarray, ""]
    pred_keypoints_3d: Float[ndarray, "joints 3"]
    pred_keypoints_2d: Float[ndarray, "joints 2"]
    pred_vertices: Float[ndarray, "verts 3"]
    pred_cam_t: Float[ndarray, "3"]
    pred_pose_raw: Float[ndarray, "pose_params"]
    global_rot: Float[ndarray, "3 3"]
    body_pose_params: Float[ndarray, "body_pose_params"]
    hand_pose_params: Float[ndarray, "hand_pose_params"]
    scale_params: Float[ndarray, "scale_params"]
    shape_params: Float[ndarray, "shape_params"]
    expr_params: Float[ndarray, "expr_params"]
    mask: UInt8[ndarray, "h w 1"] | None
    pred_joint_coords: Float[ndarray, "joints 3"]
    pred_global_rots: Float[ndarray, "joints 3 3"]
    lhand_bbox: Float[ndarray, "4"]
    rhand_bbox: Float[ndarray, "4"]


Transform = Callable[[dict], dict | None]


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model: SAM3DBody,
        model_cfg: CfgNode,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ) -> None:
        self.model: SAM3DBody = sam_3d_body_model
        self.cfg: CfgNode = model_cfg
        self.detector: Any | None = human_detector
        self.sam: Any | None = human_segmentor
        self.fov_estimator: Any | None = fov_estimator
        self.thresh_wrist_angle: float = 1.4

        # For mesh visualization
        self.faces: Int[ndarray, "n_faces=36874 3"] = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        body_transforms: list[Transform] = [
            cast(Transform, GetBBoxCenterScale()),
            cast(Transform, TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False)),
            cast(Transform, VisionTransformWrapper(ToTensor())),
        ]
        hand_transforms: list[Transform] = [
            cast(Transform, GetBBoxCenterScale(padding=0.9)),
            cast(Transform, TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False)),
            cast(Transform, VisionTransformWrapper(ToTensor())),
        ]

        self.transform: Compose = Compose(body_transforms)
        self.transform_hand: Compose = Compose(hand_transforms)

    @torch.no_grad()
    def process_one_image(
        self,
        img: str | UInt8[ndarray, "h w 3"],
        bboxes: Float[ndarray, "n 4"] | None = None,
        masks: UInt8[ndarray, "n h w 1"] | None = None,
        cam_int: Float[ndarray, "3 3"] | None = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: Literal["full", "body", "hand"] = "full",
    ) -> list[PosePredictionDict]:
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch: dict[str, Any] | None = None
        self.image_embeddings: Any | None = None
        self.output: Any | None = None
        self.prev_prompt: list[Any] = []
        self.is_crop: bool = False
        torch.cuda.empty_cache()

        if isinstance(img, str):
            img_array: UInt8[ndarray, "h w 3"] = load_image(img, backend="cv2", image_format="bgr").astype(np.uint8)
            image_format: Literal["bgr", "rgb"] = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            img_array = cast(UInt8[ndarray, "h w 3"], img)
            image_format = "rgb"
        height: int
        width: int
        height, width = img_array.shape[:2]

        if bboxes is not None:
            boxes: Float[ndarray, "n 4"] = bboxes.reshape(-1, 4).astype(np.float32)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = cast(
                Float[ndarray, "n 4"],
                self.detector.run_human_detection(
                    img_array,
                    det_cat_id=det_cat_id,
                    bbox_thr=bbox_thr,
                    nms_thr=nms_thr,
                    default_to_full_image=False,
                ).astype(np.float32),
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = cast(
                Float[ndarray, "1 4"],
                np.array([0, 0, width, height], dtype=np.float32).reshape(1, 4),
            )
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score: Float[ndarray, "n"] | None = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert bboxes is not None, "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(len(masks), dtype=np.float32)  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img_array, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch: dict[str, Any] = cast(
            dict[str, Any],
            prepare_batch(img_array, self.transform, boxes, masks, masks_score),
        )

        #################### Run model inference on an image ####################
        batch = cast(dict[str, Any], recursive_to(batch, "cuda"))
        self.model._initialize_batch(batch)
        batch_img: torch.Tensor = cast(torch.Tensor, batch["img"])

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        cam_int_tensor: torch.Tensor
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int_tensor = torch.as_tensor(cam_int, device=batch_img.device, dtype=batch_img.dtype)
            batch["cam_int"] = cam_int_tensor.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image: ndarray = cast(list[NoCollate], batch["img_ori"])[0].data
            cam_int_pred: torch.Tensor | ndarray = self.fov_estimator.get_cam_intrinsics(input_image)
            if isinstance(cam_int_pred, ndarray):
                cam_int_tensor = torch.as_tensor(cam_int_pred, device=batch_img.device, dtype=batch_img.dtype)
            else:
                cam_int_tensor = cast(torch.Tensor, cam_int_pred).to(batch_img)
            batch["cam_int"] = cam_int_tensor.clone()
        else:
            cam_int_tensor = cast(torch.Tensor, batch["cam_int"]).clone()

        outputs: BodyPredContainer = self.model.run_inference(
            img_array,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        pose_output: dict[str, Any] = outputs.pose_output
        batch_lhand: dict[str, Any] | None = outputs.batch_lhand
        batch_rhand: dict[str, Any] | None = outputs.batch_rhand

        out: dict[str, Any] = cast(dict[str, Any], pose_output["mhr"])
        out_np: dict[str, ndarray] = cast(dict[str, ndarray], recursive_to(recursive_to(out, "cpu"), "numpy"))
        all_out: list[PosePredictionDict] = []
        bbox_tensor: torch.Tensor = cast(torch.Tensor, batch["bbox"])

        for idx in range(batch_img.shape[1]):
            pred: PosePredictionDict = {
                "bbox": bbox_tensor[0, idx].cpu().numpy(),
                "focal_length": out_np["focal_length"][idx],
                "pred_keypoints_3d": out_np["pred_keypoints_3d"][idx],
                "pred_keypoints_2d": out_np["pred_keypoints_2d"][idx],
                "pred_vertices": out_np["pred_vertices"][idx],
                "pred_cam_t": out_np["pred_cam_t"][idx],
                "pred_pose_raw": out_np["pred_pose_raw"][idx],
                "global_rot": out_np["global_rot"][idx],
                "body_pose_params": out_np["body_pose"][idx],
                "hand_pose_params": out_np["hand"][idx],
                "scale_params": out_np["scale"][idx],
                "shape_params": out_np["shape"][idx],
                "expr_params": out_np["face"][idx],
                "mask": masks[idx] if masks is not None else None,
                "pred_joint_coords": out_np["pred_joint_coords"][idx],
                "pred_global_rots": out_np["joint_global_rots"][idx],
            }

            if inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
                lhand_center: torch.Tensor = cast(torch.Tensor, batch_lhand["bbox_center"]).flatten(0, 1)[idx]
                lhand_scale: torch.Tensor = cast(torch.Tensor, batch_lhand["bbox_scale"]).flatten(0, 1)[idx]
                pred["lhand_bbox"] = np.array(
                    [
                        (lhand_center[0] - lhand_scale[0] / 2).item(),
                        (lhand_center[1] - lhand_scale[1] / 2).item(),
                        (lhand_center[0] + lhand_scale[0] / 2).item(),
                        (lhand_center[1] + lhand_scale[1] / 2).item(),
                    ]
                )

                rhand_center: torch.Tensor = cast(torch.Tensor, batch_rhand["bbox_center"]).flatten(0, 1)[idx]
                rhand_scale: torch.Tensor = cast(torch.Tensor, batch_rhand["bbox_scale"]).flatten(0, 1)[idx]
                pred["rhand_bbox"] = np.array(
                    [
                        (rhand_center[0] - rhand_scale[0] / 2).item(),
                        (rhand_center[1] - rhand_scale[1] / 2).item(),
                        (rhand_center[0] + rhand_scale[0] / 2).item(),
                        (rhand_center[1] + rhand_scale[1] / 2).item(),
                    ]
                )

            all_out.append(pred)

        return all_out
