"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import Int, UInt8
from monopriors.relative_depth_models import RelativeDepthPrediction
from numpy import ndarray

from sam3d_body.api.demo import SAM3Config, SAM3DBodyE2E, SAM3DBodyE2EConfig, create_view, set_annotation_context
from sam3d_body.api.visualization import visualize_sample
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction

CFG: SAM3DBodyE2EConfig = SAM3DBodyE2EConfig(sam3_config=SAM3Config())
MODEL_E2E: SAM3DBodyE2E = SAM3DBodyE2E(config=CFG)
mesh_faces: Int[ndarray, "n_faces=36874 3"] = MODEL_E2E.sam3d_body_estimator.faces


@rr.thread_local_stream("sam3d_body_gradio_ui")
def sam3d_prediction_fn(rgb_hw3: UInt8[ndarray, "h w 3"], pending_cleanup) -> tuple[str, str]:
    # We eventually want to clean up the RRD file after it's sent to the viewer, so tracking
    # any pending files to be cleaned up when the state is deleted.
    temp = tempfile.NamedTemporaryFile(prefix="cube_", suffix=".rrd", delete=False)
    pending_cleanup.append(temp.name)

    view: rrb.ContainerLike = create_view()
    blueprint = rrb.Blueprint(view, collapse_panels=True)
    rr.save(path=temp.name, default_blueprint=blueprint)
    set_annotation_context()
    parent_log_path = Path("/world")
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    outputs: tuple[list[FinalPosePrediction], RelativeDepthPrediction] = MODEL_E2E.predict_single_image(rgb_hw3=rgb_hw3)
    pred_list: list[FinalPosePrediction] = outputs[0]
    relative_pred: RelativeDepthPrediction = outputs[1]
    rr.set_time(timeline="image_sequence", sequence=0)
    visualize_sample(
        pred_list=pred_list,
        rgb_hw3=rgb_hw3,
        parent_log_path=parent_log_path,
        faces=mesh_faces,
    )

    return temp.name, "Done"


def cleanup_rrds(pending_cleanup: list[str]) -> None:
    for f in pending_cleanup:
        os.unlink(f)


def main():
    with gr.Blocks() as demo, gr.Tab("SAM3D Body Estimation"):
        pending_cleanup = gr.State([], time_to_live=10, delete_callback=cleanup_rrds)
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                create_rrd = gr.Button("Create RRD")
                json_output = gr.Text()
            with gr.Column(scale=5):
                viewer = Rerun(
                    streaming=True,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "hidden",
                        "selection": "hidden",
                    },
                    height=800,
                )
        create_rrd.click(
            sam3d_prediction_fn,
            inputs=[img, pending_cleanup],
            outputs=[viewer, json_output],
        )
    return demo
