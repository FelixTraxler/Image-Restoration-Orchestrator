from datetime import datetime
import gc
import os
import shutil
import time
import glob
import subprocess
import sys
import gradio as gr
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.visual_util import predictions_to_glb


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def handle_uploads(input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()

    # Create a unique folder name with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Use absolute path to avoid confusion
    target_dir = os.path.abspath(f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    
    print(f"DEBUG: Creating target directory: {target_dir}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths, log_message).
    If nothing is uploaded, returns None values.
    """
    if not input_images:
        print("DEBUG: No images uploaded")
        return None, [], "Please upload images to continue."
    target_dir, image_paths = handle_uploads(input_images)
    print(f"DEBUG: update_gallery_on_upload returning target_dir='{target_dir}', num_images={len(image_paths)}")
    return target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def clear_fields():
    """Clears the 3D viewer."""
    return None


def update_log():
    """Display a quick log message while waiting."""
    return "Loading and Reconstructing..."


def run_vggt_inference(target_dir):
    """
    Run VGGT inference by calling run.py as a subprocess.
    Returns the path to the results directory.
    """
    if not target_dir or not os.path.isdir(target_dir):
        raise ValueError("No valid target directory found. Please upload images first.")
    
    input_dir = os.path.abspath(os.path.join(target_dir, "images"))
    output_dir = os.path.abspath(os.path.join(target_dir, "results"))
    
    print(f"DEBUG: input_dir={input_dir}, exists={os.path.exists(input_dir)}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Get the path to run.py relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    vggt_dir = os.path.join(project_root, "vggt")
    run_script = os.path.join(vggt_dir, "run.py")
    
    # Use the vggt venv Python interpreter instead of current one
    vggt_python = os.path.join(vggt_dir, "venv", "bin", "python3")
    
    # Fallback to current Python if vggt venv doesn't exist
    if not os.path.exists(vggt_python):
        print(f"WARNING: VGGT venv not found at {vggt_python}, using current Python")
        vggt_python = sys.executable
    
    # Call run.py as subprocess
    print(f"Running VGGT inference on {input_dir}...")
    print(f"DEBUG: run_script={run_script}, exists={os.path.exists(run_script)}")
    print(f"DEBUG: vggt_python={vggt_python}, exists={os.path.exists(vggt_python)}")
    
    cmd = [
        vggt_python,
        run_script,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
    ]
    
    print(f"DEBUG: Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=vggt_dir)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"VGGT inference failed: {result.stderr}")
    
    print("STDOUT:", result.stdout)
    return output_dir


def gradio_reconstruct(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction by calling run.py subprocess and then visualizing.
    """
    print(f"DEBUG: gradio_reconstruct called with target_dir='{target_dir}', type={type(target_dir)}")
    
    if not target_dir or not os.path.isdir(target_dir):
        print(f"DEBUG: Invalid target_dir. Exists: {os.path.exists(target_dir) if target_dir else 'N/A'}")
        return None, "No valid target directory found. Please upload images first.", None
    
    start_time = time.time()
    gc.collect()
    
    # Get frame filter choices
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files_display = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files_display
    
    try:
        # Run inference via subprocess
        output_dir = run_vggt_inference(target_dir)
        
        # Load predictions
        predictions_path = os.path.join(output_dir, "predictions.npz")
        if not os.path.exists(predictions_path):
            return None, f"Predictions file not found at {predictions_path}", gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True)
        
        loaded = np.load(predictions_path)
        key_list = [
            "pose_enc",
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "images",
            "extrinsic",
            "intrinsic",
            "world_points_from_depth",
        ]
        predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded}
        
        # Handle None frame_filter
        if frame_filter is None:
            frame_filter = "All"
        
        # Build GLB file name
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
        )
        
        # Convert predictions to GLB
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)
        
        # Cleanup
        del predictions
        gc.collect()
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Visualization complete."
        
        return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)
    
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}")
        return None, f"Error: {str(e)}", gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True)


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """
    if not target_dir or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."
    
    predictions_path = os.path.join(target_dir, "results", "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available. Please run 'Reconstruct' first."
    
    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]
    
    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded}
    
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )
    
    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)
    
    return glbfile, "Visualization updated"


# -------------------------------------------------------------------------
# Main Gradio Page
# -------------------------------------------------------------------------
def vggt_page():
    """Create and return the VGGT page component."""
    
    # Use State instead of Textbox for storing target_dir
    target_dir_output = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            
            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                # show_download_button=True,
                object_fit="contain",
                preview=True,
            )
        
        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload images, then click Reconstruct.",
                    elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)
            
            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )
            
            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                )
            
            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)
    
    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------
    
    # Auto-update gallery on upload
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_images],
        outputs=[target_dir_output, image_gallery, log_output],
    )
    
    # Reconstruct button
    submit_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[reconstruction_output]
    ).then(
        fn=update_log,
        inputs=[],
        outputs=[log_output]
    ).then(
        fn=gradio_reconstruct,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    )
    
    # Real-time visualization updates
    conf_thres.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [target_dir_output, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode],
        [reconstruction_output, log_output],
    )
