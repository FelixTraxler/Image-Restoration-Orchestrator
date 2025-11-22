import gradio as gr
from PIL import ImageOps, Image
from pathlib import Path
import subprocess

from components import vggt_page
from components import super_resolution
from components import dark_ir
from components import bw_to_color

with gr.Blocks() as demo:
    gr.Markdown("# Image Enhancement Model Comparison")
    gr.Markdown("Upload an image and select a model to enhance it. Use the slider to compare the resized input with the enhanced output.")

    with gr.Tabs():
        with gr.Tab("Super Resolution"):
           super_resolution.super_resolution()
        
        with gr.Tab("Dark IR"):
           dark_ir.dark_ir()

        with gr.Tab("B/W to Color"):
           bw_to_color.bw_to_color()
        
        with gr.Tab("3D Reconstruction"):
            vggt_page.vggt_page()

demo.launch(share=True)