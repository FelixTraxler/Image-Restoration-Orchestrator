import subprocess
from PIL import Image, ImageOps
import gradio as gr

from constants import BASE_DIR

projects = [
]

def resize_image(image, model):
    project = next((p for p in projects if p["model"] == model), None)
    if project is None:
        return (None, None), "Invalid model selection"

    resized_image = ImageOps.contain(image, (256,256))
    resized_image.save(f"input_128/temp.png")

    subprocess.run(
        [str(project["venv"]), str(project["script"]), *project["args"]],
        cwd=project["cwd"],
        check=True,
    )

    output_image = Image.open(f"output_images/temp_{model}.png")
    return (resized_image, output_image), f"Used model: {model}"

def bw_to_color():
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            model_dropdown = gr.Dropdown(
                choices=[project["model"] for project in projects],
                label="Model",
                value=projects[0]["model"] if projects else None
            )
            submit_btn = gr.Button("Process Image", variant="primary")

        with gr.Column():
            image_slider = gr.ImageSlider(label="Resized Image vs Output Image")
            model_info = gr.Text(label="Model Info")

    submit_btn.click(
        fn=resize_image,
        inputs=[input_image, model_dropdown],
        outputs=[image_slider, model_info]
    )