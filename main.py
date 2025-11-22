import gradio as gr
from PIL import ImageOps, Image
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent

projects = [
    {
        "venv": BASE_DIR / "DarkIR/venv_DarkIR/bin/python",
        "script": BASE_DIR / "DarkIR/inference.py",
        "args": ["-i", "../input_128/", "-o", "../output_images/"],
        "cwd": BASE_DIR / "DarkIR",  # important: run inside the project folder
        "model": "DarkIR",
    },
    {
        "venv": BASE_DIR / "X-Restormer/venv/bin/python",
        "script": BASE_DIR / "X-Restormer/xrestormer/test.py",
        "args": ["-opt", "options/test/001_xrestormer_sr.yml"],
        "cwd": BASE_DIR / "X-Restormer",
        "model": "X-Restormer",
    },
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

with gr.Blocks() as demo:
    gr.Markdown("# Image Enhancement Model Comparison")
    gr.Markdown("Upload an image and select a model to enhance it. Use the slider to compare the resized input with the enhanced output.")

    with gr.Tabs():
        with gr.Tab("Image Enhancement"):
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
        
        with gr.Tab("Test"):
            with gr.Row():
                with gr.Column():
                    test_dropdown = gr.Dropdown(
                        choices=["Test1", "Test2", "Test3"],
                        label="Test",
                        value="Test1"
                    )
                    submit_btn = gr.Button("Process Test", variant="primary")

                with gr.Column():
                    test_info = gr.Text(label="Test Info")

            submit_btn.click(
                fn=(lambda x: x),
                inputs=[test_dropdown],
                outputs=[test_info]
            )

demo.launch(share=True)