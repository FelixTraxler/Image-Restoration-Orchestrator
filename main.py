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
        return None, "Invalid model selection"

    resized_image = ImageOps.contain(image, (256,256))
    resized_image.save(f"input_128/temp.png")

    subprocess.run(
        [str(project["venv"]), str(project["script"]), *project["args"]],
        cwd=project["cwd"],
        check=True,
    )

    output_image = Image.open(f"output_images/temp_{model}.png")

    return resized_image, output_image, f"Used model: {model}" 

demo = gr.Interface(
    fn=resize_image,
    inputs=[gr.Image(type="pil", label="Input Image"), gr.Dropdown(choices=[project["model"] for project in projects], label="Model")],
    outputs=[
        gr.Image(type="pil", label="Resized Image"), 
        gr.Image(type="pil", label="Output Image"),
        gr.Text(label="Resized Image Size"), 
    ],
    api_name="predict"
)

demo.launch(share=True)