# Image Restoration App

This project is a [Gradio](https://www.gradio.app/) app that allows users to easily try out a few different image restoration models. The following models are available:

* Super Resolution: X-Restormer
* Brighten up image: DarkIR
* BW to color: DeOldify
* Image to 3D Scene: VGGT

## Setup

This project uses Git Submodules for the underlying models. Install each of the 4 models by going into the folder and creating a new python environment:

```python
python -m venv venv
```

Install all packages:

```python
pip install -r requirements.txt
```

You may need to download some additional model files for each model. Refer to the README.md's in the respective model folders.

Finally, install Gradio:

```python
pip install --upgrade gradio
```

## Run Gradio app

The Gradio app can be run locally or publicly exposed. If you just want to run locally, set `share=False` in `main.py` other `share=True`.

`python main.py`
