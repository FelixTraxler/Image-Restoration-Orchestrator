import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

projects = [
    {
        "venv": BASE_DIR / "DarkIR/venv_DarkIR/bin/python",
        "script": BASE_DIR / "DarkIR/inference.py",
        "args": ["-i", "../input_128/", "-o", "../output_images/"],
        "cwd": BASE_DIR / "DarkIR",  # important: run inside the project folder
    },
    {
        "venv": BASE_DIR / "X-Restormer/venv/bin/python",
        "script": BASE_DIR / "X-Restormer/xrestormer/test.py",
        "args": ["-opt", "options/test/001_xrestormer_sr.yml"],
        "cwd": BASE_DIR / "X-Restormer",
    },
]

for proj in projects:
    print(f"Running {proj['script']}...")
    subprocess.run(
        [str(proj["venv"]), str(proj["script"]), *proj["args"]],
        cwd=proj["cwd"],
        check=True,
    )