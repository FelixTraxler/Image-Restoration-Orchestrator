import os
from pathlib import Path
from PIL import Image
import sys

def resize_images(input_dir, output_dir, max_size):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        with Image.open(img_file) as img:
            img.thumbnail((max_size, max_size))
            output_file = output_path / img_file.name
            img.save(output_file)
            print(f"Saved {output_file} ({img.size[0]}x{img.size[1]})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python resize_images.py <input_dir> <output_dir> <max_size>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    max_size = int(sys.argv[3])

    resize_images(input_dir, output_dir, max_size)
