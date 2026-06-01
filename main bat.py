from detextify.text_detector import PaddleOCRTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier
import os
import sys

if len(sys.argv) != 3:
    print("Usage: python main_bat.py <source_dir> <output_dir>")
    sys.exit(1)

source_dir = sys.argv[1]
output_dir = sys.argv[2]

print("Using PaddleOCR for text detection")

text_detector = PaddleOCRTextDetector(lang='ch', use_textline_orientation=True, show_log=False)

model_path = "./mod"
if not os.path.exists(os.path.join(model_path, "model_index.json")):
    print(f"Model files not found at {model_path}, using default model from Hugging Face.")
    inpainter = LocalSDInpainter()
else:
    print(f"Using local model at: {model_path}")
    inpainter = LocalSDInpainter(model_path=model_path)

detextifier = Detextifier(text_detector, inpainter)

os.makedirs(output_dir, exist_ok=True)

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
image_files = [f for f in os.listdir(source_dir) if os.path.splitext(f.lower())[1] in image_extensions]

print(f"Found {len(image_files)} images in {source_dir}")

for filename in image_files:
    src_path = os.path.join(source_dir, filename)
    base_name = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    dst_path = os.path.join(output_dir, f"{base_name}_detextified{ext}")
    print(f"Processing: {filename} -> {base_name}_detextified{ext}")
    detextifier.detextify(src_path, dst_path)

print(f"Batch processing complete. Output saved to {output_dir}")
