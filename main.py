from detextify.text_detector import TesseractTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier
import shutil
import os

tesseract_path = shutil.which("tesseract")
if tesseract_path is None:
    raise Exception("Tesseract not found in PATH! Please install Tesseract and make sure it's in your PATH.")
print(f"Using Tesseract at: {tesseract_path}")

text_detector = TesseractTextDetector(tesseract_path)

model_path = "./mod"
if not os.path.exists(os.path.join(model_path, "model_index.json")):
    print(f"Model files not found at {model_path}, using default model from Hugging Face.")
    inpainter = LocalSDInpainter()
else:
    print(f"Using local model at: {model_path}")
    inpainter = LocalSDInpainter(model_path=model_path)

detextifier = Detextifier(text_detector, inpainter)
detextifier.detextify("./data/1.jpg", "./data/2.jpg")
