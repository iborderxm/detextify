from detextify.text_detector import PaddleOCRTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier
import os

print("Using PaddleOCR for text detection")

text_detector = PaddleOCRTextDetector(lang='ch', use_angle_cls=True, show_log=False)

model_path = "./mod"
if not os.path.exists(os.path.join(model_path, "model_index.json")):
    print(f"Model files not found at {model_path}, using default model from Hugging Face.")
    inpainter = LocalSDInpainter()
else:
    print(f"Using local model at: {model_path}")
    inpainter = LocalSDInpainter(model_path=model_path)

detextifier = Detextifier(text_detector, inpainter)
detextifier.detextify("./data/1.jpg", "./data/2.jpg")
