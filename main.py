from detextify.text_detector import PaddleOCRTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier
from detextify.product_info_extractor import ProductInfoExtractor
import os

print("Using PaddleOCR for text detection")

# 1. Initialize PaddleOCR text detector
text_detector = PaddleOCRTextDetector(lang='ch', use_textline_orientation=True, show_log=False)

# 2. Initialize Qwen model for product info extraction
qwen_model_path = "/tmp/Qwen2.5-7B-Instruct"
product_extractor = ProductInfoExtractor(qwen_model_path)

# 3. Initialize LocalSDInpainter
model_path = "./mod"
if not os.path.exists(os.path.join(model_path, "model_index.json")):
    print(f"Model files not found at {model_path}, using default model from Hugging Face.")
    inpainter = LocalSDInpainter()
else:
    print(f"Using local model at: {model_path}")
    inpainter = LocalSDInpainter(model_path=model_path)

detextifier = Detextifier(text_detector, inpainter, product_extractor)
# detextifier.detextify("./data/1.jpg", "./data/2.jpg")
detextifier.detextify("./data/1.jpg", "./data/2.jpg", prompt="Remove marketing slogans unrelated to the main product and replace the Chinese translations with English")
