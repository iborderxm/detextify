from detextify.text_detector import TesseractTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier

text_detector = TesseractTextDetector("/usr/include/tesseract")
detextifier = Detextifier(text_detector, LocalSDInpainter(model_path="../mod"))
detextifier.detextify("../data/1.jpg", "../data/2.jpg")
