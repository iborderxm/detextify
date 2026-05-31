"""Interfaces for text detection."""
import os
os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_onednn'] = 'False'
os.environ['FLAGS_use_mkldnn_bf16'] = 'False'

from absl import logging
from dataclasses import dataclass
from typing import Sequence
from PIL import Image

import time

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from paddleocr import PaddleOCR


@dataclass
class TextBox:
  # (x, y) is the top left corner of a rectangle; the origin of the coordinate system is the top-left of the image.
  # x denotes the vertical axis, y denotes the horizontal axis (to match the traditional indexing in a matrix).
  x: int
  y: int
  h: int
  w: int
  text: str = None


class TextDetector:
  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    pass


class AzureTextDetector(TextDetector):
  """Calls the Computer Vision endpoint from Microsoft Azure. Promises to work with images in the wild."""

  def __init__(self, endpoint, key):
    self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    read_response = self.client.read_in_stream(open(image_filename, "rb"), raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
      read_result = self.client.get_read_result(operation_id)
      if read_result.status not in ['notStarted', 'running']:
        break
      time.sleep(1)

    text_boxes = []
    if read_result.status == OperationStatusCodes.succeeded:
      for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
          # line.bounding_box contains the 4 corners of a polygon (not necessarily a rectangle).
          # To keep things simple, we turn them into rectangles. There are two ways: (1) use the rectangle
          # defined by the top-left and bottom-right corners, or (2) use the rectangle that encompasses the
          # entire polygon. (1) will lead to smaller surfaces, (2) to bigger surfaces.

          # Implementation for (1)
          # tl_x, tl_y = line.bounding_box[0:2]   # top left
          # br_x, br_y = line.bounding_box[4:6]   # bottom right
          # w = br_x - tl_x
          # h = br_y - tl_y

          # Implementation for (2)
          xs = [point for idx, point in enumerate(line.bounding_box) if idx % 2 == 0]
          ys = [point for idx, point in enumerate(line.bounding_box) if idx % 2 == 1]
          tl_x = min(xs)
          tl_y = min(ys)
          h = max(xs) - tl_x
          w = max(ys) - tl_y

          if h < 0 or w < 0:
            logging.error(f"Malformed bounding box from Azure: {line.bounding_box}")

          text_boxes.append(TextBox(int(tl_x), int(tl_y), int(h), int(w), line.text))
    return text_boxes


class PaddleOCRTextDetector(TextDetector):
  """Uses the `PaddleOCR` library from Baidu to do text detection."""

  def __init__(self, lang: str = 'ch', use_textline_orientation: bool = True, show_log: bool = False):
    """
    Args:
      lang: Language code, e.g. 'ch' (Chinese), 'en' (English).
      use_textline_orientation: Whether to use text line orientation (replaces use_angle_cls in newer PaddleOCR versions).
      show_log: Whether to show logs (now controlled via logging module).
    """
    import logging
    if not show_log:
      logging.getLogger('ppocr').setLevel(logging.WARNING)
    self.ocr = PaddleOCR(lang=lang, use_textline_orientation=use_textline_orientation)

  def detect_text(self, image_filename: str) -> Sequence[TextBox]:
    result = self.ocr.ocr(image_filename)
    text_boxes = []
    
    if result[0]:  # Check if any text was detected
      for line in result[0]:
        coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        text = line[1][0]
        
        # Calculate bounding box from polygon coordinates
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        tl_x = min(xs)
        tl_y = min(ys)
        h = max(xs) - tl_x
        w = max(ys) - tl_y
        
        if h >= 0 and w >= 0 and text.strip():
          text_boxes.append(TextBox(int(tl_y), int(tl_x), int(w), int(h), text))
    
    return text_boxes
