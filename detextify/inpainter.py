"""In-painting models."""
import io
import math
import openai
import replicate
import requests
import tempfile
import torch

from PIL import Image, ImageDraw
from typing import Sequence

import detextify.utils as utils
from detextify.text_detector import TextBox


class Inpainter:
  """Interface for in-painting models."""
  DEFAULT_PROMPT = "Remove all text in the masked area, fill it with clean white background that seamlessly blends with the surrounding area. Keep the product unchanged and make it stand out clearly against the white background. Maintain high image quality and natural appearance."
  DEFAULT_PROMPT_OTHER = "Add the following product information to the image: {}. Place it in a suitable position. Use a professional, clear font with appropriate size and dark color (black or dark gray) that is easy to read against the white background. Ensure the text is accurate, undistorted, well-positioned, doesn't overlap with the product or important elements, and maintains a clean, professional e-commerce style that harmonizes with the overall image."
  DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, extra elements, watermark, artifacts, noisy, pixelated, disfigured, ugly, deformed, bad anatomy, extra limbs, missing limbs"
  DEFAULT_NEGATIVE_PROMPT_OTHER = "blurry, low quality, distorted, extra elements, watermark, artifacts, noisy, pixelated, disfigured, ugly, deformed, wrong text, misspelled, garbled text, overlapping text, illegible text"

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str, negative_prompt: str = None):
    pass


class DalleInpainter(Inpainter):
  """In-painting model that calls the DALL-E API."""

  def __init__(self, openai_key: str):
    openai.api_key = openai_key

  @staticmethod
  def _make_mask(text_boxes: Sequence[TextBox], height: int, width: int) -> bytes:
    """Returns an .png where the text boxes are transparent."""
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # fully opaque
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=(0, 0, 0, 0))  # fully transparent
    # Convert mask to bytes.
    bytes_arr = io.BytesIO()
    mask.save(bytes_arr, format="PNG")
    return bytes_arr.getvalue()

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str, negative_prompt: str = None):
    image = Image.open(in_image_path)  # open the image to inspect its size

    # DALL-E API may not support negative prompt directly, we'll add it to the prompt if provided
    full_prompt = prompt
    if negative_prompt:
        full_prompt = f"{prompt}. Do NOT include: {negative_prompt}"

    response = openai.Image.create_edit(
        image=open(in_image_path, "rb"),
        mask=self._make_mask(text_boxes, image.height, image.width),
        prompt=full_prompt,
        n=1,
        size=f"{image.height}x{image.width}"
    )
    url = response['data'][0]['url']
    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    out_image.save(out_image_path)


class StableDiffusionInpainter(Inpainter):
  """Abstract class for Stable Diffusion inpainters; suppoerts any input image size. Children must implement `call_model`."""

  def call_model(self, prompt: str, image: Image, mask: Image, negative_prompt: str = None) -> Image:
    pass  # To be implemented by children.

  def _tile_has_text_box(self, crop_x: int, crop_y: int, crop_size: int, text_boxes: Sequence[TextBox]):
    # Turn the tile into a TextBox just so that we can reuse utils.boxes_intersect
    crop_box = TextBox(crop_x, crop_y, crop_size, crop_size)
    return any([utils.boxes_intersect(crop_box, text_box) for text_box in text_boxes])

  def _pad_to_size(self, image, size):
    new_image = Image.new(image.mode, (size, size), color=(0, 0, 0))
    new_image.paste(image)
    return new_image

  def _make_mask(self, text_boxes: Sequence[TextBox], height: int, width: int, mode: str) -> Image:
    """Returns a black image with white rectangles where the text boxes are."""
    num_channels = len(mode)
    background_color = tuple([0] * num_channels)
    mask_color = tuple([255] * num_channels)

    mask = Image.new(mode, (width, height), background_color)
    mask_draw = ImageDraw.Draw(mask)
    for text_box in text_boxes:
      mask_draw.rectangle(xy=(text_box.x, text_box.y, text_box.x + text_box.h, text_box.y + text_box.w),
                          fill=mask_color)
    return mask

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str, negative_prompt: str = None):
    image = Image.open(in_image_path)
    mask_image = self._make_mask(text_boxes, image.height, image.width, image.mode)

    # SD only accepts images that are exactly 512 x 512.
    SD_SIZE = 512

    if image.height == SD_SIZE and image.width == SD_SIZE:
      out_image = self.call_model(prompt=prompt, image=image, mask=mask_image, negative_prompt=negative_prompt)
    else:
      # Break the image into 512 x 512 tiles. In-paint the tiles that contain text boxes.
      out_image = image.copy()

      # Used for the final out_image.paste; required to be in mode L.
      mask_binary = self._make_mask(text_boxes, image.height, image.width, "L")

      for x in range(0, image.height, SD_SIZE):
        for y in range(0, image.width, SD_SIZE):
          if self._tile_has_text_box(x, y, SD_SIZE, text_boxes):
            crop_x1 = min(x + SD_SIZE, image.height)
            crop_y1 = min(y + SD_SIZE, image.width)
            crop_box = (x, y, crop_x1, crop_y1)

            in_tile = self._pad_to_size(image.crop(crop_box), SD_SIZE)
            in_mask = self._pad_to_size(mask_image.crop(crop_box), SD_SIZE)
            out_tile = self.call_model(prompt=prompt, image=in_tile, mask=in_mask, negative_prompt=negative_prompt)
            out_tile = out_tile.crop((0, 0, crop_x1 - x, crop_y1 - y))
            out_mask = mask_binary.crop(crop_box)
            out_image.paste(out_tile, (x, y), out_mask)

    out_image.save(out_image_path)


class ReplicateSDInpainter(StableDiffusionInpainter):
  SD_INPAINTING_V2 = "cjwbw/stable-diffusion-v2-inpainting"
  SD_INPAINTING_V2_VERSION = "f9bb0632bfdceb83196e85521b9b55895f8ff3d1d3b487fd1973210c0eb30bec"

  def __init__(self, replicate_token: str, model_name=SD_INPAINTING_V2, model_version=SD_INPAINTING_V2_VERSION):
    replicate_client = replicate.Client(api_token=replicate_token)
    self.model = replicate_client.models.get(model_name).versions.get(model_version)

  def call_model(self, prompt: str, image: Image, mask: Image, negative_prompt: str = None) -> Image:
    # Replicate expects a file object as an input.
    img_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    image.save(img_temp_file)
    mask_temp_file = tempfile.NamedTemporaryFile(suffix=".jpeg")
    mask.save(mask_temp_file)

    predict_kwargs = {
        "prompt": prompt,
        "prompt_strength": 1.0,
        "image": open(img_temp_file.name, "rb"),
        "mask": open(mask_temp_file.name, "rb"),
        "num_outputs": 1
    }
    if negative_prompt:
        predict_kwargs["negative_prompt"] = negative_prompt

    url = self.model.predict(**predict_kwargs)[0]
    out_image_data = requests.get(url).content
    out_image = Image.open(io.BytesIO(out_image_data))
    return out_image


class LocalSDInpainter(Inpainter):
  """Uses meituan-longcat/LongCat-Image-Edit-Turbo model for instruction-based image editing."""

  def __init__(self, model_path: str = None, pipe=None):
    if pipe is not None:
      self.pipe = pipe
      return

    if model_path is None:
      model_path = "meituan-longcat/LongCat-Image-Edit-Turbo"

    if not torch.cuda.is_available():
      raise Exception("You need a GPU + CUDA to run this model locally.")

    from diffusers import LongCatImageEditPipeline
    self.pipe = LongCatImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16)
    self.pipe.enable_model_cpu_offload()

  def call_model(self, prompt: str, image: Image, negative_prompt: str = None) -> Image:
    pipe_kwargs = {
        "image": image,
        "prompt": prompt,
        "guidance_scale": 1,
        "num_inference_steps": 8
    }
    if negative_prompt:
        pipe_kwargs["negative_prompt"] = negative_prompt
    
    return self.pipe(**pipe_kwargs).images[0]

  def inpaint(self, in_image_path: str, text_boxes: Sequence[TextBox], prompt: str, out_image_path: str, negative_prompt: str = None):
    image = Image.open(in_image_path)
    out_image = self.call_model(prompt=prompt, image=image, negative_prompt=negative_prompt)
    out_image.save(out_image_path)

  def release_memory(self):
    """释放LongCat-Image-Edit-Turbo模型占用的显存"""
    if hasattr(self, 'pipe') and self.pipe is not None:
      self.pipe = self.pipe.to('cpu')
      del self.pipe
      self.pipe = None
      torch.cuda.empty_cache()
      print("\tLongCat model memory released.")

