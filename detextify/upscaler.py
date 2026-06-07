"""Image upscaling models."""
import torch
from PIL import Image
from typing import Optional


class Upscaler:
    """Interface for image upscaling models."""
    
    DEFAULT_UPSCALE_PROMPT = "high quality, detailed, professional photography, clean white background"
    DEFAULT_UPSCALE_NEGATIVE_PROMPT = "blurry, low quality, distorted, artifacts, noise, pixelated"
    
    def upscale(self, in_image_path: str, out_image_path: str, scale_factor: int = 4, 
                prompt: Optional[str] = None, negative_prompt: Optional[str] = None) -> None:
        """Upscale the image by the given scale factor.
        
        Args:
            in_image_path: Path to the input image
            out_image_path: Path to save the upscaled image
            scale_factor: Upscaling factor (default: 4)
            prompt: Optional prompt to guide upscaling
            negative_prompt: Optional negative prompt
        """
        pass


class StableDiffusionUpscaler(Upscaler):
    """Uses Stable Diffusion for image upscaling."""
    
    def __init__(self, model_path: str = "stabilityai/stable-diffusion-x4-upscaler"):
        """Initialize the upscaler.
        
        Args:
            model_path: HuggingFace model path for the upscaler
        """
        if not torch.cuda.is_available():
            raise Exception("You need a GPU + CUDA to run this model locally.")
        
        from diffusers import StableDiffusionUpscalePipeline
        
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        self.pipe.enable_model_cpu_offload()
    
    def upscale(self, in_image_path: str, out_image_path: str, scale_factor: int = 4,
                prompt: Optional[str] = None, negative_prompt: Optional[str] = None) -> None:
        """Upscale the image using Stable Diffusion.
        
        Args:
            in_image_path: Path to the input image
            out_image_path: Path to save the upscaled image
            scale_factor: Upscaling factor (default: 4, model supports up to 4x)
            prompt: Optional prompt to guide upscaling
            negative_prompt: Optional negative prompt
        """
        if prompt is None:
            prompt = self.DEFAULT_UPSCALE_PROMPT
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_UPSCALE_NEGATIVE_PROMPT
        
        # Load image
        image = Image.open(in_image_path).convert("RGB")
        
        # Upscale
        upscaled_image = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        # Save
        upscaled_image.save(out_image_path)


class RealESRGANUpscaler(Upscaler):
    """Uses Real-ESRGAN for image upscaling (simpler and faster)."""
    
    def __init__(self, model_path: str = "RealESRGAN_x4plus"):
        """Initialize the upscaler.
        
        Args:
            model_path: Model name or path for Real-ESRGAN
        """
        try:
            import realesrgan
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            raise ImportError(
                "Real-ESRGAN requires basicsr and realesrgan packages. "
                "Install with: pip install basicsr realesrgan"
            )
        
        if not torch.cuda.is_available():
            raise Exception("You need a GPU + CUDA to run this model locally.")
        
        # Initialize model
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = realesrgan.RealESRGANer(
            scale=4,
            model_path=f"weights/{model_path}.pth",
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True
        )
    
    def upscale(self, in_image_path: str, out_image_path: str, scale_factor: int = 4,
                prompt: Optional[str] = None, negative_prompt: Optional[str] = None) -> None:
        """Upscale the image using Real-ESRGAN.
        
        Args:
            in_image_path: Path to the input image
            out_image_path: Path to save the upscaled image
            scale_factor: Upscaling factor (default: 4)
            prompt: Not used for Real-ESRGAN
            negative_prompt: Not used for Real-ESRGAN
        """
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(in_image_path, cv2.IMREAD_COLOR)
        
        # Upscale
        output, _ = self.upsampler.enhance(img, outscale=scale_factor)
        
        # Save
        cv2.imwrite(out_image_path, output)
