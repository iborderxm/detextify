from detextify.text_detector import PaddleOCRTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier
from detextify.product_info_extractor import ProductInfoExtractor
from detextify.upscaler import StableDiffusionUpscaler, RealESRGANUpscaler
import os
import datetime

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

# 4. Initialize Upscaler (选择一种超分辨率模型)
# 选项 1: Stable Diffusion Upscaler (更高质量，但需要更多显存)
# 使用方法: 首次运行会自动下载模型 (~5GB)
# try:
#     print("Initializing Stable Diffusion Upscaler...")
#     upscaler = StableDiffusionUpscaler()
#     print("Stable Diffusion Upscaler initialized successfully.")
# except Exception as e:
#     print(f"Failed to initialize Stable Diffusion Upscaler: {e}")
#     upscaler = None

# 选项 2: Real-ESRGAN Upscaler (更快速，显存占用较少)
# 使用方法: 需要安装依赖 `pip install tb-nightly==2.14.0a20230808 basicsr==1.4.2 realesrgan==0.3.0==4.10.0.84  -i https://mirrors.aliyun.com/pypi/simple`
# 并下载模型权重到 weights/ 目录

    print("Initializing Real-ESRGAN Upscaler...")
    upscaler = RealESRGANUpscaler()
    print("Real-ESRGAN Upscaler initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Real-ESRGAN Upscaler: {e}")
    upscaler = None

# 5. Create Detextifier with upscaler
detextifier = Detextifier(
    text_detector=text_detector,
    inpainter=inpainter,
    product_info_extractor=product_extractor,
    upscaler=upscaler
)

# 6. Process image
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"./data/{timestamp}.jpg"

print(f"Processing image...")
success, message = detextifier.detextify(
    in_image_path="./data/1.jpg",
    out_image_path=output_path,
    enable_upscale=True,  # 启用超分辨率
    upscale_factor=4      # 4倍放大
)

if success:
    print(f"Image processed successfully: {output_path}")
else:
    print(f"Image processing failed: {message}")

# 其他使用示例:
# detextifier.detextify("./data/1.jpg", "./data/output.jpg", prompt="Remove marketing slogans unrelated to the main product and replace the Chinese translations with English")
# detextifier.detextify("./data/1.jpg", "./data/output.jpg", enable_upscale=False)  # 禁用超分辨率
