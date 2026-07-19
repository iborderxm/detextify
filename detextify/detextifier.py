import os
import shutil
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from detextify.inpainter import Inpainter
from detextify.text_detector import TextDetector
from detextify.product_info_extractor import ProductInfoExtractor
from detextify.upscaler import Upscaler
from detextify.translator import Translator


def find_microsoft_yahei_font():
    """查找微软雅黑字体文件路径"""
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",      # 常规
        r"C:\Windows\Fonts\msyhbd.ttc",    # 粗体
        r"C:\Windows\Fonts\msyhl.ttc",     # 细体
        "/Library/Fonts/Microsoft YaHei.ttc",  # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


def draw_text_with_stroke(draw, text, position, font, text_color, stroke_color, stroke_width=2):
    """在指定位置绘制带描边的文字（使用Pillow原生描边功能）"""
    x, y = position
    # Pillow 8.0+ 原生支持 stroke_width 和 stroke_fill 参数
    draw.text(
        (x, y), text, font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )
    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_multiline_text(draw, text, position, font, text_color, stroke_color, stroke_width, max_width):
    """绘制多行文字，自动换行"""
    x, y = position
    current_x, current_y = x, y
    
    # 按行分割
    lines = text.split('\n')
    for line in lines:
        if not line:
            current_y += font.size + 4
            continue
        
        # 如果单行过长，按空格分割
        words = line.split(' ')
        current_line = ''
        for word in words:
            test_line = current_line + (word if not current_line else ' ' + word)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
            
            if test_width <= max_width or not current_line:
                current_line = test_line
            else:
                # 绘制当前行
                tw, th = draw_text_with_stroke(draw, current_line, (current_x, current_y), font, text_color, stroke_color, stroke_width)
                current_y += th + 8
                current_line = word
        
        # 绘制最后一行
        if current_line:
            tw, th = draw_text_with_stroke(draw, current_line, (current_x, current_y), font, text_color, stroke_color, stroke_width)
            current_y += th + 8
    
    return current_y - y


def render_product_info(image_path, product_info, output_path, position=(0.05, 0.05), text_scale=1.0, use_bold=True):
    """在图像上渲染商品信息文字
    
    Args:
        image_path: 输入图像路径
        product_info: 商品信息文本
        output_path: 输出图像路径
        position: 文字起始位置比例 (x_ratio, y_ratio)，范围 0.0-1.0
        text_scale: 文字缩放因子，默认为 1.0
        use_bold: 是否使用粗体
    """
    image = Image.open(image_path).convert('RGBA')
    draw = ImageDraw.Draw(image)
    
    base_font_size = image.height // 40
    font_size = max(12, int(base_font_size * text_scale))
    
    base_stroke_width = max(1, font_size // 10)
    stroke_width = int(base_stroke_width * text_scale)
    
    x = int(image.width * position[0])
    y = int(image.height * position[1])
    rendered_position = (x, y)
    
    font_path = find_microsoft_yahei_font()
    if font_path is None:
        print("警告：未找到微软雅黑字体，使用默认字体")
        font = ImageFont.truetype('arial.ttf', font_size) if os.path.exists('arial.ttf') else ImageFont.load_default()
    else:
        font_index = 1 if use_bold and 'msyh.ttc' in font_path.lower() else 0
        font = ImageFont.truetype(font_path, font_size, index=font_index)
    
    text_color = (0, 0, 0, 200)
    stroke_color = (255, 255, 255, 200)
    
    max_width = image.width - rendered_position[0] - int(image.width * 0.05)
    
    draw_multiline_text(draw, product_info, rendered_position, font, text_color, stroke_color, stroke_width, max_width)
    
    image.save(output_path, quality=95)
    print(f"图像已保存到: {output_path}")
    return image


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter, 
                 product_info_extractor: Optional[ProductInfoExtractor] = None,
                 upscaler: Optional[Upscaler] = None,
                 translator: Optional[Translator] = None):
        self.text_detector = text_detector
        self.inpainter = inpainter
        self.product_info_extractor = product_info_extractor
        self.upscaler = upscaler
        self.translator = translator
        # 使用实例变量替代全局变量
        self.ocr_extraction_result: Optional[str] = None
        self.product_info_result: Optional[str] = None

    def _format_ocr_text(self, text_boxes):
        """Format text boxes into a string for OCR results."""
        lines = [f"Detected {len(text_boxes)} text boxes."]
        for idx, box in enumerate(text_boxes):
            lines.append(f"   Text Box {idx + 1}:")
            lines.append(f"\t Text: '{box.text}'")
            lines.append(f"\t Position: x={box.x}, y={box.y}, w={box.w}, h={box.h}")
        return "\n".join(lines)

    def detextify(self, in_image_path: str, out_image_path: str, 
                prompt: str = Inpainter.DEFAULT_PROMPT, 
                negative_prompt: str = Inpainter.DEFAULT_NEGATIVE_PROMPT,
                max_retries: int = 5,
                enable_upscale: bool = True,
                upscale_factor: int = 4,
                text_position: tuple = (0.05, 0.05),
                text_scale: float = 1.0,
                text_use_bold: bool = True) -> Tuple[bool, str]:
        """
        Remove text from image and optionally add product information.
        
        Args:
            in_image_path: Input image path
            out_image_path: Output image path
            prompt: Inpainting prompt
            negative_prompt: Negative prompt for inpainting
            max_retries: Maximum retries for text removal
            enable_upscale: Whether to enable upscaling (default: True)
            upscale_factor: Upscaling factor (default: 4)
            text_position: Text position as ratio (x_ratio, y_ratio) 0.0-1.0 (default: (0.05, 0.05))
            text_scale: Text scale factor relative to image height (default: 1.0)
            text_use_bold: Whether to use bold font (default: True)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        to_inpaint_path = in_image_path
        product_info_prompt = None
        
        try:
            # Extract product info using Qwen model (only first iteration)
            if self.product_info_extractor:
                print("\tCalling Qwen model to extract product info...")
                try:
                    self.product_info_result = self.product_info_extractor.extract_product_info(to_inpaint_path)
                    print(f"\tProduct info extracted: {self.product_info_result}")
                    # Release Qwen model after extraction
                    print("\tReleasing Qwen model to free CUDA memory...")
                    self.product_info_extractor.release()
                    self.product_info_extractor = None
                except Exception as e:
                    print(f"\tFailed to extract product info: {e}")
                    self.product_info_result = None

            for i in range(max_retries):
                print(f"Iteration {i} of {max_retries} for image {in_image_path}:")

                print("\tCalling text detector...")
                text_boxes = self.text_detector.detect_text(to_inpaint_path)
                print(f"\tDetected {len(text_boxes)} text boxes.")
                
                formatted_result = self._format_ocr_text(text_boxes)
                self.ocr_extraction_result = formatted_result
                print(f"\tOCR extraction result saved.")
                # 使用 _format_ocr_text 的结果进行打印，避免代码重复
                print("\t" + "\n\t".join(formatted_result.split("\n")))

                if not text_boxes:
                    print("\tNo text boxes detected, stopping iterations.")
                    break

                # Prepare prompt with product info
                product_info_prompt = None
                product_info_negative_prompt = None
                if self.product_info_result:
                    # 图片中文本全部移除后，后面的迭代中，只添加商品信息到图片，不移除文本
                    product_info_prompt = Inpainter.DEFAULT_PROMPT_OTHER.format(self.product_info_result)
                    product_info_negative_prompt = Inpainter.DEFAULT_NEGATIVE_PROMPT_OTHER
            
                print(f"\tCalling in-painting model with prompt: {prompt}")
                try:
                    self.inpainter.inpaint(to_inpaint_path, text_boxes, prompt, out_image_path, negative_prompt)
                except Exception as e:
                    print(f"\tInpainting failed: {e}")
                    if i < max_retries - 1:
                        print("\tRetrying...")
                        continue
                    return False, f"Inpainting failed after {max_retries} attempts: {e}"
                
                # 使用条件判断替代 assert
                if not os.path.exists(out_image_path):
                    return False, f"Inpainting did not produce output file: {out_image_path}"
                
                to_inpaint_path = out_image_path

            # 图片中文本全部移除后，添加商品信息到图片
            if self.product_info_result and self.product_info_result != "none" and product_info_prompt:
                print(f"\tAdding product info to image using LongCat model: {product_info_prompt}")
                try:
                    self.inpainter.inpaint(to_inpaint_path, None, product_info_prompt, out_image_path, product_info_negative_prompt)
                    print("\tProduct info added to image.")
                except Exception as e:
                    print(f"\tFailed to add product info: {e}")
                    return False, f"Failed to add product info: {e}"
            
            # 释放FLUX.2-klein-4B模型占用的显存
            if hasattr(self.inpainter, 'release_memory'):
                self.inpainter.release_memory()
            
            # 超分辨率放大（Upscaling）
            if enable_upscale and self.upscaler:
                print(f"\tUpscaling image by {upscale_factor}x...")
                try:
                    # Create a temporary path for upscaled image
                    base, ext = os.path.splitext(out_image_path)
                    upscaled_path = f"{base}_upscaled{ext}"
                    
                    self.upscaler.upscale(
                        in_image_path=out_image_path,
                        out_image_path=upscaled_path,
                        scale_factor=upscale_factor,
                        prompt=Upscaler.DEFAULT_UPSCALE_PROMPT,
                        negative_prompt=Upscaler.DEFAULT_UPSCALE_NEGATIVE_PROMPT
                    )
                    
                    # Replace original output with upscaled version
                    if os.path.exists(upscaled_path):
                        import shutil
                        shutil.move(upscaled_path, out_image_path)
                        print(f"\tImage upscaled successfully to {out_image_path}")
                except Exception as e:
                    print(f"\tUpscaling failed: {e}")
                    # Don't return error, just log it - the main task is done
            elif enable_upscale and not self.upscaler:
                print("\tUpscaling requested but no upscaler provided. Skipping upscaling.")
        
            return True, "Success"
            
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def detextifybat(self, in_image_path: str, out_dir_path: str, 
                prompt: str = Inpainter.DEFAULT_PROMPT, 
                negative_prompt: str = Inpainter.DEFAULT_NEGATIVE_PROMPT,
                max_retries: int = 5,
                enable_upscale: bool = True,
                upscale_factor: int = 4,
                text_position: tuple = (0.05, 0.05),
                text_scale: float = 1.5,
                text_use_bold: bool = True) -> Tuple[bool, str]:
        if not os.path.exists(in_image_path):
            return False, f"Input image path not found: {in_image_path}"

        shutil.rmtree(out_dir_path, ignore_errors=True)
        os.makedirs(out_dir_path, exist_ok=True)

        image_files = os.listdir(in_image_path)
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not image_files:
            return False, f"No image files found in {in_image_path}"

        print("\t------------------Extract product info------------------")
        for image_file in image_files:
            image_path = os.path.join(in_image_path, image_file)
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt") 

            # Extract product info using Qwen model (only first iteration)
            if self.product_info_extractor:
                try:
                    self.product_info_result = self.product_info_extractor.extract_product_info(image_path)
                    print(f"\tProduct info extracted: {self.product_info_result}")
                    # Release Qwen model after extraction
                    print("\tReleasing Qwen model to free CUDA memory...")
                    self.product_info_extractor.release()
                    self.product_info_extractor = None
                    
                    if not self.product_info_result:
                        continue
                    self.product_info_result = self.product_info_result.strip()
                    if self.product_info_result == "none":
                        continue
                    # 保存格式化文本到文件
                    with open(out_ocr_path, 'w', encoding='utf-8') as f:
                        f.write(self.product_info_result)
                    print(f"\tFormatted text saved to {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to extract product info: {e}")
                    self.product_info_result = None

        if self.product_info_extractor:
            print("\tReleasing Qwen model to free CUDA memory...")
            self.product_info_extractor.release()
            self.product_info_extractor = None

        # 执行翻译商品信息
        print("\t------------------Processing images product info translation------------------")
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt")

            if not os.path.exists(out_ocr_path):
                continue

            # 读取格式化文本
            with open(out_ocr_path, 'r', encoding='utf-8') as f:
                formatted_result = f.read().strip()

            if not formatted_result:
                print(f"\tNo formatted text found in {out_ocr_path}")
                # 删除格式化文本文件
                os.remove(out_ocr_path)
                continue

            if self.translator and formatted_result.strip():
                try:
                    translated_result = self.translator.translate(formatted_result)
                    print(f"\tTranslated product info: {translated_result}")

                    if translated_result and translated_result != "none":
                        with open(out_ocr_path, 'w', encoding='utf-8') as f:
                            f.write(translated_result)
                        print(f"\tTranslated product info saved to {out_ocr_path}")
                    else:
                        os.remove(out_ocr_path)
                        print(f"\tNo valid translation result, removed {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to translate product info for {image_file}: {e}")

        if self.translator:
            print("\tReleasing Hy-MT2-1.8B model to free CUDA memory...")
            self.translator.release()
            self.translator = None

        # 遍历图片移除文本
        print("\t------------------Removing text from images------------------")
        for image_file in image_files:
            image_path = os.path.join(in_image_path, image_file)
            out_image_path = os.path.join(out_dir_path, image_file)
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt")

            to_inpaint_path = image_path

            print(f"\t@@@@@ Current processing image {image_file}...")
            for i in range(max_retries):
                print(f"\tIteration {i} of {max_retries} for image {image_file}:")

                text_boxes = self.text_detector.detect_text(to_inpaint_path)
                print(f"\t\tDetected {len(text_boxes)} text boxes.")
                
                if not text_boxes:
                    if i != 0:
                        print("\t\tNo text boxes detected, stopping iterations.")
                        break

                print(f"\t\tCalling in-painting model with prompt: {prompt}")
                try:
                    self.inpainter.inpaint(to_inpaint_path, text_boxes, prompt, out_image_path, negative_prompt)
                except Exception as e:
                    print(f"\t\tInpainting failed: {e}")
                    if i < max_retries - 1:
                        print("\t\tRetrying...")
                        continue
                    print(f"\t\tInpainting failed after {max_retries} attempts: {e}")
                    break
                
                if not os.path.exists(out_image_path):
                    print(f"\t\tInpainting did not produce output file: {out_image_path}")
                    break
                
                to_inpaint_path = out_image_path

            if not os.path.exists(out_ocr_path):
                # 无需写入商品信息，继续下一个图片
                print(f"\t\tNo product info found in {out_ocr_path}, skip next image.")
                continue
            if os.path.exists(out_ocr_path):
                with open(out_ocr_path, 'r', encoding='utf-8') as f:
                    product_info = f.read().strip()
                    print(f"\t\tProduct info: {product_info}")
                    print(f"\t\tAdding product info to image using PIL with Microsoft YaHei font")
                    try:
                        render_product_info(
                            image_path=out_image_path,
                            product_info=product_info,
                            output_path=out_image_path,
                            position=text_position,
                            text_scale=text_scale,
                            use_bold=text_use_bold
                        )
                        print("\t\tProduct info added to image.")
                    except Exception as e:
                        print(f"\t\tFailed to add product info: {e}")

        if hasattr(self.inpainter, 'release_memory'):
            self.inpainter.release_memory()

        # 超分辨率放大（Upscaling）
        if enable_upscale and self.upscaler:
            print(f"\t------------------Upscaling images by {upscale_factor}x------------------")
            for image_file in image_files:
                out_image_path = os.path.join(out_dir_path, image_file)
                if not os.path.exists(out_image_path):
                    continue
                    
                try:
                    base, ext = os.path.splitext(out_image_path)
                    upscaled_path = f"{base}_upscaled{ext}"
                    
                    self.upscaler.upscale(
                        in_image_path=out_image_path,
                        out_image_path=upscaled_path,
                        scale_factor=upscale_factor,
                        prompt=Upscaler.DEFAULT_UPSCALE_PROMPT,
                        negative_prompt=Upscaler.DEFAULT_UPSCALE_NEGATIVE_PROMPT
                    )
                    
                    if os.path.exists(upscaled_path):
                        shutil.move(upscaled_path, out_image_path)
                        print(f"\t\tImage upscaled successfully to {out_image_path}")
                except Exception as e:
                    print(f"\t\tUpscaling failed for {image_file}: {e}")
        elif enable_upscale and not self.upscaler:
            print("\tUpscaling requested but no upscaler provided. Skipping upscaling.")

        return True, "Success"

    def detextifybat1(self, in_image_path: str, out_dir_path: str, 
                prompt: str = Inpainter.DEFAULT_PROMPT, 
                negative_prompt: str = Inpainter.DEFAULT_NEGATIVE_PROMPT,
                max_retries: int = 5,
                enable_upscale: bool = True,
                upscale_factor: int = 4,
                text_position: tuple = (0.05, 0.05),
                text_scale: float = 1.0,
                text_use_bold: bool = True) -> Tuple[bool, str]:
        if not os.path.exists(in_image_path):
            return False, f"Input image path not found: {in_image_path}"

        shutil.rmtree(out_dir_path, ignore_errors=True)
        os.makedirs(out_dir_path, exist_ok=True)

        image_files = os.listdir(in_image_path)
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not image_files:
            return False, f"No image files found in {in_image_path}"

        # 遍历 image_files，每个图片执行OCR和提取商品信息
        print("\t------------------Processing images OCR------------------")
        for image_file in image_files:
            image_path = os.path.join(in_image_path, image_file)
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt") 
            
            text_boxes = []
            for attempt in range(5):
                text_boxes = self.text_detector.detect_text(image_path)
                print(f"\tDetected {len(text_boxes)} text boxes in {image_file} (attempt {attempt + 1}/5).")
                if text_boxes:
                    break
            if not text_boxes:
                continue
            if len(text_boxes) == 0:
                continue
            formatted_result = self._format_ocr_text(text_boxes)
            if formatted_result:
                formatted_result = formatted_result.strip()
                with open(out_ocr_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_result)
                print(f"\tFormatted text saved to {out_ocr_path}")

        print("\t------------------Processing images product info------------------")
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt")

            if not os.path.exists(out_ocr_path):
                continue

            # 读取格式化文本
            with open(out_ocr_path, 'r', encoding='utf-8') as f:
                formatted_result = f.read().strip()

            if not formatted_result:
                print(f"\tNo formatted text found in {out_ocr_path}")
                # 删除格式化文本文件
                os.remove(out_ocr_path)
                continue

            if self.product_info_extractor and formatted_result.strip():
                try:
                    product_info_result = self.product_info_extractor.extract_product_info(formatted_result)
                    print(f"\tProduct info extracted: {product_info_result}")

                    if product_info_result:
                        product_info_result = product_info_result.strip()
                        if product_info_result == "none":
                            # 删除格式化文本文件
                            os.remove(out_ocr_path)
                            continue
                        # 执行提取商品信息
                        with open(out_ocr_path, 'a', encoding='utf-8') as f:
                            f.write(product_info_result)
                        print(f"\tProduct info saved to {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to extract product info for {image_file}: {e}")

        if self.product_info_extractor:
            print("\tReleasing Qwen model to free CUDA memory...")
            self.product_info_extractor.release()
            self.product_info_extractor = None

                # 执行翻译商品信息
        print("\t------------------Processing images product info translation------------------")
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            out_ocr_path = os.path.join(out_dir_path, f"{base_name}.txt")

            print(f"\t#########Processing {image_file}...")

            if not os.path.exists(out_ocr_path):
                continue

            # 读取格式化文本
            with open(out_ocr_path, 'r', encoding='utf-8') as f:
                formatted_result = f.read().strip()

            if not formatted_result:
                print(f"\tNo formatted text found in {out_ocr_path}")
                # 删除格式化文本文件
                os.remove(out_ocr_path)
                continue

            if self.translator and formatted_result.strip():
                try:
                    translated_result = self.translator.translate(formatted_result)
                    print(f"\tTranslated product info: {translated_result}")

                    if translated_result and translated_result != "none":
                        with open(out_ocr_path, 'w', encoding='utf-8') as f:
                            f.write(translated_result)
                        print(f"\tTranslated product info saved to {out_ocr_path}")
                    else:
                        os.remove(out_ocr_path)
                        print(f"\tNo valid translation result, removed {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to translate product info for {image_file}: {e}")

        if self.translator:
            print("\tReleasing Hy-MT2-1.8B model to free CUDA memory...")
            self.translator.release()
            self.translator = None

        return True, "Success"
