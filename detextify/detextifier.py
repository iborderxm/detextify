import os
import shutil
from typing import Optional, Tuple

from detextify.inpainter import Inpainter
from detextify.text_detector import TextDetector
from detextify.product_info_extractor import ProductInfoExtractor
from detextify.upscaler import Upscaler


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter, 
                 product_info_extractor: Optional[ProductInfoExtractor] = None,
                 upscaler: Optional[Upscaler] = None):
        self.text_detector = text_detector
        self.inpainter = inpainter
        self.product_info_extractor = product_info_extractor
        self.upscaler = upscaler
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
                upscale_factor: int = 4) -> Tuple[bool, str]:
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
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        to_inpaint_path = in_image_path
        product_info_prompt = None
        
        try:
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

                # Extract product info using Qwen model (only first iteration)
                if i == 0 and self.product_info_extractor:
                    print("\tCalling Qwen model to extract product info...")
                    try:
                        self.product_info_result = self.product_info_extractor.extract_product_info(formatted_result)
                        print(f"\tProduct info extracted: {self.product_info_result}")
                        # Release Qwen model after extraction
                        print("\tReleasing Qwen model to free CUDA memory...")
                        self.product_info_extractor.release()
                        self.product_info_extractor = None
                    except Exception as e:
                        print(f"\tFailed to extract product info: {e}")
                        self.product_info_result = None

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
            if self.product_info_result and product_info_prompt:
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
                upscale_factor: int = 4) -> Tuple[bool, str]:
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
                        with open(out_ocr_path, 'w', encoding='utf-8') as f:
                            f.write(product_info_result)
                        print(f"\tProduct info saved to {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to extract product info for {image_file}: {e}")

        if self.product_info_extractor:
            print("\tReleasing Qwen model to free CUDA memory...")
            self.product_info_extractor.release()
            self.product_info_extractor = None

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
                    product_info_prompt = Inpainter.DEFAULT_PROMPT_OTHER.format(product_info)
                    product_info_negative_prompt = Inpainter.DEFAULT_NEGATIVE_PROMPT_OTHER
                    print(f"\t\tAdding product info to image using LongCat model: {product_info_prompt}")
                    try:
                        self.inpainter.inpaint(out_image_path, None, product_info_prompt, out_image_path, product_info_negative_prompt)
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
                upscale_factor: int = 4) -> Tuple[bool, str]:
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
                        with open(out_ocr_path, 'a', encoding='utf-8') as f:
                            f.write('\n\n' + product_info_result + '\n')
                        print(f"\tProduct info saved to {out_ocr_path}")
                except Exception as e:
                    print(f"\tFailed to extract product info for {image_file}: {e}")

        if self.product_info_extractor:
            print("\tReleasing Qwen model to free CUDA memory...")
            self.product_info_extractor.release()
            self.product_info_extractor = None

        return True, "Success"
