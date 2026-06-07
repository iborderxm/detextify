from detextify.inpainter import Inpainter
from detextify.text_detector import TextDetector
from detextify.product_info_extractor import ProductInfoExtractor


# Global variable to store OCR extraction results for later use with Qwen model
ocr_extraction_result = None
# Global variable to store extracted product info
product_info_result = None


class Detextifier:
    def __init__(self, text_detector: TextDetector, inpainter: Inpainter, product_info_extractor: ProductInfoExtractor = None):
        self.text_detector = text_detector
        self.inpainter = inpainter
        self.product_info_extractor = product_info_extractor

    def _format_ocr_text(self, text_boxes):
        """Format text boxes into a string for OCR results."""
        lines = [f"Detected {len(text_boxes)} text boxes."]
        for idx, box in enumerate(text_boxes):
            lines.append(f"   Text Box {idx + 1}:")
            lines.append(f"\t Text: '{box.text}'")
            lines.append(f"\t Position: x={box.x}, y={box.y}, w={box.w}, h={box.h}")
        return "\n".join(lines)

    def _merge_product_info_into_prompt(self, prompt: str, product_info: str) -> str:
        """Merge product information into the inpainting prompt."""
        if not product_info:
            return prompt
        return f"{prompt}. Product information: {product_info}"

    def detextify(self, in_image_path: str, out_image_path: str, prompt=Inpainter.DEFAULT_PROMPT, max_retries=5):
        global ocr_extraction_result, product_info_result
        to_inpaint_path = in_image_path
        for i in range(max_retries):
            print(f"Iteration {i} of {max_retries} for image {in_image_path}:")

            print(f"\tCalling text detector...")
            text_boxes = self.text_detector.detect_text(to_inpaint_path)
            print(f"\tDetected {len(text_boxes)} text boxes.")
            
            formatted_result = self._format_ocr_text(text_boxes)
            ocr_extraction_result = formatted_result
            print(f"\tOCR extraction result saved to global variable.")
            
            for idx, box in enumerate(text_boxes):
                print(f"\t  Text Box {idx + 1}:")
                print(f"\t    Text: '{box.text}'")
                print(f"\t    Position: x={box.x}, y={box.y}, w={box.w}, h={box.h}")

            if not text_boxes:
                break

            # Extract product info using Qwen model
            if i == 0 and self.product_info_extractor:
                print(f"\tCalling Qwen model to extract product info...")
                try:
                    product_info_result = self.product_info_extractor.extract_product_info(formatted_result)
                    print(f"\tProduct info extracted: {product_info_result}")
                    # Release Qwen model after extraction to free up CUDA memory for inpainting
                    print(f"\tReleasing Qwen model to free CUDA memory...")
                    self.product_info_extractor.release()
                    self.product_info_extractor = None
                except Exception as e:
                    print(f"\tFailed to extract product info: {e}")
                    product_info_result = None

            # Prepare prompt with product info
            if product_info_result:
                current_prompt = f"{prompt}, {Inpainter.DEFAULT_PROMPT_OTHER.format(product_info_result)}"
            else:
                current_prompt = prompt
            
            print(f"\tCalling in-painting model with prompt: {current_prompt}")
            self.inpainter.inpaint(to_inpaint_path, text_boxes, current_prompt, out_image_path)
            import os
            assert os.path.exists(out_image_path)
            to_inpaint_path = out_image_path

