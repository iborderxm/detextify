from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


class ProductInfoExtractor:
    """Extract product information from OCR text using Qwen model."""

    # Default prompt template with {ocr_text} placeholder
    DEFAULT_PROMPT = """请从以下OCR识别结果中提取商品信息，移除营销词（如工厂名、联系方式、广告语等），只保留商品相关描述。所有输出内容必须翻译为英文。

OCR识别结果：
{ocr_text}

请根据商品类型自动识别并提取相关属性，输出格式如下：
属性名1: 属性值1
属性名2: 属性值2
...

注意：
1. 请简要回答,只输出包含商品信息的字段，没有的字段不要输出,
2. 如果提取不到商品信息，请输出“none”,不要输出其他内容
3. 根据商品类型选择合适的属性名（如 Brand、Model、Year、Size、Color、Material 等）"""

    def __init__(self, model_path: str, custom_prompt: str = None):
        """Initialize the ProductInfoExtractor.

        Args:
            model_path: Path to the Qwen model.
            custom_prompt: Custom prompt template with {ocr_text} placeholder.
                          If None, the default prompt will be used.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._released = False
        self.custom_prompt = custom_prompt

    def release(self):
        """Release the model and free up CUDA memory."""
        if not self._released:
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU first to ensure proper cleanup
                self.model = self.model.to('cpu')
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            # Clear CUDA cache
            torch.cuda.empty_cache()
            self._released = True
            print("Qwen model released and CUDA memory freed.")

    def extract_product_info(self, ocr_text: str, custom_prompt: str = None) -> str:
        """Extract product information from OCR text.

        Args:
            ocr_text: OCR recognition result string.
            custom_prompt: Custom prompt template with {ocr_text} placeholder.
                          If None, uses the custom_prompt from __init__ or the default.

        Returns:
            Extracted product info as formatted string.
        """
        # Determine which prompt to use (method-level > class-level > default)
        prompt_template = custom_prompt or self.custom_prompt or self.DEFAULT_PROMPT
        prompt = prompt_template.format(ocr_text=ocr_text)

        messages = [
            {"role": "system", "content": "你是一个专业的商品信息提取助手，擅长从OCR识别结果中提取关键商品信息并过滤营销内容。"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
