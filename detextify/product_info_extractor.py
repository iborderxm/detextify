from modelscope import AutoModelForCausalLM, AutoTokenizer


class ProductInfoExtractor:
    """Extract product information from OCR text using Qwen model."""

    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def extract_product_info(self, ocr_text: str) -> str:
        """Extract product information from OCR text.

        Args:
            ocr_text: OCR recognition result string.

        Returns:
            Extracted product info as formatted string.
        """
        prompt = f"""请从以下OCR识别结果中提取商品信息，移除营销词（如工厂名、联系方式、广告语等），只保留商品相关描述。所有输出内容必须翻译为英文。

OCR识别结果：
{ocr_text}

请根据商品类型自动识别并提取相关属性，输出格式如下：
属性名1: 属性值1
属性名2: 属性值2
...

注意：
1. 只输出包含商品信息的字段，没有的字段不要输出
2. 属性名和属性值都必须使用英文
3. 根据商品类型选择合适的属性名（如 Brand、Model、Year、Size、Color、Material 等）"""

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
