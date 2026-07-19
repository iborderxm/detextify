from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os


class ProductInfoExtractor:
    """Extract product information from OCR text using Qwen model."""

    DEFAULT_PROMPT = """请识别提取图像中的全部文本,识别结果需要移除营销广告词(工厂或公司名称、电子邮箱、手机号码、电话号码、qq号、微信、抖音、快手、网址等)，只保留商品相关信息(没有商品信息的输出none)"""

    DEFAULT_HUGGINGFACE_REPO = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self, model_path: str, custom_prompt: str = None):
        """Initialize the ProductInfoExtractor.

        Args:
            model_path: Path to the Qwen model, or HuggingFace repo ID.
            custom_prompt: Custom prompt template with {ocr_text} placeholder.
                          If None, the default prompt will be used.
        """
        self._released = False
        self.custom_prompt = custom_prompt
        
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            self._load_local_model(model_path)
        else:
            print(f"Model files not found at {model_path}. Downloading from HuggingFace...")
            self._load_huggingface_model(model_path)

    def _load_local_model(self, model_path: str):
        """Load model from local files."""
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    def _load_huggingface_model(self, model_path: str):
        """Load model from HuggingFace, falling back to default repo if needed."""
        repo_id = model_path if "/" in model_path else self.DEFAULT_HUGGINGFACE_REPO
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            dtype="auto",
            device_map="auto",
            local_files_only=False
        )
        self.processor = AutoProcessor.from_pretrained(repo_id, local_files_only=False)

    def release(self):
        """Release the model and free up CUDA memory."""
        if not self._released:
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU first to ensure proper cleanup
                self.model = self.model.to('cpu')
                del self.model
                self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            # Clear CUDA cache
            torch.cuda.empty_cache()
            self._released = True
            print("Qwen model released and CUDA memory freed.")

    def extract_product_info(self, image_path: str, custom_prompt: str = None) -> str:
        """Extract product information from OCR text.

        Args:
            image_path: Path to the image file.
            custom_prompt: Custom prompt template with {ocr_text} placeholder.
                          If None, uses the custom_prompt from __init__ or the default.

        Returns:
            Extracted product info as formatted string.
        """
        # Determine which prompt to use (method-level > class-level > default)
        prompt = custom_prompt or self.custom_prompt or self.DEFAULT_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
