from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch


class Translator:
    """Translate text using Hy-MT2-1.8B model."""

    DEFAULT_PROMPT = """将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：
{source_text}"""

    def __init__(self, model_path: str):
        """Initialize the Translator.

        Args:
            model_path: Path to the Hy-MT2-1.8B model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._released = False

    def release(self):
        """Release the model and free up CUDA memory."""
        if not self._released:
            if hasattr(self, 'model') and self.model is not None:
                self.model = self.model.to('cpu')
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            torch.cuda.empty_cache()
            self._released = True
            print("Hy-MT2-1.8B model released and CUDA memory freed.")

    def translate(self, source_text: str, target_lang: str = "英语") -> str:
        """Translate source text to target language.

        Args:
            source_text: Text to translate.
            target_lang: Target language name (default: "英语").

        Returns:
            Translated text as string.
        """
        prompt = self.DEFAULT_PROMPT.format(source_text=source_text)

        messages = [{"role": "user", "content": prompt}]

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
        return response.strip()