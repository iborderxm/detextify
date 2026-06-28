from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


class Translator:
    """Translate text using Hy-MT2-1.8B model."""

    DEFAULT_PROMPT = """将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：
{source_text}"""

    DEFAULT_HUGGINGFACE_REPO = "tencent/Hy-MT2-1.8B"

    def __init__(self, model_path: str):
        """Initialize the Translator.

        Args:
            model_path: Path to the Hy-MT2-1.8B model, or HuggingFace repo ID.
        """
        self._released = False
        
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            print(f"Model files found at {model_path}. Loading from local files...")
            self._load_local_model(model_path)
            print("Model loaded successfully from local files.")
        else:
            print(f"Model files not found at {model_path}. Downloading from HuggingFace...")
            self._load_huggingface_model(model_path)

    def _load_local_model(self, model_path: str):
        """Load model from local files."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        self.model.eval()
    

    def _load_huggingface_model(self, model_path: str):
        """Load model from HuggingFace, falling back to default repo if needed."""
        repo_id = model_path if "/" in model_path else self.DEFAULT_HUGGINGFACE_REPO
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=False)

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