"""LLM Baseline for comparison (using T5-large for efficient paraphrasing)."""

import torch
from typing import List, Optional, Union
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class LLMBaseline:
    """LLM Baseline for paraphrase generation comparison using T5-large."""

    # Simple prefix for T5 paraphrasing
    PARAPHRASE_PREFIX = "paraphrase: "

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: Optional[str] = None,
        **kwargs  # Accept but ignore legacy parameters like quantization
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine dtype based on device
        if self.device == "cuda":
            dtype = torch.float16
        elif self.device == "mps":
            dtype = torch.float32  # MPS works better with float32
        else:
            dtype = torch.float32

        # Load model (Seq2Seq for T5)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate paraphrase using T5-large."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        results = []
        for t in texts:
            # Add paraphrase prefix
            input_text = self.PARAPHRASE_PREFIX + t

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            # Calculate min_length based on input if not specified
            input_word_count = len(t.split())
            if min_length is None:
                min_length_tokens = int(input_word_count * 0.8)
            else:
                min_length_tokens = min_length

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length_tokens,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=3,
                )

            # Decode output
            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(paraphrase)

        return results[0] if is_single else results

