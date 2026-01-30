"""Custom Paraphrase Generator using fine-tuned T5/FLAN-T5."""

import torch
from typing import Dict, List, Optional, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class CustomParaphraseGenerator:
    """Custom Paraphrase Generator based on fine-tuned T5."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None
    ):
        import os
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.use_lora = use_lora

        # Check if model_name is a local path
        is_local = os.path.isdir(model_name)

        # Load tokenizer and model (use local_files_only if local path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=is_local
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            local_files_only=is_local
        )

        # Apply LoRA if specified
        if use_lora:
            self._apply_lora(lora_config)

        self.model.to(self.device)
    
    def _apply_lora(self, config: Optional[Dict] = None):
        """Apply LoRA configuration for efficient fine-tuning."""
        default_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q", "v"],
            "task_type": TaskType.SEQ_2_SEQ_LM
        }
        
        if config:
            default_config.update(config)
        
        lora_config = LoraConfig(**default_config)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        max_input_length: int = 1024,
        min_length: Optional[int] = None,
        min_length_ratio: float = 0.8,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = True,
        length_penalty: float = 2.0,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate paraphrase for given text.

        Args:
            text: Input text or list of texts to paraphrase
            max_length: Maximum output length in tokens
            max_input_length: Maximum input length in tokens (default: 1024 for 200-400 word inputs)
            min_length: Minimum output length in tokens (overrides min_length_ratio)
            min_length_ratio: Minimum output length as ratio of input (default: 0.8 = 80%)
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            no_repeat_ngram_size: Prevent repeating n-grams
            do_sample: Whether to use sampling
            length_penalty: Exponential penalty for length (>1 encourages longer outputs)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Add prefix for paraphrasing
        inputs = [f"paraphrase: {t}" for t in texts]

        # Tokenize with separate max_input_length to avoid truncating long inputs
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt"
        ).to(self.device)

        # Calculate min_length in TOKENS based on input if not specified
        # Use tokenizer to get accurate token count, apply min_length_ratio
        if min_length is None:
            # Tokenize original texts to get token counts
            input_token_counts = [
                len(self.tokenizer.encode(t, add_special_tokens=False))
                for t in texts
            ]
            # Set min_length to min_length_ratio of the shortest input
            min_length = int(min(input_token_counts) * min_length_ratio)
            # Ensure min_length is at least 50 tokens for meaningful output
            min_length = max(min_length, 50)

        # Generate with length penalty to encourage longer outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=do_sample,
                length_penalty=length_penalty,
                early_stopping=False,  # Don't stop early to encourage longer outputs
                **kwargs
            )

        # Decode
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return results[0] if is_single else results

    def generate_diverse(
        self,
        text: str,
        num_paraphrases: int = 5,
        max_length: int = 512,
        max_input_length: int = 1024,
        min_length: Optional[int] = None,
        min_length_ratio: float = 0.8,
        num_beams: int = 5,
        num_beam_groups: int = 5,
        diversity_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 2.0,
        **kwargs
    ) -> List[str]:
        """Generate multiple diverse paraphrases using Diverse Beam Search.

        Diverse Beam Search generates multiple candidate sequences that are
        meaningfully different from each other by penalizing similar beams.

        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of diverse paraphrases to generate
            max_length: Maximum output length in tokens
            max_input_length: Maximum input length in tokens (default: 1024 for 200-400 word inputs)
            min_length: Minimum output length in tokens (overrides min_length_ratio)
            min_length_ratio: Minimum output length as ratio of input (default: 0.8 = 80%)
            num_beams: Total number of beams (must be divisible by num_beam_groups)
            num_beam_groups: Number of beam groups for diversity
            diversity_penalty: Strength of diversity penalty (higher = more diverse)
            no_repeat_ngram_size: Prevent repeating n-grams
            length_penalty: Exponential penalty for length (>1 encourages longer outputs)

        Returns:
            List of diverse paraphrases
        """
        # Ensure num_beams is divisible by num_beam_groups
        if num_beams % num_beam_groups != 0:
            num_beams = num_beam_groups * (num_beams // num_beam_groups + 1)

        # Add prefix for paraphrasing
        input_text = f"paraphrase: {text}"

        # Tokenize with separate max_input_length to avoid truncating long inputs
        encoded = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt"
        ).to(self.device)

        # Calculate min_length in TOKENS based on input if not specified
        if min_length is None:
            input_token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            min_length = int(input_token_count * min_length_ratio)
            min_length = max(min_length, 50)  # Ensure minimum of 50 tokens

        # Generate with Diverse Beam Search
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=num_paraphrases,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                do_sample=False,  # Diverse beam search doesn't use sampling
                early_stopping=False,  # Don't stop early to encourage longer outputs
                **kwargs
            )

        # Decode all sequences
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        return unique_results[:num_paraphrases]
    
    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None, base_model_path: Optional[str] = None):
        """Load saved model.

        Args:
            path: Path to the saved model (can be LoRA adapters or full model)
            device: Device to load model on
            base_model_path: Path to local base model (to avoid downloading)
        """
        import os
        import json
        from peft import PeftModel

        instance = cls.__new__(cls)
        instance.device = device or ("cuda" if torch.cuda.is_available() else
                                      "mps" if torch.backends.mps.is_available() else "cpu")
        instance.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

        # Check if this is a LoRA adapter (has adapter_config.json)
        adapter_config_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # Load adapter config to get base model info
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)

            # Determine base model path
            if base_model_path and os.path.exists(base_model_path):
                base_path = base_model_path
                local_only = True
            else:
                # Try default local path
                default_local = "./outputs/pretrained/flan-t5-base"
                if os.path.exists(default_local):
                    base_path = default_local
                    local_only = True
                else:
                    base_path = adapter_config.get("base_model_name_or_path", "google/flan-t5-base")
                    local_only = False

            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_path, local_files_only=local_only)
            # Load LoRA adapters on top
            instance.model = PeftModel.from_pretrained(base_model, path)
            instance.use_lora = True
        else:
            # Full model (not LoRA)
            instance.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
            instance.use_lora = False

        instance.model.to(instance.device)
        return instance

