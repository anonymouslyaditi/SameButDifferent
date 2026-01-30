"""Back-translation data augmentation for paraphrase generation."""

import random
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class BackTranslationAugmenter:
    """Augment paraphrase data using back-translation.
    
    Back-translation works by:
    1. Translating source text from English to an intermediate language
    2. Translating back to English
    
    This produces semantically similar but lexically different text,
    which helps improve paraphrase model robustness and diversity.
    """
    
    # Translation model pairs: (en->lang, lang->en)
    TRANSLATION_MODELS = {
        "de": ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
        "fr": ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
        "es": ("Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-es-en"),
        "ru": ("Helsinki-NLP/opus-mt-en-ru", "Helsinki-NLP/opus-mt-ru-en"),
        "zh": ("Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en"),
    }
    
    def __init__(
        self,
        languages: List[str] = ["de", "fr"],
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """Initialize back-translation augmenter.
        
        Args:
            languages: List of intermediate languages to use (e.g., ["de", "fr"])
            device: Device to run translation models on
            batch_size: Batch size for translation
        """
        self.languages = [l for l in languages if l in self.TRANSLATION_MODELS]
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.batch_size = batch_size
        
        # Lazy load models
        self._models: Dict[str, Tuple] = {}
    
    def _load_translation_pair(self, lang: str):
        """Load translation models for a language pair."""
        if lang in self._models:
            return self._models[lang]
        
        en_to_lang, lang_to_en = self.TRANSLATION_MODELS[lang]
        
        print(f"Loading translation models for {lang}...")
        # Load English -> Language model
        fwd_tokenizer = AutoTokenizer.from_pretrained(en_to_lang)
        fwd_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_lang)
        fwd_model.to(self.device)
        fwd_model.eval()
        
        # Load Language -> English model
        bwd_tokenizer = AutoTokenizer.from_pretrained(lang_to_en)
        bwd_model = AutoModelForSeq2SeqLM.from_pretrained(lang_to_en)
        bwd_model.to(self.device)
        bwd_model.eval()
        
        self._models[lang] = (
            (fwd_tokenizer, fwd_model),
            (bwd_tokenizer, bwd_model)
        )
        return self._models[lang]
    
    def _translate_batch(
        self,
        texts: List[str],
        tokenizer,
        model,
        max_length: int = 512
    ) -> List[str]:
        """Translate a batch of texts."""
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def back_translate(
        self,
        texts: List[str],
        lang: str = "de"
    ) -> List[str]:
        """Back-translate texts through an intermediate language.
        
        Args:
            texts: List of English texts to augment
            lang: Intermediate language code
            
        Returns:
            List of back-translated English texts
        """
        if lang not in self.languages:
            raise ValueError(f"Language {lang} not supported. Use: {self.languages}")
        
        (fwd_tok, fwd_model), (bwd_tok, bwd_model) = self._load_translation_pair(lang)
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # English -> Intermediate language
            intermediate = self._translate_batch(batch, fwd_tok, fwd_model)
            
            # Intermediate language -> English
            back_translated = self._translate_batch(intermediate, bwd_tok, bwd_model)
            
            results.extend(back_translated)
        
        return results
    
    def augment_dataset(
        self,
        sources: List[str],
        targets: List[str],
        augmentation_ratio: float = 0.5,
        seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """Augment a paraphrase dataset using back-translation.
        
        Args:
            sources: List of source texts
            targets: List of target paraphrases
            augmentation_ratio: Fraction of data to augment (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (augmented_sources, augmented_targets)
        """
        random.seed(seed)
        
        num_to_augment = int(len(sources) * augmentation_ratio)
        indices = random.sample(range(len(sources)), num_to_augment)
        
        aug_sources = []
        aug_targets = []
        
        # Distribute augmentation across languages
        chunks = [indices[i::len(self.languages)] for i in range(len(self.languages))]
        
        for lang, chunk_indices in zip(self.languages, chunks):
            if not chunk_indices:
                continue
                
            chunk_sources = [sources[i] for i in chunk_indices]
            chunk_targets = [targets[i] for i in chunk_indices]
            
            print(f"Back-translating {len(chunk_sources)} samples via {lang}...")
            
            # Back-translate sources
            bt_sources = self.back_translate(chunk_sources, lang)
            
            # Back-translate targets
            bt_targets = self.back_translate(chunk_targets, lang)
            
            aug_sources.extend(bt_sources)
            aug_targets.extend(bt_targets)
        
        return aug_sources, aug_targets
    
    def clear_models(self):
        """Clear loaded models to free memory."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

