"""Dataset loading and processing for paraphrase generation."""

import torch
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from .preprocessing import TextPreprocessor


def load_paraphrase_datasets(
    dataset_names: List[str] = ["paws", "quora"],
    max_samples: Optional[int] = None,
    seed: int = 42,
    use_augmentation: bool = False,
    augmentation_languages: Optional[List[str]] = None,
    augmentation_ratio: float = 0.5
) -> DatasetDict:
    """Load and combine multiple paraphrase datasets.

    Args:
        dataset_names: List of datasets to load ("paws", "quora")
        max_samples: Maximum samples per dataset (None for all)
        seed: Random seed for reproducibility
        use_augmentation: Whether to apply back-translation augmentation
        augmentation_languages: Languages for back-translation (e.g., ["de", "fr"])
        augmentation_ratio: Fraction of training data to augment

    Returns:
        DatasetDict with train/validation/test splits
    """
    all_data = {"source": [], "target": []}

    for name in dataset_names:
        if name == "paws":
            data = _load_paws(max_samples)
        elif name == "quora":
            data = _load_quora(max_samples)
        else:
            continue

        all_data["source"].extend(data["source"])
        all_data["target"].extend(data["target"])

    # Create dataset and split
    dataset = Dataset.from_dict(all_data)
    dataset = dataset.shuffle(seed=seed)

    # Split into train/val/test
    train_test = dataset.train_test_split(test_size=0.2, seed=seed)
    test_val = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    train_dataset = train_test["train"]

    # Apply back-translation augmentation to training data
    if use_augmentation and augmentation_languages:
        train_dataset = _augment_training_data(
            train_dataset,
            languages=augmentation_languages,
            ratio=augmentation_ratio,
            seed=seed
        )

    return DatasetDict({
        "train": train_dataset,
        "validation": test_val["train"],
        "test": test_val["test"]
    })


def _augment_training_data(
    dataset: Dataset,
    languages: List[str],
    ratio: float = 0.5,
    seed: int = 42
) -> Dataset:
    """Apply back-translation augmentation to training data."""
    from .augmentation import BackTranslationAugmenter

    print(f"\n[Augmentation] Applying back-translation via {languages}...")

    augmenter = BackTranslationAugmenter(languages=languages)

    sources = dataset["source"]
    targets = dataset["target"]

    aug_sources, aug_targets = augmenter.augment_dataset(
        sources=sources,
        targets=targets,
        augmentation_ratio=ratio,
        seed=seed
    )

    # Clear models to free memory
    augmenter.clear_models()

    # Combine original + augmented data
    combined_sources = list(sources) + aug_sources
    combined_targets = list(targets) + aug_targets

    print(f"[Augmentation] Original: {len(sources)}, Augmented: {len(aug_sources)}, Total: {len(combined_sources)}")

    return Dataset.from_dict({
        "source": combined_sources,
        "target": combined_targets
    }).shuffle(seed=seed)


def _load_paws(max_samples: Optional[int] = None) -> Dict[str, List[str]]:
    """Load PAWS dataset."""
    try:
        dataset = load_dataset("paws", "labeled_final", split="train", trust_remote_code=True)
        # Filter for paraphrase pairs (label=1)
        dataset = dataset.filter(lambda x: x["label"] == 1)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        return {
            "source": dataset["sentence1"],
            "target": dataset["sentence2"]
        }
    except Exception as e:
        print(f"Error loading PAWS: {e}")
        return {"source": [], "target": []}


def _load_quora(max_samples: Optional[int] = None) -> Dict[str, List[str]]:
    """Load Quora Question Pairs dataset."""
    try:
        dataset = load_dataset("quora", split="train", trust_remote_code=True)
        # Filter for duplicate pairs (paraphrases)
        dataset = dataset.filter(lambda x: x["is_duplicate"] == True)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        sources = [q["text"][0] for q in dataset["questions"]]
        targets = [q["text"][1] for q in dataset["questions"]]
        
        return {"source": sources, "target": targets}
    except Exception as e:
        print(f"Error loading Quora: {e}")
        return {"source": [], "target": []}


class ParaphraseDataset(TorchDataset):
    """PyTorch Dataset for paraphrase generation."""
    
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 512,
        prefix: str = "paraphrase: "
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.prefix = prefix
        self.preprocessor = TextPreprocessor()
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        source = self.preprocessor.prepare_for_model(item["source"], self.prefix)
        target = item["target"]
        
        # Tokenize input
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels
        }

