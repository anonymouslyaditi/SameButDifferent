"""Training pipeline for paraphrase generation models."""

import os
from typing import Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

from ..data.dataset import ParaphraseDataset


@dataclass
class LengthPenaltyConfig:
    """Configuration for length penalty during training."""
    enabled: bool = True
    alpha: float = 0.5  # Weight for length penalty in loss
    target_ratio: float = 0.9  # Target output/input length ratio
    penalty_type: str = "smooth"  # "smooth" or "hard"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./outputs/checkpoints"
    logging_dir: str = "./outputs/logs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    # Length penalty configuration
    length_penalty: LengthPenaltyConfig = field(default_factory=LengthPenaltyConfig)


class ParaphraseTrainer:
    """Trainer for Custom Paraphrase Generator."""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: ParaphraseDataset,
        eval_dataset: Optional[ParaphraseDataset] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        # Setup training arguments
        self.training_args = self._create_training_args()
        
        # Setup trainer
        self.trainer = self._create_trainer()
    
    def _create_training_args(self) -> TrainingArguments:
        """Create training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            eval_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end if self.eval_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=["tensorboard"],
            remove_unused_columns=False
        )
    
    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer."""
        callbacks = []
        if self.eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=callbacks
        )
    
    def train(self) -> Dict:
        """Run training."""
        print(f"Starting training...")
        print(f"  - Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"  - Validation samples: {len(self.eval_dataset)}")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        
        result = self.trainer.train()
        
        # Save final model
        self.save_model(os.path.join(self.config.output_dir, "final_model"))
        
        return result
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def evaluate(self) -> Dict:
        """Evaluate the model."""
        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
        return self.trainer.evaluate()


class LengthAwareTrainer(Trainer):
    """Custom Trainer with length-aware loss for better length preservation.

    Adds a length penalty term to the standard cross-entropy loss to encourage
    the model to generate outputs of similar length to the inputs.
    """

    def __init__(
        self,
        length_penalty_config: LengthPenaltyConfig = None,
        tokenizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.length_penalty_config = length_penalty_config or LengthPenaltyConfig()
        self._tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with optional length penalty."""
        # Get standard model outputs
        outputs = model(**inputs)
        loss = outputs.loss

        if self.length_penalty_config.enabled:
            length_penalty = self._compute_length_penalty(inputs, outputs)
            loss = loss + self.length_penalty_config.alpha * length_penalty

        return (loss, outputs) if return_outputs else loss

    def _compute_length_penalty(self, inputs, outputs) -> torch.Tensor:
        """Compute length penalty based on input/output length ratio."""
        # Get input lengths (non-padding tokens)
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        if input_ids is None or labels is None:
            return torch.tensor(0.0, device=outputs.loss.device)

        # Calculate input lengths
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).float()
        else:
            input_lengths = (input_ids != self._tokenizer.pad_token_id).sum(dim=1).float()

        # Calculate target lengths (labels != -100)
        target_lengths = (labels != -100).sum(dim=1).float()

        # Calculate predicted lengths from logits
        logits = outputs.logits
        pred_tokens = logits.argmax(dim=-1)

        # Count non-padding predicted tokens
        if self._tokenizer is not None:
            pred_lengths = (pred_tokens != self._tokenizer.pad_token_id).sum(dim=1).float()
        else:
            pred_lengths = target_lengths  # Fallback

        # Calculate length ratio
        target_ratio = self.length_penalty_config.target_ratio
        actual_ratio = pred_lengths / (input_lengths + 1e-8)

        # Compute penalty based on type
        if self.length_penalty_config.penalty_type == "smooth":
            # Smooth penalty: penalize deviation from target ratio
            penalty = F.mse_loss(actual_ratio, torch.full_like(actual_ratio, target_ratio))
        else:
            # Hard penalty: penalize if ratio is below target
            deviation = torch.clamp(target_ratio - actual_ratio, min=0)
            penalty = deviation.mean()

        return penalty


class LengthAwareParaphraseTrainer(ParaphraseTrainer):
    """Paraphrase Trainer with length-aware loss."""

    def _create_trainer(self) -> Trainer:
        """Create length-aware HuggingFace Trainer."""
        callbacks = []
        if self.eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        return LengthAwareTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            callbacks=callbacks,
            length_penalty_config=self.config.length_penalty,
            tokenizer=self.tokenizer
        )
