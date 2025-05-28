from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, IntervalStrategy
from datasets import load_dataset
import torch
import numpy as np
from typing import Dict, List
import evaluate
import nltk
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentTrainer:
    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        output_dir: str = "fine_tuned_model",
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Log device information
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Initialize metric
        self.rouge = evaluate.load("rouge")
        
    def load_legal_dataset(self, dataset_name: str = "billsum", max_samples: int = 1000):
        """
        Load and prepare the billsum dataset with a limited number of samples.
        """
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            
            # Take only max_samples from train and a proportional amount from test
            train_size = min(len(dataset["train"]), max_samples)
            test_size = min(len(dataset["test"]), max_samples // 5)
            
            dataset["train"] = dataset["train"].select(range(train_size))
            dataset["test"] = dataset["test"].select(range(test_size))
            
            logger.info(f"Using {train_size} training samples and {test_size} test samples")
            
            # Log some sample lengths before filtering
            for split in ["train", "test"]:
                sample_idx = 0
                text_len = len(dataset[split][sample_idx]["text"])
                summary_len = len(dataset[split][sample_idx]["summary"])
                logger.info(f"Sample {split} text length: {text_len}")
                logger.info(f"Sample {split} summary length: {summary_len}")
            
            # Filter out examples where summary is too long or text is empty
            def filter_examples(example):
                text_len = len(example["text"])
                summary_len = len(example["summary"])
                
                # More lenient filtering criteria
                is_valid = (
                    text_len >= 100 and  # Minimum text length
                    summary_len >= 10 and  # Minimum summary length
                    text_len <= 15000 and  # Increased maximum length
                    summary_len <= 2000     # Increased maximum length
                )
                
                return is_valid
            
            filtered_dataset = dataset.filter(
                filter_examples,
                desc="Filtering dataset"
            )
            
            filtered_train_size = len(filtered_dataset["train"])
            filtered_test_size = len(filtered_dataset["test"])
            
            logger.info(f"Dataset size after filtering: train={filtered_train_size}, test={filtered_test_size}")
            
            if filtered_train_size == 0 or filtered_test_size == 0:
                raise ValueError("All examples were filtered out. Please check the filtering criteria.")
            
            return filtered_dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_data(self, examples: Dict) -> Dict:
        """Preprocess the data for training"""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Ensure we don't get tensors yet
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():  # Properly handle target tokenization
            labels = self.tokenizer(
                examples["summary"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,  # Ensure we don't get tensors yet
            )

        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding token id with -100 for loss calculation
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in model_inputs["labels"]
        ]
        
        # Verify we have valid labels (debug)
        if any(all(l == -100 for l in label) for label in model_inputs["labels"]):
            logger.warning("Found sequence with all -100 labels!")
            
        return model_inputs

    def compute_metrics(self, eval_pred) -> Dict:
        """Compute ROUGE metrics"""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Log a sample prediction
        if len(decoded_preds) > 0:
            logger.info("\nSample Prediction:")
            logger.info(f"Predicted: {decoded_preds[0][:200]}...")
            logger.info(f"Reference: {decoded_labels[0][:200]}...")
        
        return {k: round(v * 100, 2) for k, v in result.items()}

    def train(
        self,
        train_dataset,
        eval_dataset,
        batch_size: int = 8,
        num_epochs: int = 20,
        learning_rate: float = 3e-5
    ):
        """Train the model"""
        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_total_limit=2,
            save_steps=50,
            eval_steps=50,
            gradient_accumulation_steps=4,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            generation_num_beams=2,
            fp16=False,
            remove_unused_columns=False,
            label_names=["labels"],
            include_inputs_for_metrics=True,
            dataloader_num_workers=0,  # Set to 0 to debug
            gradient_checkpointing=False,  # Disable for now
            optim="adamw_torch",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            lr_scheduler_type="linear"  # Correct parameter name for learning rate schedule
        )

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100  # Explicitly set label padding
        )

        # Reset model weights to ensure proper initialization
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Ensure model is in training mode
        self.model.train()
        
        # Zero gradients
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = None

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(self.output_dir / "final_model")
        self.tokenizer.save_pretrained(self.output_dir / "final_model")
        logger.info(f"Model saved to {self.output_dir / 'final_model'}")

def main():
    # Initialize trainer
    trainer = LegalDocumentTrainer()
    
    # Load dataset
    dataset = trainer.load_legal_dataset()
    
    # Split dataset
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # Preprocess datasets
    train_dataset = train_dataset.map(
        trainer.preprocess_data,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing training data"
    )
    
    eval_dataset = eval_dataset.map(
        trainer.preprocess_data,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing validation data"
    )
    
    # Train the model
    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main() 