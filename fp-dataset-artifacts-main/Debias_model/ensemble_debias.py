"""
Ensemble Debiasing for NLI
Based on He et al. 2019: "Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual"

Approach:
1. Use hypothesis-only model as the "bias model" (already trained)
2. Train a "debiased model" that learns to correct the bias model's mistakes
3. At inference: combine both models' predictions
"""

import sys
sys.path.insert(0, '..')  # Add parent directory

import datasets
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    HfArgumentParser
)
from transformers.modeling_outputs import SequenceClassifierOutput
# Import from parent directory
from helpers import compute_accuracy
import os
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

NUM_PREPROCESSING_WORKERS = 2


def prepare_dataset_nli_with_bias_scores(examples, tokenizer, bias_model, device, max_seq_length=None):
    """
    Prepare dataset with bias model predictions.
    The main model will learn to correct these bias predictions.
    """
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    # Tokenize premise + hypothesis for main model
    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    # Get bias model predictions (hypothesis-only)
    hypothesis_inputs = tokenizer(
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Move to device and get bias predictions
    hypothesis_inputs = {k: v.to(device) for k, v in hypothesis_inputs.items()}
    
    with torch.no_grad():
        bias_outputs = bias_model(**hypothesis_inputs)
        bias_logits = bias_outputs.logits
        bias_probs = torch.nn.functional.softmax(bias_logits, dim=-1)
    
    # Store bias predictions
    tokenized_examples['bias_probs'] = bias_probs.cpu().numpy().tolist()
    tokenized_examples['label'] = examples['label']
    
    return tokenized_examples


class DebiasedNLIModel(nn.Module):
    """
    Model that combines a bias model (hypothesis-only) with a main model.
    The main model learns to predict the residual/correction.
    """
    def __init__(self, main_model, bias_model=None, ensemble_weight=0.5):
        super().__init__()
        self.main_model = main_model
        self.bias_model = bias_model
        self.ensemble_weight = ensemble_weight  # Weight for bias model (0.5 = equal weight)
        
        if bias_model is not None:
            # Freeze bias model parameters
            for param in self.bias_model.parameters():
                param.requires_grad = False
            self.bias_model.eval()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, bias_probs=None):
        """
        Forward pass that combines bias and main model.
        
        If bias_probs are provided (from preprocessing), use those.
        Otherwise, compute bias predictions on the fly.
        """
        # Get main model predictions
        main_outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        main_logits = main_outputs.logits
        
        # Get or use bias predictions
        if bias_probs is not None:
            # Use precomputed bias probabilities
            if isinstance(bias_probs, list):
                bias_probs = torch.tensor(bias_probs).to(main_logits.device)
            elif not isinstance(bias_probs, torch.Tensor):
                bias_probs = torch.tensor(bias_probs).to(main_logits.device)
            bias_probs = bias_probs.to(main_logits.device)
        elif self.bias_model is not None:
            # Compute bias predictions (hypothesis-only)
            # Extract hypothesis tokens only (after [SEP] token)
            with torch.no_grad():
                bias_outputs = self.bias_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                bias_logits = bias_outputs.logits
                bias_probs = torch.nn.functional.softmax(bias_logits, dim=-1)
        else:
            # No bias model, just use uniform distribution
            bias_probs = torch.ones_like(main_logits) / main_logits.shape[-1]
        
        # Convert main logits to probabilities
        main_probs = torch.nn.functional.softmax(main_logits, dim=-1)
        
        # Ensemble: weighted combination
        combined_probs = (
            self.ensemble_weight * bias_probs + 
            (1 - self.ensemble_weight) * main_probs
        )
        
        # Convert back to logits for loss computation
        combined_logits = torch.log(combined_probs + 1e-10)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(combined_logits, labels)
        
        # Return in the format expected by Trainer (as a tuple or dict-like object)
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=combined_logits,
            hidden_states=None,
            attentions=None
        )


class DebiasedTrainer(Trainer):
    """Custom trainer that handles bias_probs in the dataset."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to pass bias_probs to the model."""
        labels = inputs.pop("labels")
        bias_probs = inputs.pop("bias_probs", None)
        
        outputs = model(**inputs, labels=labels, bias_probs=bias_probs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--main_model', type=str,
                      default='google/electra-small-discriminator',
                      help='Main model to train (will learn debiased features)')
    argp.add_argument('--bias_model', type=str, required=True,
                      help='Path to trained hypothesis-only model (bias model)')
    argp.add_argument('--dataset', type=str, default='snli',
                      help='Dataset to use')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit training examples')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    argp.add_argument('--ensemble_weight', type=float, default=0.3,
                      help='Weight for bias model in ensemble (0-1). Lower = less bias influence')
    argp.add_argument('--precompute_bias', action='store_true',
                      help='Precompute bias predictions (faster but uses more disk)')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 70)
    print("ENSEMBLE DEBIASING FOR NLI")
    print("=" * 70)
    print(f"Bias model (hypothesis-only): {args.bias_model}")
    print(f"Main model: {args.main_model}")
    print(f"Ensemble weight for bias model: {args.ensemble_weight}")
    print(f"(Lower weight = less bias influence)")
    print("=" * 70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load dataset
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        dataset_id = tuple(args.dataset.split(':')) if ':' in args.dataset else (args.dataset,)
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = datasets.load_dataset(*dataset_id)
    
    # Remove SNLI examples with no label
    if args.dataset == 'snli':
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.main_model, use_fast=True)

    # Load bias model (hypothesis-only)
    print(f"\nLoading bias model from: {args.bias_model}")
    bias_model = AutoModelForSequenceClassification.from_pretrained(args.bias_model, num_labels=3)
    bias_model.to(device)
    bias_model.eval()
    
    # Fix tensor contiguity for ELECTRA
    if hasattr(bias_model, 'electra'):
        for param in bias_model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    print("✓ Bias model loaded successfully")

    # Prepare datasets
    print("\nPreparing datasets with bias predictions...")
    
    if args.precompute_bias:
        print("(Precomputing bias predictions - this may take a few minutes)")
        # Precompute bias predictions and add to dataset
        def add_bias_predictions(examples):
            return prepare_dataset_nli_with_bias_scores(
                examples, tokenizer, bias_model, device, args.max_length
            )
        
        train_dataset = None
        eval_dataset = None
        
        if training_args.do_train:
            train_dataset = dataset['train']
            if args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
            train_dataset = train_dataset.map(
                add_bias_predictions,
                batched=True,
                batch_size=32,
                remove_columns=dataset['train'].column_names
            )
        
        if training_args.do_eval:
            eval_dataset = dataset[eval_split]
            if args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            eval_dataset = eval_dataset.map(
                add_bias_predictions,
                batched=True,
                batch_size=32,
                remove_columns=dataset[eval_split].column_names
            )
    else:
        # Use standard preprocessing without precomputed bias
        from helpers import prepare_dataset_nli
        
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        
        train_dataset = None
        eval_dataset = None
        
        if training_args.do_train:
            train_dataset = dataset['train']
            if args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
            train_dataset = train_dataset.map(
                prepare_train_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=dataset['train'].column_names
            )
        
        if training_args.do_eval:
            eval_dataset = dataset[eval_split]
            if args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            eval_dataset = eval_dataset.map(
                prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=dataset[eval_split].column_names
            )

    print("✓ Datasets prepared")

    # Load main model
    print(f"\nInitializing main model: {args.main_model}")
    main_model = AutoModelForSequenceClassification.from_pretrained(args.main_model, num_labels=3)
    
    # Fix tensor contiguity for ELECTRA
    if hasattr(main_model, 'electra'):
        for param in main_model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    # Create debiased model
    debiased_model = DebiasedNLIModel(
        main_model=main_model,
        bias_model=bias_model if not args.precompute_bias else None,
        ensemble_weight=args.ensemble_weight
    )

    print("✓ Debiased model created")
    print(f"  - Ensemble weight: {args.ensemble_weight}")
    print(f"  - Bias model frozen: Yes")

    # Store predictions for evaluation
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_accuracy(eval_preds)

    # Initialize trainer
    trainer_class = DebiasedTrainer if args.precompute_bias else Trainer
    
    trainer = trainer_class(
        model=debiased_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    # Train
    if training_args.do_train:
        print("\n" + "="*70)
        print("TRAINING DEBIASED MODEL")
        print("="*70)
        trainer.train()
        trainer.save_model()
        print("\n✓ Training complete!")
        print(f"✓ Model saved to: {training_args.output_dir}")

    # Evaluate
    if training_args.do_eval:
        print("\n" + "="*70)
        print("EVALUATING DEBIASED MODEL")
        print("="*70)
        results = trainer.evaluate()

        print('\nEvaluation Results:')
        print('─' * 70)
        for key, value in results.items():
            print(f"{key:30s}: {value}")
        print('─' * 70)

        # Save results
        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 
                  encoding='utf-8', mode='w') as f:
            json.dump(results, f, indent=2)

        # Save predictions
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), 
                  encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset):
                # Reconstruct original example
                example_dict = {
                    'premise': dataset[eval_split][i]['premise'],
                    'hypothesis': dataset[eval_split][i]['hypothesis'],
                    'label': dataset[eval_split][i]['label'],
                    'predicted_scores': eval_predictions.predictions[i].tolist(),
                    'predicted_label': int(eval_predictions.predictions[i].argmax())
                }
                f.write(json.dumps(example_dict))
                f.write('\n')

        print(f"\n✓ Results saved to: {training_args.output_dir}")
        print(f"✓ Predictions saved to: {training_args.output_dir}/eval_predictions.jsonl")


if __name__ == "__main__":
    main()