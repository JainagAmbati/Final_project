"""
Residual Debiasing - Simpler Approach
Train model on examples where bias model is uncertain or wrong.
"""

import sys
sys.path.insert(0, '..')

import datasets
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    HfArgumentParser
)
from helpers import compute_accuracy, prepare_dataset_nli
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def get_bias_predictions(dataset, bias_model, tokenizer, device, batch_size=32):
    """Get predictions from bias model on all examples."""
    print("Computing bias model predictions...")
    
    bias_model.eval()
    predictions = []
    confidences = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # Tokenize hypotheses only
        inputs = tokenizer(
            batch['hypothesis'],
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bias_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            pred_labels = torch.argmax(logits, dim=-1).cpu().numpy()
            max_probs = torch.max(probs, dim=-1)[0].cpu().numpy()
            
            predictions.extend(pred_labels.tolist())
            confidences.extend(max_probs.tolist())
    
    return predictions, confidences


def filter_dataset_by_bias_performance(dataset, bias_predictions, bias_confidences, strategy='hard_only'):
    """
    Filter dataset based on bias model performance.
    
    Strategies:
    - 'hard_only': Keep only examples where bias model is wrong or uncertain
    - 'reweight': Keep all but weight by difficulty
    - 'balanced': Balance hard and easy examples
    """
    print(f"\nFiltering dataset using strategy: {strategy}")
    
    indices_to_keep = []
    weights = []
    
    for idx, (example, bias_pred, bias_conf) in enumerate(zip(dataset, bias_predictions, bias_confidences)):
        true_label = example['label']
        is_correct = (bias_pred == true_label)
        
        if strategy == 'hard_only':
            # Keep only examples where bias is wrong OR uncertain (confidence < 0.6)
            if not is_correct or bias_conf < 0.6:
                indices_to_keep.append(idx)
                weights.append(1.0)
        
        elif strategy == 'reweight':
            # Keep all, but upweight hard examples
            indices_to_keep.append(idx)
            if not is_correct:
                weights.append(3.0)  # 3x weight for wrong predictions
            elif bias_conf < 0.6:
                weights.append(2.0)  # 2x weight for uncertain
            else:
                weights.append(1.0)
        
        elif strategy == 'balanced':
            # Keep hard examples + sample of easy examples
            if not is_correct or bias_conf < 0.6:
                indices_to_keep.append(idx)
                weights.append(1.0)
            elif np.random.random() < 0.3:  # Keep 30% of easy examples
                indices_to_keep.append(idx)
                weights.append(0.5)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(indices_to_keep)}")
    print(f"Kept {len(indices_to_keep)/len(dataset)*100:.1f}% of examples")
    
    filtered_dataset = dataset.select(indices_to_keep)
    
    return filtered_dataset, weights


class WeightedTrainer(Trainer):
    """Trainer that supports example weighting."""
    
    def __init__(self, *args, example_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.example_weights = example_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
        
        # Apply weights if available
        if self.example_weights is not None:
            batch_start = self.state.global_step * self.args.per_device_train_batch_size
            batch_end = batch_start + len(labels)
            # Make sure we don't go out of bounds
            if batch_end > len(self.example_weights):
                batch_end = len(self.example_weights)
            if batch_start < len(self.example_weights):
                batch_weights = torch.tensor(
                    self.example_weights[batch_start:batch_end],
                    device=loss.device,
                    dtype=loss.dtype
                )
                # Ensure weights match batch size
                if len(batch_weights) == len(loss):
                    loss = loss * batch_weights
        
        loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--main_model', type=str,
                      default='google/electra-small-discriminator',
                      help='Main model to train')
    argp.add_argument('--bias_model', type=str, required=True,
                      help='Path to hypothesis-only bias model')
    argp.add_argument('--dataset', type=str, default='snli',
                      help='Dataset to use')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit training examples')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    argp.add_argument('--filter_strategy', type=str, 
                      choices=['hard_only', 'reweight', 'balanced'],
                      default='reweight',
                      help='How to handle bias model predictions')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 70)
    print("RESIDUAL DEBIASING - FOCUS ON HARD EXAMPLES")
    print("=" * 70)
    print(f"Bias model: {args.bias_model}")
    print(f"Filter strategy: {args.filter_strategy}")
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
    
    if args.dataset == 'snli':
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.main_model, use_fast=True)

    # Load bias model
    print(f"\nLoading bias model from: {args.bias_model}")
    bias_model = AutoModelForSequenceClassification.from_pretrained(args.bias_model, num_labels=3)
    bias_model.to(device)
    bias_model.eval()
    
    if hasattr(bias_model, 'electra'):
        for param in bias_model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    print("✓ Bias model loaded")

    # Get training data
    train_dataset = dataset['train']
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    # Get bias predictions on training data
    bias_preds, bias_confs = get_bias_predictions(
        train_dataset, bias_model, tokenizer, device
    )
    
    # Analyze bias model performance
    bias_correct = sum(1 for pred, ex in zip(bias_preds, train_dataset) if pred == ex['label'])
    bias_acc = bias_correct / len(train_dataset)
    print(f"\nBias model accuracy on training data: {bias_acc:.2%}")
    
    # Filter/reweight dataset
    filtered_train_dataset, example_weights = filter_dataset_by_bias_performance(
        train_dataset, bias_preds, bias_confs, args.filter_strategy
    )
    
    # Prepare datasets
    print("\nTokenizing datasets...")
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    
    train_dataset_featurized = filtered_train_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=filtered_train_dataset.column_names
    )
    
    # Prepare eval dataset (use full dataset, not filtered)
    eval_dataset = dataset[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )
    
    print("✓ Datasets prepared")

    # Load main model
    print(f"\nInitializing main model: {args.main_model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.main_model, num_labels=3)
    
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    print("✓ Model loaded")

    # Store predictions
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_accuracy(eval_preds)

    # Initialize trainer
    trainer_class = WeightedTrainer if args.filter_strategy == 'reweight' else Trainer
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset_featurized,
        'eval_dataset': eval_dataset_featurized,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics_and_store_predictions
    }
    
    if args.filter_strategy == 'reweight':
        trainer_kwargs['example_weights'] = example_weights
    
    trainer = trainer_class(**trainer_kwargs)

    # Train
    if training_args.do_train:
        print("\n" + "="*70)
        print("TRAINING DEBIASED MODEL ON HARD EXAMPLES")
        print("="*70)
        trainer.train()
        trainer.save_model()
        print("\n✓ Training complete!")

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

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save predictions
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), 'w') as f:
            for i, example in enumerate(eval_dataset):
                example_dict = {
                    'premise': example['premise'],
                    'hypothesis': example['hypothesis'],
                    'label': example['label'],
                    'predicted_scores': eval_predictions.predictions[i].tolist(),
                    'predicted_label': int(eval_predictions.predictions[i].argmax())
                }
                f.write(json.dumps(example_dict))
                f.write('\n')

        print(f"\n✓ Results saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()