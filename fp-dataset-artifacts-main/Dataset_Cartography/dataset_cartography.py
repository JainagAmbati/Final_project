"""
Dataset Cartography for NLI
Based on Swayamdipta et al. 2020: "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"

Maps examples based on training dynamics:
- Confidence: Model's average confidence on correct label across epochs
- Variability: Standard deviation of confidence
- Correctness: Fraction of epochs where predicted correctly

This identifies:
- Easy-to-learn examples (high confidence, low variability)
- Hard-to-learn examples (low confidence, high variability)  
- Ambiguous examples (medium confidence, high variability)
"""

import sys
sys.path.insert(0, '..')

import datasets
import torch
import torch.nn as nn
import numpy as np
import json
import os
from collections import defaultdict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    HfArgumentParser,
    TrainerCallback
)
from helpers import compute_accuracy, prepare_dataset_nli
import matplotlib.pyplot as plt
import seaborn as sns

NUM_PREPROCESSING_WORKERS = 2


class DataMapCallback(TrainerCallback):
    """
    Callback to record training dynamics for dataset cartography.
    Records confidence and correctness for each example at each epoch.
    """
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.training_dynamics = defaultdict(lambda: {
            'confidences': [],
            'correctness': []
        })
        
    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Record predictions at the end of each epoch."""
        print(f"\n  Recording training dynamics for epoch {state.epoch}...")
        
        # Get predictions on training data
        model.eval()
        device = next(model.parameters()).device
        
        # Process in batches
        batch_size = 32
        for idx in range(0, len(self.train_dataset), batch_size):
            batch_end = min(idx + batch_size, len(self.train_dataset))
            batch = self.train_dataset[idx:batch_end]
            
            # Prepare inputs
            inputs = {
                'input_ids': torch.tensor(batch['input_ids']).to(device),
                'attention_mask': torch.tensor(batch['attention_mask']).to(device),
            }
            if 'token_type_ids' in batch:
                inputs['token_type_ids'] = torch.tensor(batch['token_type_ids']).to(device)
            
            labels = torch.tensor(batch['label']).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Record for each example
            for i in range(len(labels)):
                example_idx = idx + i
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                
                # Confidence = probability of true label
                confidence = probs[i, true_label].item()
                correct = (pred_label == true_label)
                
                self.training_dynamics[example_idx]['confidences'].append(confidence)
                self.training_dynamics[example_idx]['correctness'].append(correct)
        
        model.train()
        print(f"  ✓ Recorded dynamics for {len(self.training_dynamics)} examples")
    
    def get_data_map_statistics(self):
        """
        Calculate dataset cartography statistics.
        
        Returns:
            dict: Statistics for each example with keys:
                - confidence: mean confidence across epochs
                - variability: std of confidence across epochs
                - correctness: fraction of epochs predicted correctly
        """
        statistics = {}
        
        for idx, dynamics in self.training_dynamics.items():
            confidences = dynamics['confidences']
            correctness = dynamics['correctness']
            
            if len(confidences) > 0:
                statistics[idx] = {
                    'confidence': np.mean(confidences),
                    'variability': np.std(confidences),
                    'correctness': np.mean(correctness)
                }
            else:
                # If no dynamics recorded, mark as easy (default)
                statistics[idx] = {
                    'confidence': 1.0,
                    'variability': 0.0,
                    'correctness': 1.0
                }
        
        return statistics


def categorize_examples(statistics):
    """
    Categorize examples into easy, hard, and ambiguous based on Swayamdipta et al. 2020.
    
    Heuristics:
    - Easy-to-learn: High confidence (>0.7), Low variability (<0.15), High correctness (>0.8)
    - Hard-to-learn: Low confidence (<0.5), High variability (>0.15), Low correctness (<0.6)
    - Ambiguous: Medium confidence, High variability
    """
    categories = {
        'easy': [],
        'hard': [],
        'ambiguous': []
    }
    
    for idx, stats in statistics.items():
        conf = stats['confidence']
        var = stats['variability']
        corr = stats['correctness']
        
        if conf > 0.7 and var < 0.15 and corr > 0.8:
            categories['easy'].append(idx)
        elif conf < 0.5 or corr < 0.6:
            categories['hard'].append(idx)
        elif var > 0.15:
            categories['ambiguous'].append(idx)
        else:
            # Default to easy if doesn't fit other categories
            categories['easy'].append(idx)
    
    return categories


def visualize_data_map(statistics, output_dir):
    """Create data map visualization (confidence vs variability)."""
    print("\nCreating data map visualization...")
    
    confidences = [stats['confidence'] for stats in statistics.values()]
    variabilities = [stats['variability'] for stats in statistics.values()]
    correctness = [stats['correctness'] for stats in statistics.values()]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        variabilities, 
        confidences, 
        c=correctness,
        cmap='RdYlGn',
        alpha=0.5,
        s=10
    )
    plt.colorbar(scatter, label='Correctness')
    plt.xlabel('Variability (std of confidence)')
    plt.ylabel('Confidence (mean)')
    plt.title('Dataset Cartography Map')
    
    # Add region labels
    plt.text(0.02, 0.85, 'Easy-to-learn', fontsize=12, weight='bold')
    plt.text(0.25, 0.3, 'Hard-to-learn', fontsize=12, weight='bold')
    plt.text(0.25, 0.6, 'Ambiguous', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_map.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/data_map.png")
    plt.close()
    
    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(confidences, bins=50, edgecolor='black')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    
    axes[1].hist(variabilities, bins=50, edgecolor='black')
    axes[1].set_xlabel('Variability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Variability Distribution')
    
    axes[2].hist(correctness, bins=50, edgecolor='black')
    axes[2].set_xlabel('Correctness')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Correctness Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/distributions.png")
    plt.close()


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help='Model to train')
    argp.add_argument('--dataset', type=str, default='snli',
                      help='Dataset to use')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit training examples (for faster mapping)')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    argp.add_argument('--cartography_only', action='store_true',
                      help='Only create data map, do not retrain on filtered data')
    argp.add_argument('--load_cartography', type=str, default=None,
                      help='Path to existing cartography directory to skip Phase 1')
    argp.add_argument('--training_strategy', type=str,
                      choices=['hard_only', 'remove_easy', 'reweight', 'balanced'],
                      default='reweight',
                      help='Training strategy after cartography')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 70)
    print("DATASET CARTOGRAPHY FOR NLI")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Cartography only: {args.cartography_only}")
    if args.load_cartography:
        print(f"Loading existing cartography from: {args.load_cartography}")
    if not args.cartography_only:
        print(f"Training strategy: {args.training_strategy}")
    print("=" * 70)

    # Load dataset
    if args.dataset == 'snli':
        dataset = datasets.load_dataset('snli')
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
        eval_split = 'validation'
    else:
        print(f"Error: Only SNLI supported")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare datasets
    train_dataset = dataset['train']
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    eval_dataset = dataset[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    # Tokenize
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    
    train_dataset_featurized = train_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    # Check if we should load existing cartography or create new
    if args.load_cartography:
        print("\n" + "="*70)
        print("LOADING EXISTING CARTOGRAPHY")
        print("="*70)
        
        cartography_dir = args.load_cartography
        categories_path = os.path.join(cartography_dir, 'categories.json')
        statistics_path = os.path.join(cartography_dir, 'statistics.json')
        
        if not os.path.exists(categories_path):
            print(f"Error: {categories_path} not found!")
            print("Please run with --cartography_only first to create the data map.")
            return
        
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        
        with open(statistics_path, 'r') as f:
            statistics_raw = json.load(f)
            statistics = {int(k): v for k, v in statistics_raw.items()}
        
        print(f"✓ Loaded cartography from: {cartography_dir}")
        print(f"\nDataset Categorization:")
        print(f"  Easy-to-learn:  {len(categories['easy']):6d} ({len(categories['easy'])/len(statistics)*100:.1f}%)")
        print(f"  Hard-to-learn:  {len(categories['hard']):6d} ({len(categories['hard'])/len(statistics)*100:.1f}%)")
        print(f"  Ambiguous:      {len(categories['ambiguous']):6d} ({len(categories['ambiguous'])/len(statistics)*100:.1f}%)")
        
    else:
        # Phase 1: Create cartography (original code)
        # Load model
        print(f"\nInitializing model: {args.model}")
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        
        if hasattr(model, 'electra'):
            for param in model.electra.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

        # Create data map callback
        datamap_callback = DataMapCallback(train_dataset_featurized)

        # Store predictions for evaluation
        eval_predictions = None
        def compute_metrics_and_store_predictions(eval_preds):
            nonlocal eval_predictions
            eval_predictions = eval_preds
            return compute_accuracy(eval_preds)

        # Phase 1: Train to create data map
        print("\n" + "="*70)
        print("PHASE 1: TRAINING TO CREATE DATA MAP")
        print("="*70)
        print("Training model to record training dynamics...")
        
        # Use separate output dir for cartography
        cartography_output = os.path.join(training_args.output_dir, 'cartography')
        os.makedirs(cartography_output, exist_ok=True)
        
        cartography_args = TrainingArguments(
            output_dir=cartography_output,
            num_train_epochs=training_args.num_train_epochs if training_args.num_train_epochs else 3,
            per_device_train_batch_size=training_args.per_device_train_batch_size if training_args.per_device_train_batch_size else 32,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size if training_args.per_device_eval_batch_size else 32,
            logging_steps=training_args.logging_steps if training_args.logging_steps else 500,
            save_strategy='no',  # Don't save during cartography phase
            fp16=training_args.fp16 if hasattr(training_args, 'fp16') else False,
        )

        trainer = Trainer(
            model=model,
            args=cartography_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            callbacks=[datamap_callback],
            compute_metrics=compute_metrics_and_store_predictions
        )

        trainer.train()
        
        # Get data map statistics
        print("\n" + "="*70)
        print("ANALYZING TRAINING DYNAMICS")
        print("="*70)
        
        statistics = datamap_callback.get_data_map_statistics()
        categories = categorize_examples(statistics)
        
        print(f"\nDataset Categorization:")
        print(f"  Easy-to-learn:  {len(categories['easy']):6d} ({len(categories['easy'])/len(statistics)*100:.1f}%)")
        print(f"  Hard-to-learn:  {len(categories['hard']):6d} ({len(categories['hard'])/len(statistics)*100:.1f}%)")
        print(f"  Ambiguous:      {len(categories['ambiguous']):6d} ({len(categories['ambiguous'])/len(statistics)*100:.1f}%)")
        
        # Save statistics and categories
        with open(os.path.join(cartography_output, 'statistics.json'), 'w') as f:
            json.dump({idx: stats for idx, stats in statistics.items()}, f, indent=2)
        
        with open(os.path.join(cartography_output, 'categories.json'), 'w') as f:
            json.dump(categories, f, indent=2)
        
        print(f"\n✓ Saved statistics to {cartography_output}/statistics.json")
        print(f"✓ Saved categories to {cartography_output}/categories.json")
        
        # Visualize
        visualize_data_map(statistics, cartography_output)
        
        # If cartography_only, stop here
        if args.cartography_only:
            print("\n" + "="*70)
            print("CARTOGRAPHY COMPLETE")
            print("="*70)
            print(f"Results saved to: {cartography_output}")
            print("\nTo train with filtered data, run:")
            print(f"  python dataset_cartography.py --do_train --do_eval \\")
            print(f"    --load_cartography {cartography_output} \\")
            print(f"    --training_strategy reweight \\")
            print(f"    --output_dir ./cartography_reweight \\")
            print(f"    [other training arguments...]")
            return
    
    # Phase 2: Retrain on filtered/reweighted data
    print("\n" + "="*70)
    print("PHASE 2: RETRAINING WITH FILTERED DATA")
    print("="*70)
    print(f"Strategy: {args.training_strategy}")
    
    # Store predictions for evaluation
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_accuracy(eval_preds)
    
    # Apply training strategy
    if args.training_strategy == 'hard_only':
        # Train only on hard examples
        indices = categories['hard']
        print(f"Training on {len(indices)} hard examples only")
        
    elif args.training_strategy == 'remove_easy':
        # Remove easy examples, keep hard + ambiguous
        indices = categories['hard'] + categories['ambiguous']
        print(f"Training on {len(indices)} hard + ambiguous examples")
        
    elif args.training_strategy == 'balanced':
        # Balance: all hard, all ambiguous, sample of easy
        easy_sample = np.random.choice(
            categories['easy'],
            size=min(len(categories['easy']), len(categories['hard'])),
            replace=False
        ).tolist()
        indices = categories['hard'] + categories['ambiguous'] + easy_sample
        print(f"Training on {len(indices)} balanced examples")
        
    else:  # reweight
        # Keep all examples but reweight
        indices = list(range(len(train_dataset_featurized)))
        print(f"Training on all {len(indices)} examples with reweighting")
    
    # Create filtered dataset
    filtered_train = train_dataset_featurized.select(indices)
    
    # Initialize new model
    print(f"\nInitializing fresh model...")
    model_retrain = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    
    if hasattr(model_retrain, 'electra'):
        for param in model_retrain.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    # Train
    trainer_retrain = Trainer(
        model=model_retrain,
        args=training_args,
        train_dataset=filtered_train,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    
    trainer_retrain.train()
    trainer_retrain.save_model()
    
    # Evaluate
    if training_args.do_eval:
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        results = trainer_retrain.evaluate()
        
        print('\nEvaluation Results:')
        print('─' * 70)
        for key, value in results.items():
            print(f"{key:30s}: {value}")
        print('─' * 70)
        
        # Save results
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