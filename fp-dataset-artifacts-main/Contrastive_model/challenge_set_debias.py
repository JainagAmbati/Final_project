"""
Challenge Set Training for NLI Debiasing

Uses existing high-quality adversarial and contrastive datasets:
1. HANS (Heuristic Analysis for NLI Systems) - syntactic challenges
2. ANLI (Adversarial NLI) - adversarial examples
3. Stress Tests - targeted linguistic phenomena

These are professionally constructed, not synthetic, so should be higher quality.
"""

import sys
sys.path.insert(0, '..')

import datasets
import json
import torch
import pandas as pd
import requests
from io import StringIO
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    HfArgumentParser
)
from helpers import compute_accuracy, prepare_dataset_nli
import os

NUM_PREPROCESSING_WORKERS = 2


def load_hans_dataset():
    """
    Load HANS dataset (Heuristic Analysis for NLI Systems).
    Designed to test lexical overlap, subsequence, and constituent heuristics.
    """
    print("Loading HANS dataset...")
    try:
        # Try loading from HuggingFace datasets (newer method)
        hans = datasets.load_dataset("hans", trust_remote_code=True)
        print(f"✓ HANS loaded: {len(hans['train'])} train, {len(hans['validation'])} validation")
        return hans
    except Exception as e1:
        print(f"  First method failed: {e1}")
        try:
            # Try alternative: load from direct URL
            print("  Trying to load HANS from alternative source...")
            hans_train_url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.txt"
            hans_val_url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
            
            import pandas as pd
            import requests
            from io import StringIO
            
            # Download and parse
            train_response = requests.get(hans_train_url)
            train_data = pd.read_csv(StringIO(train_response.text), sep='\t')
            
            val_response = requests.get(hans_val_url)
            val_data = pd.read_csv(StringIO(val_response.text), sep='\t')
            
            # Convert to HuggingFace dataset format
            def convert_hans_df(df):
                return {
                    'premise': df['sentence1'].tolist(),
                    'hypothesis': df['sentence2'].tolist(),
                    'label': [0 if x == 'entailment' else 1 for x in df['gold_label'].tolist()]
                }
            
            hans_train_dict = convert_hans_df(train_data)
            hans_val_dict = convert_hans_df(val_data)
            
            hans_dataset = datasets.DatasetDict({
                'train': datasets.Dataset.from_dict(hans_train_dict),
                'validation': datasets.Dataset.from_dict(hans_val_dict)
            })
            
            print(f"✓ HANS loaded from GitHub: {len(hans_dataset['train'])} train, {len(hans_dataset['validation'])} validation")
            return hans_dataset
            
        except Exception as e2:
            print(f"✗ Could not load HANS from alternative source: {e2}")
            return None


def load_anli_dataset():
    """
    Load ANLI dataset (Adversarial NLI).
    Human-adversarial examples designed to fool models.
    """
    print("Loading ANLI dataset...")
    try:
        anli = datasets.load_dataset("anli")
        print(f"✓ ANLI loaded:")
        print(f"  Round 1: {len(anli['train_r1'])} examples")
        print(f"  Round 2: {len(anli['train_r2'])} examples")
        print(f"  Round 3: {len(anli['train_r3'])} examples")
        return anli
    except Exception as e:
        print(f"✗ Could not load ANLI: {e}")
        return None


def load_contrast_sets():
    """
    Load contrastive examples if available.
    These are minimal edits to SNLI that change labels.
    """
    print("Checking for contrastive examples...")
    # Note: Contrast sets may need to be downloaded separately
    # Repository: https://github.com/allenai/contrast-sets
    print("  (Contrast sets require separate download)")
    return None


def standardize_dataset(dataset, label_map=None):
    """
    Standardize dataset to have 'premise', 'hypothesis', 'label' columns.
    Maps labels to SNLI format: 0=entailment, 1=neutral, 2=contradiction.
    """
    if label_map:
        def map_labels(example):
            example['label'] = label_map.get(example['label'], example['label'])
            return example
        dataset = dataset.map(map_labels)
    
    return dataset


def combine_datasets(snli_train, challenge_datasets, mixing_strategy='balanced', max_challenge=50000):
    """
    Combine SNLI with challenge datasets.
    
    Strategies:
    - 'balanced': Equal weight to SNLI and challenge data
    - 'augment': Add challenge data to full SNLI (larger dataset)
    - 'challenge_only': Train only on challenge sets (risky)
    """
    print(f"\nCombining datasets with strategy: {mixing_strategy}")
    
    combined_data = {
        'premise': [],
        'hypothesis': [],
        'label': []
    }
    
    # Add SNLI data
    if mixing_strategy in ['balanced', 'augment']:
        print(f"Adding SNLI training data: {len(snli_train)} examples")
        for example in snli_train:
            combined_data['premise'].append(example['premise'])
            combined_data['hypothesis'].append(example['hypothesis'])
            combined_data['label'].append(example['label'])
    
    # Add challenge data
    total_challenge = 0
    for name, dataset in challenge_datasets.items():
        if dataset is None:
            continue
        
        # Limit challenge data if needed
        dataset_size = min(len(dataset), max_challenge // len(challenge_datasets))
        
        print(f"Adding {name}: {dataset_size} examples")
        
        for i, example in enumerate(dataset):
            if i >= dataset_size:
                break
            
            combined_data['premise'].append(example['premise'])
            combined_data['hypothesis'].append(example['hypothesis'])
            combined_data['label'].append(example['label'])
            total_challenge += 1
    
    print(f"\nTotal combined dataset:")
    print(f"  SNLI examples: {len(snli_train) if mixing_strategy != 'challenge_only' else 0}")
    print(f"  Challenge examples: {total_challenge}")
    print(f"  Total: {len(combined_data['premise'])}")
    
    # Create dataset
    combined_dataset = datasets.Dataset.from_dict(combined_data)
    
    return combined_dataset


def analyze_challenge_performance(model, tokenizer, challenge_datasets, device):
    """
    Analyze model performance on challenge sets before training.
    """
    print("\n" + "="*70)
    print("BASELINE PERFORMANCE ON CHALLENGE SETS")
    print("="*70)
    
    model.eval()
    
    for name, dataset in challenge_datasets.items():
        if dataset is None:
            continue
        
        correct = 0
        total = min(len(dataset), 1000)  # Test on subset
        
        for i, example in enumerate(dataset):
            if i >= total:
                break
            
            inputs = tokenizer(
                example['premise'],
                example['hypothesis'],
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()
                
                if pred == example['label']:
                    correct += 1
        
        accuracy = correct / total
        print(f"{name:20s}: {accuracy:.2%} ({correct}/{total})")


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help='Model to train')
    argp.add_argument('--dataset', type=str, default='snli',
                      help='Base dataset (SNLI)')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit SNLI training examples')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    argp.add_argument('--use_hans', action='store_true',
                      help='Include HANS dataset')
    argp.add_argument('--use_anli', action='store_true',
                      help='Include ANLI dataset')
    argp.add_argument('--mixing_strategy', type=str, 
                      choices=['balanced', 'augment', 'challenge_only'],
                      default='augment',
                      help='How to combine SNLI with challenge sets')
    argp.add_argument('--max_challenge', type=int, default=50000,
                      help='Maximum challenge examples to add')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 70)
    print("CHALLENGE SET TRAINING FOR NLI DEBIASING")
    print("=" * 70)
    print(f"Base dataset: {args.dataset}")
    print(f"Using HANS: {args.use_hans}")
    print(f"Using ANLI: {args.use_anli}")
    print(f"Mixing strategy: {args.mixing_strategy}")
    print("=" * 70)

    # Load SNLI
    if args.dataset == 'snli':
        snli = datasets.load_dataset('snli')
        snli = snli.filter(lambda ex: ex['label'] != -1)
        eval_split = 'validation'
    else:
        print(f"Error: Only SNLI supported as base dataset")
        return

    # Load challenge datasets
    challenge_datasets = {}
    
    if args.use_hans:
        hans = load_hans_dataset()
        if hans:
            # HANS uses different label scheme, need to map
            # HANS: entailment=0, non-entailment=1
            # We need: entailment=0, neutral=1, contradiction=2
            # For HANS "non-entailment", we map to contradiction (2)
            def map_hans_labels(example):
                if example['label'] == 0:
                    example['label'] = 0  # entailment
                else:
                    example['label'] = 2  # non-entailment -> contradiction
                return example
            
            hans_train = hans['train'].map(map_hans_labels)
            challenge_datasets['HANS'] = hans_train
    
    if args.use_anli:
        anli = load_anli_dataset()
        if anli:
            # ANLI has same label scheme as SNLI
            # Combine all rounds
            anli_combined = datasets.concatenate_datasets([
                anli['train_r1'],
                anli['train_r2'],
                anli['train_r3']
            ])
            challenge_datasets['ANLI'] = anli_combined
    
    if not challenge_datasets:
        print("\nError: No challenge datasets loaded!")
        print("Use --use_hans and/or --use_anli flags")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare training data
    snli_train = snli['train']
    if args.max_train_samples:
        snli_train = snli_train.select(range(args.max_train_samples))
    
    # Combine datasets
    combined_train = combine_datasets(
        snli_train,
        challenge_datasets,
        mixing_strategy=args.mixing_strategy,
        max_challenge=args.max_challenge
    )
    
    # Tokenize
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    
    train_dataset_featurized = combined_train.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=combined_train.column_names
    )
    
    # Prepare eval dataset (standard SNLI validation)
    eval_dataset = snli[eval_split]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )
    
    print("✓ Datasets prepared")

    # Load model
    print(f"\nInitializing model: {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    # Train
    if training_args.do_train:
        print("\n" + "="*70)
        print("TRAINING WITH CHALLENGE SETS")
        print("="*70)
        trainer.train()
        trainer.save_model()
        print("\n✓ Training complete!")

    # Evaluate
    if training_args.do_eval:
        print("\n" + "="*70)
        print("EVALUATING ON SNLI VALIDATION")
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
        
        # Also evaluate on challenge sets if available
        if args.use_hans and 'HANS' in challenge_datasets:
            print("\n" + "="*70)
            print("EVALUATING ON HANS")
            print("="*70)
            hans_eval = hans['validation']
            hans_eval = hans_eval.map(lambda ex: {'label': 0 if ex['label'] == 0 else 2})
            hans_eval_featurized = hans_eval.map(
                prepare_fn,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=hans_eval.column_names
            )
            hans_results = trainer.evaluate(hans_eval_featurized)
            print(f"HANS Accuracy: {hans_results['eval_accuracy']:.2%}")


if __name__ == "__main__":
    main()