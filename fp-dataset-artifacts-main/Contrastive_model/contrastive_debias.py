"""
Contrastive Data Augmentation for NLI Debiasing

Creates augmented training examples by:
1. Adding negations to create contradictions
2. Adding specific details to create neutral examples
3. Replacing with hypernyms/hyponyms for entailment
4. Word substitutions that change labels

This forces the model to learn semantic understanding beyond lexical overlap.
"""

import sys
sys.path.insert(0, '..')

import datasets
import random
import re
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
import numpy as np

NUM_PREPROCESSING_WORKERS = 2

# Negation words for creating contradictions
NEGATION_WORDS = ['not', 'never', 'no', "n't"]

# Words that add specific details (for creating neutral examples)
SPECIFIC_DETAILS = [
    'quickly', 'slowly', 'happily', 'sadly', 'angrily',
    'red', 'blue', 'green', 'yellow', 'black', 'white',
    'tall', 'short', 'young', 'old', 'large', 'small',
    'inside', 'outside', 'indoors', 'outdoors',
    'because it is raining', 'in the morning', 'at night'
]

# Hypernym/hyponym pairs for entailment
HYPERNYM_PAIRS = {
    'dog': 'animal',
    'cat': 'animal',
    'bird': 'animal',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bicycle': 'vehicle',
    'rose': 'flower',
    'tulip': 'flower',
    'apple': 'fruit',
    'banana': 'fruit',
    'running': 'moving',
    'walking': 'moving',
    'sprinting': 'running',
    'jogging': 'running',
    'man': 'person',
    'woman': 'person',
    'boy': 'child',
    'girl': 'child',
}


def has_negation(text):
    """Check if text contains negation."""
    return any(neg in text.lower() for neg in NEGATION_WORDS)


def add_negation_to_sentence(sentence):
    """
    Add negation to a sentence to create contradiction.
    Simple heuristic: add 'not' before main verb or 'is/are'.
    """
    # Pattern 1: "X is Y" -> "X is not Y"
    sentence = re.sub(r'\b(is|are|was|were)\b', r'\1 not', sentence, count=1)
    
    # If no auxiliary verb, try to add "does not" or "do not"
    if 'not' not in sentence.lower():
        sentence = re.sub(r'\b(walks|runs|plays|eats|sits)\b', r'does not \1', sentence, count=1)
        sentence = re.sub(r'\b(walk|run|play|eat|sit)\b', r'do not \1', sentence, count=1)
    
    return sentence


def add_specific_detail(sentence):
    """
    Add specific detail to create neutral example.
    """
    detail = random.choice(SPECIFIC_DETAILS)
    
    # Add at the end if it's a phrase
    if ' ' in detail:
        return sentence.rstrip('.') + ' ' + detail + '.'
    
    # Add before a noun if it's an adjective
    words = sentence.split()
    if len(words) > 3:
        # Insert detail before a random noun (simplified heuristic)
        insert_pos = random.randint(1, len(words) - 1)
        words.insert(insert_pos, detail)
        return ' '.join(words)
    
    return sentence.rstrip('.') + ', ' + detail + '.'


def replace_with_hypernym(sentence, word_pairs):
    """
    Replace specific words with hypernyms to create entailment.
    """
    words = sentence.lower().split()
    for i, word in enumerate(words):
        clean_word = word.strip('.,!?;:')
        if clean_word in word_pairs:
            words[i] = word.replace(clean_word, word_pairs[clean_word])
            break
    return ' '.join(words)


def create_contrastive_examples(example, augmentation_prob=0.3):
    """
    Create contrastive augmented examples from an original example.
    
    Returns list of (premise, hypothesis, label) tuples including:
    - Original example
    - Potentially augmented examples with different labels
    """
    premise = example['premise']
    hypothesis = example['hypothesis']
    original_label = example['label']
    
    examples = [(premise, hypothesis, original_label)]
    
    # Don't augment if random check fails
    if random.random() > augmentation_prob:
        return examples
    
    # Strategy 1: Add negation to create contradiction (if original is entailment)
    if original_label == 0 and not has_negation(hypothesis):  # entailment
        neg_hypothesis = add_negation_to_sentence(hypothesis)
        if neg_hypothesis != hypothesis and 'not' in neg_hypothesis.lower():
            examples.append((premise, neg_hypothesis, 2))  # contradiction
    
    # Strategy 2: Add specific details to create neutral (if original is entailment)
    if original_label == 0:  # entailment
        detailed_hypothesis = add_specific_detail(hypothesis)
        if detailed_hypothesis != hypothesis:
            examples.append((premise, detailed_hypothesis, 1))  # neutral
    
    # Strategy 3: Replace with hypernym (if words match)
    hyp_words = set(hypothesis.lower().split())
    matching_words = [w for w in HYPERNYM_PAIRS.keys() if w in ' '.join(hyp_words)]
    
    if matching_words and original_label == 0:  # Can only create entailment from entailment
        hypernym_hypothesis = replace_with_hypernym(hypothesis, HYPERNYM_PAIRS)
        if hypernym_hypothesis != hypothesis.lower():
            examples.append((premise, hypernym_hypothesis, 0))  # entailment
    
    return examples


def augment_dataset(dataset, augmentation_rate=0.3, max_augmented=None):
    """
    Augment dataset with contrastive examples.
    
    Args:
        dataset: Original dataset
        augmentation_rate: Probability of augmenting each example
        max_augmented: Maximum number of augmented examples to add (None = no limit)
    
    Returns:
        Augmented dataset
    """
    print(f"\nAugmenting dataset with contrastive examples...")
    print(f"Augmentation rate: {augmentation_rate}")
    
    augmented_data = {
        'premise': [],
        'hypothesis': [],
        'label': []
    }
    
    augmented_count = 0
    
    for example in dataset:
        # Get augmented examples
        examples = create_contrastive_examples(example, augmentation_rate)
        
        # Add all examples
        for premise, hypothesis, label in examples:
            augmented_data['premise'].append(premise)
            augmented_data['hypothesis'].append(hypothesis)
            augmented_data['label'].append(label)
            
            if len(examples) > 1:  # Count augmented examples only
                augmented_count += 1
        
        # Check if we've reached max augmented examples
        if max_augmented and augmented_count >= max_augmented:
            print(f"Reached maximum augmented examples: {max_augmented}")
            break
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Augmented examples created: {augmented_count}")
    print(f"Total dataset size: {len(augmented_data['premise'])}")
    print(f"Augmentation ratio: {augmented_count / len(dataset) * 100:.1f}%")
    
    # Create dataset from dict
    augmented_dataset = datasets.Dataset.from_dict(augmented_data)
    
    return augmented_dataset


def analyze_augmentation_quality(dataset, augmented_dataset):
    """Print some examples to verify augmentation quality."""
    print("\n" + "="*70)
    print("AUGMENTATION QUALITY CHECK")
    print("="*70)
    
    # Find some augmented examples
    original_size = len(dataset)
    
    print("\nSample augmented examples:")
    print("-"*70)
    
    samples_shown = 0
    for i in range(min(len(augmented_dataset), original_size + 100)):
        if i >= original_size and samples_shown < 5:
            example = augmented_dataset[i]
            print(f"\nExample {samples_shown + 1}:")
            print(f"  Premise:    {example['premise'][:80]}...")
            print(f"  Hypothesis: {example['hypothesis']}")
            print(f"  Label:      {['Entailment', 'Neutral', 'Contradiction'][example['label']]}")
            samples_shown += 1


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
                      help='Limit training examples')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    argp.add_argument('--augmentation_rate', type=float, default=0.3,
                      help='Probability of augmenting each example (0-1)')
    argp.add_argument('--max_augmented', type=int, default=50000,
                      help='Maximum number of augmented examples to create')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 70)
    print("CONTRASTIVE DATA AUGMENTATION FOR NLI DEBIASING")
    print("=" * 70)
    print(f"Augmentation rate: {args.augmentation_rate}")
    print(f"Max augmented examples: {args.max_augmented}")
    print("=" * 70)

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare datasets
    print("\nPreparing datasets...")
    
    train_dataset = dataset['train']
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    # Augment training dataset
    augmented_train_dataset = augment_dataset(
        train_dataset, 
        augmentation_rate=args.augmentation_rate,
        max_augmented=args.max_augmented
    )
    
    # Analyze quality
    analyze_augmentation_quality(train_dataset, augmented_train_dataset)
    
    # Tokenize
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    
    train_dataset_featurized = augmented_train_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=augmented_train_dataset.column_names
    )
    
    # Prepare eval dataset (no augmentation)
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
        print("TRAINING WITH CONTRASTIVE AUGMENTATION")
        print("="*70)
        trainer.train()
        trainer.save_model()
        print("\n✓ Training complete!")

    # Evaluate
    if training_args.do_eval:
        print("\n" + "="*70)
        print("EVALUATING AUGMENTED MODEL")
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
