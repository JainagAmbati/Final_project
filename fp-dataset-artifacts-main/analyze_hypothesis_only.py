"""
Analysis script for hypothesis-only predictions.
This script helps you understand what artifacts the hypothesis-only model learned.
"""

import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Label mapping
LABEL_NAMES = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
LABEL_NAMES_REV = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


def load_predictions(filepath):
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def analyze_overall_performance(predictions):
    """Analyze overall accuracy and confusion matrix."""
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate accuracy
    correct = sum(1 for p in predictions if p['correct'])
    total = len(predictions)
    accuracy = correct / total
    
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Random baseline (33.3%): {1/3:.2%}")
    print(f"Improvement over random: {accuracy - 1/3:.2%}")
    
    # Confusion matrix
    true_labels = [p['label'] for p in predictions]
    pred_labels = [p['predicted_label'] for p in predictions]
    
    # Create confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[true][pred] += 1
    
    print("\nConfusion Matrix:")
    print("                 Predicted:")
    print(f"              Ent    Neu    Con")
    print(f"True: Ent    {confusion[0][0]:4d}   {confusion[0][1]:4d}   {confusion[0][2]:4d}")
    print(f"      Neu    {confusion[1][0]:4d}   {confusion[1][1]:4d}   {confusion[1][2]:4d}")
    print(f"      Con    {confusion[2][0]:4d}   {confusion[2][1]:4d}   {confusion[2][2]:4d}")
    
    # Per-class accuracy
    print("\nPer-class Performance:")
    for label_id, label_name in LABEL_NAMES.items():
        class_total = sum(1 for p in predictions if p['label'] == label_id)
        class_correct = sum(1 for p in predictions if p['label'] == label_id and p['correct'])
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label_name:15s}: {class_acc:.2%} ({class_correct}/{class_total})")
    
    return confusion


def analyze_word_correlations(predictions, top_n=20):
    """Find words in hypotheses that correlate with specific labels."""
    print("\n" + "="*60)
    print("WORD-LABEL CORRELATION ANALYSIS")
    print("="*60)
    print("(Finding words that appear more often with specific labels)\n")
    
    # Count words per label
    words_by_label = defaultdict(lambda: Counter())
    total_by_label = Counter()
    
    for pred in predictions:
        label = pred['label']
        hypothesis = pred['hypothesis'].lower()
        words = hypothesis.split()
        
        for word in words:
            # Clean word
            word = word.strip('.,!?;:')
            if len(word) > 2:  # Ignore very short words
                words_by_label[label][word] += 1
        
        total_by_label[label] += 1
    
    # Calculate correlation strength (normalized frequency)
    for label_id, label_name in LABEL_NAMES.items():
        print(f"\n{label_name.upper()} - Most predictive words:")
        print("-" * 40)
        
        # Get word frequencies normalized by label frequency
        word_freq = words_by_label[label_id]
        
        # Calculate relative frequency (compared to other labels)
        relative_scores = {}
        for word, count in word_freq.items():
            # Frequency in this label
            freq_this = count / total_by_label[label_id]
            # Average frequency in other labels
            freq_other = sum(words_by_label[other][word] / total_by_label[other] 
                           for other in LABEL_NAMES.keys() if other != label_id) / 2
            
            # Relative score (how much more common in this label)
            if freq_other > 0:
                relative_scores[word] = freq_this / freq_other
            else:
                relative_scores[word] = freq_this * 100  # Very unique to this label
        
        # Get top words
        top_words = sorted(relative_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        for i, (word, score) in enumerate(top_words, 1):
            count = word_freq[word]
            print(f"  {i:2d}. {word:15s} (appears {count:4d} times, score: {score:.2f}x)")


def analyze_correct_vs_incorrect(predictions):
    """Analyze what the model gets right vs wrong."""
    print("\n" + "="*60)
    print("CORRECT vs INCORRECT PREDICTIONS")
    print("="*60)
    
    correct_preds = [p for p in predictions if p['correct']]
    incorrect_preds = [p for p in predictions if not p['correct']]
    
    print(f"\nCorrect predictions: {len(correct_preds)}")
    print(f"Incorrect predictions: {len(incorrect_preds)}")
    
    # Show some examples of correct predictions (the artifacts!)
    print("\n" + "-"*60)
    print("EXAMPLES OF CORRECT HYPOTHESIS-ONLY PREDICTIONS:")
    print("(These reveal the artifacts the model exploits!)")
    print("-"*60)
    
    # Sample correct predictions from each label
    for label_id, label_name in LABEL_NAMES.items():
        correct_for_label = [p for p in correct_preds if p['label'] == label_id]
        if correct_for_label:
            print(f"\n{label_name.upper()} examples (predicted correctly with hypothesis only):")
            for pred in correct_for_label[:3]:  # Show 3 examples
                print(f"  Hypothesis: {pred['hypothesis']}")
                print(f"  True label: {label_name}")
                print()


def analyze_hypothesis_length(predictions):
    """Analyze if hypothesis length correlates with labels."""
    print("\n" + "="*60)
    print("HYPOTHESIS LENGTH ANALYSIS")
    print("="*60)
    
    lengths_by_label = defaultdict(list)
    
    for pred in predictions:
        length = len(pred['hypothesis'].split())
        lengths_by_label[pred['label']].append(length)
    
    print("\nAverage hypothesis length by TRUE label:")
    for label_id, label_name in LABEL_NAMES.items():
        lengths = lengths_by_label[label_id]
        if lengths:
            avg_length = np.mean(lengths)
            print(f"  {label_name:15s}: {avg_length:.2f} words")
    
    # Check if model exploits length
    print("\nDoes the model exploit length bias?")
    pred_lengths_by_pred_label = defaultdict(list)
    for pred in predictions:
        length = len(pred['hypothesis'].split())
        pred_lengths_by_pred_label[pred['predicted_label']].append(length)
    
    print("Average hypothesis length by PREDICTED label:")
    for label_id, label_name in LABEL_NAMES.items():
        lengths = pred_lengths_by_pred_label[label_id]
        if lengths:
            avg_length = np.mean(lengths)
            print(f"  {label_name:15s}: {avg_length:.2f} words")


def create_visualizations(predictions, output_dir):
    """Create visualization plots."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    true_labels = [p['label'] for p in predictions]
    pred_labels = [p['predicted_label'] for p in predictions]
    
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[true][pred] += 1
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Entailment', 'Neutral', 'Contradiction'],
                yticklabels=['Entailment', 'Neutral', 'Contradiction'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Hypothesis-Only Model: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'confusion_matrix.png'}")
    plt.close()
    
    # 2. Per-class Accuracy Bar Chart
    accuracies = []
    for label_id in range(3):
        class_total = sum(1 for p in predictions if p['label'] == label_id)
        class_correct = sum(1 for p in predictions if p['label'] == label_id and p['correct'])
        class_acc = class_correct / class_total if class_total > 0 else 0
        accuracies.append(class_acc * 100)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Entailment', 'Neutral', 'Contradiction'], accuracies, 
                   color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.axhline(y=33.33, color='r', linestyle='--', label='Random Baseline (33.3%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Hypothesis-Only Model: Per-Class Accuracy')
    plt.ylim(0, 100)
    plt.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'per_class_accuracy.png'}")
    plt.close()
    
    # 3. Hypothesis Length Distribution
    lengths_by_label = defaultdict(list)
    for pred in predictions:
        length = len(pred['hypothesis'].split())
        lengths_by_label[pred['label']].append(length)
    
    plt.figure(figsize=(10, 5))
    for label_id, label_name in LABEL_NAMES.items():
        plt.hist(lengths_by_label[label_id], bins=20, alpha=0.5, label=label_name)
    plt.xlabel('Hypothesis Length (words)')
    plt.ylabel('Frequency')
    plt.title('Hypothesis Length Distribution by Label')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'length_distribution.png'}")
    plt.close()
    
    print("\n✓ All visualizations created!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze hypothesis-only model predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to eval_predictions.jsonl file')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs',
                       help='Directory to save analysis outputs')
    parser.add_argument('--top_n_words', type=int, default=20,
                       help='Number of top words to show for each label')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HYPOTHESIS-ONLY MODEL ANALYSIS")
    print("="*60)
    print(f"Loading predictions from: {args.predictions}")
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Run analyses
    confusion = analyze_overall_performance(predictions)
    analyze_word_correlations(predictions, top_n=args.top_n_words)
    analyze_hypothesis_length(predictions)
    analyze_correct_vs_incorrect(predictions)
    
    # Create visualizations
    create_visualizations(predictions, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nCheck the '{args.output_dir}' directory for visualizations.")
    print("\nKey takeaways to include in your report:")
    print("1. Overall hypothesis-only accuracy vs random baseline")
    print("2. Which label classes are easiest to predict from hypothesis alone")
    print("3. What words/patterns correlate with each label")
    print("4. Example predictions that succeeded with hypothesis only")
    print("5. Any length biases in the data")


if __name__ == '__main__':
    main()
