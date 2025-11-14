"""
jainag
Error Analysis Script for Full SNLI Model
Categorizes errors and finds patterns in what the model gets wrong.
"""

import json
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LABEL_NAMES = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


def load_predictions(filepath):
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def has_negation(text):
    """Check if text contains negation words."""
    negation_words = ['not', 'no', 'never', 'nobody', 'nothing', 'nowhere', 
                      'neither', 'none', "n't", 'without']
    text_lower = text.lower()
    return any(word in text_lower for word in negation_words)


def calculate_word_overlap(premise, hypothesis):
    """Calculate word overlap between premise and hypothesis."""
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())
    
    if len(hypothesis_words) == 0:
        return 0.0
    
    overlap = len(premise_words & hypothesis_words)
    return overlap / len(hypothesis_words)


def contains_quantifiers(text):
    """Check if text contains quantifiers."""
    quantifiers = ['all', 'some', 'many', 'few', 'most', 'several', 'every', 'each']
    text_lower = text.lower()
    return any(word in text_lower.split() for word in quantifiers)


def categorize_error(example):
    """
    Categorize an error into types.
    Returns list of categories (can have multiple).
    """
    categories = []
    
    premise = example['premise']
    hypothesis = example['hypothesis']
    true_label = example['label']
    pred_label = example['predicted_label']
    
    # 1. Negation errors
    if has_negation(premise) or has_negation(hypothesis):
        categories.append('negation')
    
    # 2. Word overlap issues
    overlap = calculate_word_overlap(premise, hypothesis)
    if overlap > 0.7:
        categories.append('high_overlap')
    elif overlap < 0.2:
        categories.append('low_overlap')
    
    # 3. Length-based errors
    hyp_length = len(hypothesis.split())
    if hyp_length <= 4:
        categories.append('short_hypothesis')
    elif hyp_length >= 15:
        categories.append('long_hypothesis')
    
    # 4. Quantifier reasoning
    if contains_quantifiers(premise) or contains_quantifiers(hypothesis):
        categories.append('quantifier')
    
    # 5. Specific confusion patterns
    if true_label == 0 and pred_label == 1:  # True: entailment, Pred: neutral
        categories.append('missed_entailment')
    elif true_label == 1 and pred_label == 0:  # True: neutral, Pred: entailment
        categories.append('false_entailment')
    elif true_label == 2 and pred_label == 0:  # True: contradiction, Pred: entailment
        categories.append('missed_contradiction')
    elif true_label == 0 and pred_label == 2:  # True: entailment, Pred: contradiction
        categories.append('false_contradiction')
    
    # If no specific category, mark as other
    if not categories:
        categories.append('other')
    
    return categories


def analyze_errors_by_category(predictions):
    """Analyze errors broken down by category."""
    print("\n" + "="*70)
    print("ERROR ANALYSIS BY CATEGORY")
    print("="*70)
    
    errors = [p for p in predictions if not p['correct']]
    total_errors = len(errors)
    
    print(f"\nTotal errors: {total_errors} out of {len(predictions)} examples")
    print(f"Error rate: {total_errors/len(predictions)*100:.2f}%\n")
    
    # Categorize all errors
    category_counts = Counter()
    examples_by_category = defaultdict(list)
    
    for error in errors:
        categories = categorize_error(error)
        for cat in categories:
            category_counts[cat] += 1
            if len(examples_by_category[cat]) < 5:  # Store up to 5 examples per category
                examples_by_category[cat].append(error)
    
    # Sort by frequency
    sorted_categories = category_counts.most_common()
    
    print("ERROR BREAKDOWN:")
    print("-" * 70)
    for category, count in sorted_categories:
        percentage = (count / total_errors) * 100
        print(f"{category:25s}: {count:4d} ({percentage:5.2f}%)")
    
    # Show examples for top categories
    print("\n" + "="*70)
    print("EXAMPLE ERRORS BY CATEGORY")
    print("="*70)
    
    for category, _ in sorted_categories[:5]:  # Top 5 categories
        print(f"\n{'─'*70}")
        print(f"Category: {category.upper()}")
        print('─'*70)
        
        for i, example in enumerate(examples_by_category[category][:3], 1):
            true_label = LABEL_NAMES[example['label']]
            pred_label = LABEL_NAMES[example['predicted_label']]
            
            print(f"\nExample {i}:")
            print(f"  Premise:    {example['premise']}")
            print(f"  Hypothesis: {example['hypothesis']}")
            print(f"  True label: {true_label}")
            print(f"  Predicted:  {pred_label}")
    
    return category_counts, examples_by_category


def analyze_confusion_patterns(predictions):
    """Analyze specific confusion patterns."""
    print("\n" + "="*70)
    print("CONFUSION PATTERN ANALYSIS")
    print("="*70)
    
    confusion_examples = defaultdict(list)
    confusion_counts = Counter()
    
    for pred in predictions:
        if not pred['correct']:
            true_label = LABEL_NAMES[pred['label']]
            pred_label = LABEL_NAMES[pred['predicted_label']]
            confusion_key = f"{true_label} → {pred_label}"
            confusion_counts[confusion_key] += 1
            
            if len(confusion_examples[confusion_key]) < 3:
                confusion_examples[confusion_key].append(pred)
    
    print("\nMost common confusion patterns:")
    print("-" * 70)
    for pattern, count in confusion_counts.most_common():
        print(f"{pattern:30s}: {count:4d} errors")
    
    # Show examples of most common confusions
    print("\n" + "="*70)
    print("EXAMPLES OF COMMON CONFUSIONS")
    print("="*70)
    
    for pattern, _ in confusion_counts.most_common(3):
        print(f"\n{'─'*70}")
        print(f"Pattern: {pattern}")
        print('─'*70)
        
        for i, example in enumerate(confusion_examples[pattern], 1):
            print(f"\nExample {i}:")
            print(f"  Premise:    {example['premise']}")
            print(f"  Hypothesis: {example['hypothesis']}")
            print(f"  Overlap:    {calculate_word_overlap(example['premise'], example['hypothesis']):.2%}")


def analyze_overlap_vs_accuracy(predictions):
    """Analyze how word overlap affects accuracy."""
    print("\n" + "="*70)
    print("WORD OVERLAP vs ACCURACY ANALYSIS")
    print("="*70)
    
    # Bin predictions by overlap
    overlap_bins = {
        'Very Low (0-20%)': [],
        'Low (20-40%)': [],
        'Medium (40-60%)': [],
        'High (60-80%)': [],
        'Very High (80-100%)': []
    }
    
    for pred in predictions:
        overlap = calculate_word_overlap(pred['premise'], pred['hypothesis'])
        
        if overlap < 0.2:
            overlap_bins['Very Low (0-20%)'].append(pred)
        elif overlap < 0.4:
            overlap_bins['Low (20-40%)'].append(pred)
        elif overlap < 0.6:
            overlap_bins['Medium (40-60%)'].append(pred)
        elif overlap < 0.8:
            overlap_bins['High (60-80%)'].append(pred)
        else:
            overlap_bins['Very High (80-100%)'].append(pred)
    
    print("\nAccuracy by word overlap:")
    print("-" * 70)
    
    for bin_name, bin_preds in overlap_bins.items():
        if bin_preds:
            accuracy = sum(1 for p in bin_preds if p['correct']) / len(bin_preds)
            print(f"{bin_name:25s}: {accuracy:.2%} ({len(bin_preds)} examples)")


def create_visualizations(predictions, category_counts, output_dir):
    """Create visualization plots."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Error category breakdown
    categories = [cat for cat, _ in category_counts.most_common(10)]
    counts = [count for _, count in category_counts.most_common(10)]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(categories, counts, color='#e74c3c')
    plt.xlabel('Number of Errors')
    plt.title('Top Error Categories in Full Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_categories.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'error_categories.png'}")
    plt.close()
    
    # 2. Word overlap distribution for correct vs incorrect
    correct_overlaps = []
    incorrect_overlaps = []
    
    for pred in predictions:
        overlap = calculate_word_overlap(pred['premise'], pred['hypothesis'])
        if pred['correct']:
            correct_overlaps.append(overlap)
        else:
            incorrect_overlaps.append(overlap)
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_overlaps, bins=20, alpha=0.5, label='Correct', color='green')
    plt.hist(incorrect_overlaps, bins=20, alpha=0.5, label='Incorrect', color='red')
    plt.xlabel('Word Overlap Ratio')
    plt.ylabel('Frequency')
    plt.title('Word Overlap: Correct vs Incorrect Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'overlap_distribution.png'}")
    plt.close()
    
    # 3. Confusion matrix
    true_labels = [p['label'] for p in predictions]
    pred_labels = [p['predicted_label'] for p in predictions]
    
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[true][pred] += 1
    
    # Normalize by row (true label)
    confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    im1 = ax1.imshow(confusion, cmap='Blues')
    ax1.set_xticks(np.arange(3))
    ax1.set_yticks(np.arange(3))
    ax1.set_xticklabels(['Ent', 'Neu', 'Con'])
    ax1.set_yticklabels(['Ent', 'Neu', 'Con'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, confusion[i, j], ha="center", va="center", color="black")
    
    # Normalized
    im2 = ax2.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(3))
    ax2.set_yticks(np.arange(3))
    ax2.set_xticklabels(['Ent', 'Neu', 'Con'])
    ax2.set_yticklabels(['Ent', 'Neu', 'Con'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{confusion_norm[i, j]:.2f}', 
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'full_model_confusion.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'full_model_confusion.png'}")
    plt.close()
    
    print("\n✓ All visualizations created!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze full model errors')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to eval_predictions.jsonl from full model')
    parser.add_argument('--output_dir', type=str, default='error_analysis_outputs',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL MODEL ERROR ANALYSIS")
    print("="*70)
    print(f"Loading predictions from: {args.predictions}")
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    # Overall stats
    correct = sum(1 for p in predictions if p['correct'])
    accuracy = correct / len(predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
    
    # Run analyses
    category_counts, examples_by_category = analyze_errors_by_category(predictions)
    analyze_confusion_patterns(predictions)
    analyze_overlap_vs_accuracy(predictions)
    
    # Create visualizations
    create_visualizations(predictions, category_counts, args.output_dir)
    
    # Save detailed error report
    errors = [p for p in predictions if not p['correct']]
    error_report = []
    
    for error in errors:
        categories = categorize_error(error)
        overlap = calculate_word_overlap(error['premise'], error['hypothesis'])
        
        error_report.append({
            'premise': error['premise'],
            'hypothesis': error['hypothesis'],
            'true_label': LABEL_NAMES[error['label']],
            'predicted_label': LABEL_NAMES[error['predicted_label']],
            'word_overlap': f"{overlap:.2%}",
            'categories': ', '.join(categories),
            'has_negation': has_negation(error['premise']) or has_negation(error['hypothesis'])
        })
    
    # Save to CSV for manual inspection
    df = pd.DataFrame(error_report)
    output_path = Path(args.output_dir) / 'detailed_errors.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed error report: {output_path}")
    print(f"  (You can open this in Excel/Google Sheets for manual inspection)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nKey findings to include in your report:")
    print("1. Error rate and most common error categories")
    print("2. Specific confusion patterns (e.g., entailment → neutral)")
    print("3. How word overlap affects model performance")
    print("4. Examples of each major error type")
    print(f"\nCheck '{args.output_dir}' for visualizations and detailed error CSV")


if __name__ == '__main__':
    main()
