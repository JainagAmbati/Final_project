"""
Compare performance between baseline and debiased models.
Analyzes where improvements occur.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

LABEL_NAMES = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


def load_predictions(filepath):
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r') as f:
        for line in f:
            pred = json.loads(line)
            if 'correct' not in pred:
                pred['correct'] = (pred['predicted_label'] == pred['label'])
            predictions.append(pred)
    return predictions


def calculate_word_overlap(premise, hypothesis):
    """Calculate word overlap ratio."""
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())
    if len(hypothesis_words) == 0:
        return 0.0
    return len(premise_words & hypothesis_words) / len(hypothesis_words)


def has_negation(text):
    """Check for negation words."""
    negation_words = ['not', 'no', 'never', 'nobody', 'nothing', 'nowhere', 
                      'neither', 'none', "n't", 'without']
    return any(word in text.lower() for word in negation_words)


def compare_overall_performance(baseline_preds, debiased_preds):
    """Compare overall accuracy."""
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE COMPARISON")
    print("="*70)
    
    baseline_acc = sum(1 for p in baseline_preds if p['correct']) / len(baseline_preds)
    debiased_acc = sum(1 for p in debiased_preds if p['correct']) / len(debiased_preds)
    
    print(f"\nBaseline Model:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"Debiased Model:  {debiased_acc:.4f} ({debiased_acc*100:.2f}%)")
    print(f"Improvement:     {(debiased_acc - baseline_acc):.4f} ({(debiased_acc - baseline_acc)*100:.2f}%)")
    
    # Per-class accuracy
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY")
    print("-"*70)
    print(f"{'Label':<20} {'Baseline':<15} {'Debiased':<15} {'Change':<15}")
    print("-"*70)
    
    for label_id, label_name in LABEL_NAMES.items():
        baseline_class = [p for p in baseline_preds if p['label'] == label_id]
        debiased_class = [p for p in debiased_preds if p['label'] == label_id]
        
        baseline_class_acc = sum(1 for p in baseline_class if p['correct']) / len(baseline_class)
        debiased_class_acc = sum(1 for p in debiased_class if p['correct']) / len(debiased_class)
        change = debiased_class_acc - baseline_class_acc
        
        print(f"{label_name:<20} {baseline_class_acc:.4f} ({baseline_class_acc*100:.1f}%)  "
              f"{debiased_class_acc:.4f} ({debiased_class_acc*100:.1f}%)  "
              f"{change:+.4f} ({change*100:+.1f}%)")
    
    return baseline_acc, debiased_acc


def analyze_fixed_errors(baseline_preds, debiased_preds):
    """Find examples where debiased model fixed baseline errors."""
    print("\n" + "="*70)
    print("EXAMPLES FIXED BY DEBIASING")
    print("="*70)
    
    fixed_examples = []
    new_errors = []
    
    for baseline, debiased in zip(baseline_preds, debiased_preds):
        # Verify they're the same example
        assert baseline['premise'] == debiased['premise']
        assert baseline['hypothesis'] == debiased['hypothesis']
        
        if not baseline['correct'] and debiased['correct']:
            fixed_examples.append(debiased)
        elif baseline['correct'] and not debiased['correct']:
            new_errors.append(debiased)
    
    print(f"\nExamples fixed by debiased model: {len(fixed_examples)}")
    print(f"New errors introduced: {len(new_errors)}")
    print(f"Net improvement: {len(fixed_examples) - len(new_errors)}")
    
    # Show some fixed examples
    if fixed_examples:
        print("\n" + "-"*70)
        print("EXAMPLES OF FIXES (showing first 10):")
        print("-"*70)
        
        for i, example in enumerate(fixed_examples[:10], 1):
            baseline_pred = next(p for p in baseline_preds 
                               if p['premise'] == example['premise'] and 
                                  p['hypothesis'] == example['hypothesis'])
            
            overlap = calculate_word_overlap(example['premise'], example['hypothesis'])
            
            print(f"\nExample {i}:")
            print(f"  Premise:         {example['premise']}")
            print(f"  Hypothesis:      {example['hypothesis']}")
            print(f"  True label:      {LABEL_NAMES[example['label']]}")
            print(f"  Baseline pred:   {LABEL_NAMES[baseline_pred['predicted_label']]} ✗")
            print(f"  Debiased pred:   {LABEL_NAMES[example['predicted_label']]} ✓")
            print(f"  Word overlap:    {overlap:.2%}")
            print(f"  Has negation:    {has_negation(example['premise']) or has_negation(example['hypothesis'])}")
    
    return fixed_examples, new_errors


def analyze_by_word_overlap(baseline_preds, debiased_preds):
    """Compare performance by word overlap bins."""
    print("\n" + "="*70)
    print("PERFORMANCE BY WORD OVERLAP")
    print("="*70)
    
    overlap_bins = {
        'Very Low (0-20%)': (0.0, 0.2),
        'Low (20-40%)': (0.2, 0.4),
        'Medium (40-60%)': (0.4, 0.6),
        'High (60-80%)': (0.6, 0.8),
        'Very High (80-100%)': (0.8, 1.0)
    }
    
    print(f"\n{'Overlap Range':<25} {'Baseline':<15} {'Debiased':<15} {'Change':<15} {'Count':<10}")
    print("-"*80)
    
    overlap_results = {}
    
    for bin_name, (min_overlap, max_overlap) in overlap_bins.items():
        baseline_bin = []
        debiased_bin = []
        
        for baseline, debiased in zip(baseline_preds, debiased_preds):
            overlap = calculate_word_overlap(baseline['premise'], baseline['hypothesis'])
            
            if min_overlap <= overlap < max_overlap or (max_overlap == 1.0 and overlap == 1.0):
                baseline_bin.append(baseline)
                debiased_bin.append(debiased)
        
        if baseline_bin:
            baseline_acc = sum(1 for p in baseline_bin if p['correct']) / len(baseline_bin)
            debiased_acc = sum(1 for p in debiased_bin if p['correct']) / len(debiased_bin)
            change = debiased_acc - baseline_acc
            
            overlap_results[bin_name] = {
                'baseline': baseline_acc,
                'debiased': debiased_acc,
                'change': change,
                'count': len(baseline_bin)
            }
            
            print(f"{bin_name:<25} {baseline_acc:.4f} ({baseline_acc*100:.1f}%)  "
                  f"{debiased_acc:.4f} ({debiased_acc*100:.1f}%)  "
                  f"{change:+.4f} ({change*100:+.2f}%)  "
                  f"{len(baseline_bin):<10}")
    
    return overlap_results


def analyze_by_negation(baseline_preds, debiased_preds):
    """Compare performance on examples with/without negation."""
    print("\n" + "="*70)
    print("PERFORMANCE ON NEGATION EXAMPLES")
    print("="*70)
    
    baseline_with_neg = [p for p in baseline_preds 
                         if has_negation(p['premise']) or has_negation(p['hypothesis'])]
    debiased_with_neg = [p for p in debiased_preds 
                         if has_negation(p['premise']) or has_negation(p['hypothesis'])]
    
    baseline_without_neg = [p for p in baseline_preds 
                           if not (has_negation(p['premise']) or has_negation(p['hypothesis']))]
    debiased_without_neg = [p for p in debiased_preds 
                           if not (has_negation(p['premise']) or has_negation(p['hypothesis']))]
    
    baseline_neg_acc = sum(1 for p in baseline_with_neg if p['correct']) / len(baseline_with_neg)
    debiased_neg_acc = sum(1 for p in debiased_with_neg if p['correct']) / len(debiased_with_neg)
    
    baseline_no_neg_acc = sum(1 for p in baseline_without_neg if p['correct']) / len(baseline_without_neg)
    debiased_no_neg_acc = sum(1 for p in debiased_without_neg if p['correct']) / len(debiased_without_neg)
    
    print(f"\n{'Category':<30} {'Baseline':<15} {'Debiased':<15} {'Change':<15} {'Count':<10}")
    print("-"*80)
    print(f"{'With Negation':<30} {baseline_neg_acc:.4f} ({baseline_neg_acc*100:.1f}%)  "
          f"{debiased_neg_acc:.4f} ({debiased_neg_acc*100:.1f}%)  "
          f"{(debiased_neg_acc - baseline_neg_acc):+.4f}  "
          f"{len(baseline_with_neg):<10}")
    print(f"{'Without Negation':<30} {baseline_no_neg_acc:.4f} ({baseline_no_neg_acc*100:.1f}%)  "
          f"{debiased_no_neg_acc:.4f} ({debiased_no_neg_acc*100:.1f}%)  "
          f"{(debiased_no_neg_acc - baseline_no_neg_acc):+.4f}  "
          f"{len(baseline_without_neg):<10}")


def create_comparison_visualizations(baseline_preds, debiased_preds, overlap_results, output_dir):
    """Create comparison visualizations."""
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Overall accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy
    baseline_acc = sum(1 for p in baseline_preds if p['correct']) / len(baseline_preds)
    debiased_acc = sum(1 for p in debiased_preds if p['correct']) / len(debiased_preds)
    
    axes[0].bar(['Baseline', 'Debiased'], [baseline_acc * 100, debiased_acc * 100], 
                color=['#3498db', '#2ecc71'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Overall Accuracy Comparison')
    axes[0].set_ylim(85, 92)
    
    for i, (label, acc) in enumerate([('Baseline', baseline_acc), ('Debiased', debiased_acc)]):
        axes[0].text(i, acc * 100 + 0.2, f'{acc*100:.2f}%', ha='center', fontweight='bold')
    
    # Per-class accuracy
    labels = list(LABEL_NAMES.values())
    baseline_class_accs = []
    debiased_class_accs = []
    
    for label_id in range(3):
        baseline_class = [p for p in baseline_preds if p['label'] == label_id]
        debiased_class = [p for p in debiased_preds if p['label'] == label_id]
        
        baseline_class_accs.append(
            sum(1 for p in baseline_class if p['correct']) / len(baseline_class) * 100
        )
        debiased_class_accs.append(
            sum(1 for p in debiased_class if p['correct']) / len(debiased_class) * 100
        )
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[1].bar(x - width/2, baseline_class_accs, width, label='Baseline', color='#3498db')
    axes[1].bar(x + width/2, debiased_class_accs, width, label='Debiased', color='#2ecc71')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Per-Class Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].set_ylim(80, 95)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()
    
    # 2. Performance by word overlap
    if overlap_results:
        bin_names = list(overlap_results.keys())
        baseline_accs = [overlap_results[bn]['baseline'] * 100 for bn in bin_names]
        debiased_accs = [overlap_results[bn]['debiased'] * 100 for bn in bin_names]
        
        x = np.arange(len(bin_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='#3498db')
        ax.bar(x + width/2, debiased_accs, width, label='Debiased', color='#2ecc71')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Performance by Word Overlap')
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(80, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overlap_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'overlap_comparison.png'}")
        plt.close()
    
    print("\n✓ All visualizations created!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline and debiased models')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline model eval_predictions.jsonl')
    parser.add_argument('--debiased', type=str, required=True,
                       help='Path to debiased model eval_predictions.jsonl')
    parser.add_argument('--output_dir', type=str, default='comparison_outputs',
                       help='Directory to save comparison outputs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BASELINE vs DEBIASED MODEL COMPARISON")
    print("="*70)
    
    # Load predictions
    print(f"\nLoading baseline predictions: {args.baseline}")
    baseline_preds = load_predictions(args.baseline)
    
    print(f"Loading debiased predictions: {args.debiased}")
    debiased_preds = load_predictions(args.debiased)
    
    print(f"\nLoaded {len(baseline_preds)} prediction pairs")
    
    # Run comparisons
    baseline_acc, debiased_acc = compare_overall_performance(baseline_preds, debiased_preds)
    fixed_examples, new_errors = analyze_fixed_errors(baseline_preds, debiased_preds)
    overlap_results = analyze_by_word_overlap(baseline_preds, debiased_preds)
    analyze_by_negation(baseline_preds, debiased_preds)
    
    # Create visualizations
    create_comparison_visualizations(baseline_preds, debiased_preds, overlap_results, args.output_dir)
    
    # Save detailed comparison
    comparison_report = {
        'overall': {
            'baseline_accuracy': float(baseline_acc),
            'debiased_accuracy': float(debiased_acc),
            'improvement': float(debiased_acc - baseline_acc),
            'examples_fixed': len(fixed_examples),
            'new_errors': len(new_errors),
            'net_improvement': len(fixed_examples) - len(new_errors)
        },
        'overlap_analysis': overlap_results
    }
    
    output_path = Path(args.output_dir) / 'comparison_report.json'
    with open(output_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\n✓ Saved comparison report: {output_path}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nKey takeaways for your report:")
    print(f"1. Overall improvement: {(debiased_acc - baseline_acc)*100:+.2f}%")
    print(f"2. Examples fixed: {len(fixed_examples)}")
    print(f"3. Check overlap analysis for where improvements occur")
    print(f"4. Review visualizations in '{args.output_dir}'")


if __name__ == '__main__':
    main()
