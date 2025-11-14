import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, HfArgumentParser
from helpers import compute_accuracy
import os
import json

NUM_PREPROCESSING_WORKERS = 2


# Modified preprocessing function that ONLY uses hypothesis (ignores premise)
def prepare_dataset_nli_hypothesis_only(examples, tokenizer, max_seq_length=None):
    """
    This function only tokenizes the hypothesis, completely ignoring the premise.
    This will reveal if the model can predict labels based on hypothesis alone (dataset artifacts).
    """
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    # Only tokenize hypothesis - treat it as if it's a single sentence classification task
    tokenized_examples = tokenizer(
        examples['hypothesis'],  # ONLY hypothesis, no premise!
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help='Base model to fine-tune')
    argp.add_argument('--dataset', type=str, default='snli',
                      help='Dataset to use (default: snli)')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on')

    training_args, args = argp.parse_args_into_dataclasses()

    print("=" * 50)
    print("HYPOTHESIS-ONLY MODEL")
    print("This model only sees the hypothesis, NOT the premise!")
    print("High accuracy indicates dataset artifacts.")
    print("=" * 50)

    # Load dataset
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        dataset_id = tuple(args.dataset.split(':')) if ':' in args.dataset else (args.dataset,)
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI has 3 labels: 0=entailment, 1=neutral, 2=contradiction
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    
    # Fix tensor contiguity issue for ELECTRA
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Use our hypothesis-only preprocessing function
    prepare_train_dataset = prepare_eval_dataset = \
        lambda exs: prepare_dataset_nli_hypothesis_only(exs, tokenizer, args.max_length)

    print("Preprocessing data (hypothesis-only)...")
    
    # Remove SNLI examples with no label
    if args.dataset == 'snli':
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Store predictions for later analysis
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
    
    # Train and/or evaluate
    if training_args.do_train:
        print("\nTraining hypothesis-only model...")
        trainer.train()
        trainer.save_model()
        print("Model saved!")

    if training_args.do_eval:
        print("\nEvaluating hypothesis-only model...")
        results = trainer.evaluate()

        print('\n' + "=" * 50)
        print('HYPOTHESIS-ONLY EVALUATION RESULTS:')
        print("=" * 50)
        print(results)
        print("=" * 50)
        
        # Calculate accuracy compared to random baseline
        random_baseline = 1.0 / 3.0  # 33.3% for 3-class classification
        print(f"\nRandom baseline: {random_baseline:.1%}")
        print(f"Hypothesis-only: {results['eval_accuracy']:.1%}")
        print(f"Improvement over random: {results['eval_accuracy'] - random_baseline:.1%}")
        
        if results['eval_accuracy'] > 0.5:
            print("\n⚠️  WARNING: Hypothesis-only model achieves >50% accuracy!")
            print("This indicates STRONG dataset artifacts in the hypotheses.")
        elif results['eval_accuracy'] > 0.4:
            print("\n⚠️  Hypothesis-only model achieves >40% accuracy.")
            print("This indicates moderate dataset artifacts.")
        
        print("=" * 50)

        # Save results
        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 
                  encoding='utf-8', mode='w') as f:
            json.dump(results, f, indent=2)

        # Save predictions with examples for analysis
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), 
                  encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                example_with_prediction['correct'] = (example_with_prediction['predicted_label'] == example['label'])
                f.write(json.dumps(example_with_prediction))
                f.write('\n')
        
        print(f"\nPredictions saved to: {training_args.output_dir}/eval_predictions.jsonl")
        print("You can analyze these to see what patterns the hypothesis-only model learned!")


if __name__ == "__main__":
    main()
