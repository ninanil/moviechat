import evaluate
import numpy as np
import mlflow

# Initialize evaluation metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def compute_metrics(eval_preds, tokenizer):
    predictions, labels = eval_preds

    # Handle different output formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits to predicted token IDs if necessary
    if predictions.ndim == 3:
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions

    # Replace -100 in labels with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up decoded texts
    decoded_predictions = [pred.strip() for pred in decoded_predictions]
    decoded_references = [ref.strip() for ref in decoded_references]

    # Prepare references for BLEU
    formatted_references = [[ref] for ref in decoded_references]

    # Compute BLEU metric
    bleu_results = bleu_metric.compute(predictions=decoded_predictions, references=formatted_references)

    # Compute ROUGE metric
    rouge_results = rouge_metric.compute(predictions=decoded_predictions, references=decoded_references)

    # Compute BERTScore metric
    bertscore_results = bertscore_metric.compute(
        predictions=decoded_predictions, references=decoded_references, lang="en"
    )

    # Log metrics to MLflow
    try:
        mlflow.log_metric("bleu", bleu_results["bleu"])
        for key, value in rouge_results.items():
            mlflow.log_metric(f"rouge_{key}", value)
        mlflow.log_metric("bertscore_precision", np.mean(bertscore_results["precision"]))
        mlflow.log_metric("bertscore_recall", np.mean(bertscore_results["recall"]))
        mlflow.log_metric("bertscore_f1", np.mean(bertscore_results["f1"]))
    except Exception as e:
        print(f"Error logging metrics to MLflow: {e}")

    # Prepare metrics to return
    metrics = {
        "bleu": bleu_results["bleu"],
        "bertscore_precision": np.mean(bertscore_results["precision"]),
        "bertscore_recall": np.mean(bertscore_results["recall"]),
        "bertscore_f1": np.mean(bertscore_results["f1"]),
    }
    metrics.update({f"rouge_{key}": value for key, value in rouge_results.items()})

    return metrics
