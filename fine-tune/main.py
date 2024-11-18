
!pip install --upgrade pip


!pip-autoremove torch torchvision torchaudio -y
!pip install unsloth
!pip install wandb
!pip install dvc
!pip install boto3 --upgrade

# Uninstall the problematic packages
!pip uninstall -y cffi pycparser

# Install compatible versions
!pip install cffi==1.16.0 pycparser==2.20




!pip install evaluate
!pip install rouge_score bert_score


!nvidia-smi



from huggingface_hub import login
login('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')


import wandb
wandb.login(key = '2e7ba1c4f4eb4c439719e22f2276e30f13c4c7b5')

run = wandb.init(project= 'Moviechat', entity='niloufarcolab6-n')
# run = wandb.init(project= cfg.wandb.project_name, entity=cfg.wandb.entity)


import os
os.environ["WANDB_DISABLED"] = "true"



!pip install mlflow
!pip install dagshub


import dagshub, mlflow

dagshub.init(repo_owner='ninanildev', repo_name='moviechat', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ninanildev/moviechat.mlflow")



!python --version


from unsloth import FastLanguageModel
import torch
max_seq_length = 1024#2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-bnb-4bit", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
#     device_map='auto',  
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


print(tokenizer)

# %% [markdown] {"id":"SXd9bTZd1aaL"}
# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
#     prompt = """### Instruction:{}
# ### Input:{}
# ### Context:{}
# """
    prompt = """Instruction:{}
Input:{}
Context:{}
"""
    # Extract fields from examples
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]  
    contexts = examples["context"]
    split = examples["split"]
    texts = []  # List to store formatted prompts

    # Iterate through the data and format each example
    for instruction, input, context, output in zip(instructions, inputs, contexts, outputs):
        # Initialize the context string
        context_str = ""
        if context and isinstance(context, dict):
            # Extract specific context fields
            movie_name = context.get('movie_name', 'Unknown')
            genre = context.get('genre', 'Unknown')
            year = context.get('year', 'Unknown')
            plot_outline = context.get('plot_outline', 'Unknown')
            context_str += f"Movie: {movie_name}\nGenre: {genre}\nYear: {year}\nPlot Outline: {plot_outline}"
            # Handle NaN values explicitly for plot_outline and other fields
            if isinstance(plot_outline, float) and math.isnan(plot_outline):
                plot_outline = 'Unknown'
            
            # Additional context fields
            if 'character_names' in context:
                character_names = context.get('character_names', 'Unknown')
                context_str += f"Character names: {character_names}\n"
            if 'imdb_rating' in context:
                imdb_rating = context.get('imdb_rating', 'Unknown')
                context_str += f"IMDb rating: {imdb_rating}\n"
            if 'num_imdb_votes' in context:
                num_imdb_votes = context.get('num_imdb_votes', 'Unknown')
                context_str += f"Number of IMDb votes: {num_imdb_votes}\n"
#             if split == 'train':
            text = prompt.format(instruction, input, context_str,'\nResponse:', output) + EOS_TOKEN
#             else:
#                 text = prompt.format(instruction, input, context_str) + EOS_TOKEN
        # Format the prompt with instruction, input, context, and response
        
        texts.append(text)

    # Return the formatted text as a dictionary to be used with map function
    return {"text": texts}




# dataset_url = 'https://huggingface.co/datasets/niloufarna/MovieChat/resolve/main/dataset/combined_movie_dataset.json'


artifact = run.use_artifact('niloufarcolab6-n/Moviechat/combined_movie_dataset:v0', type='dataset')
artifact_dir = artifact.download()

# %% [code]



from datasets import load_dataset

# Load the entire dataset
# dataset = load_dataset('json', data_files=dataset_url)
dataset = load_dataset('json', data_dir=artifact_dir)
# Filter each split based on the 'split' field
train_dataset = dataset['train'].filter(lambda x: x['split'] == 'train')
test_dataset = dataset['train'].filter(lambda x: x['split'] == 'test')
val_dataset = dataset['train'].filter(lambda x: x['split'] == 'val')

# Example usage
print("Train Dataset:", train_dataset)
print("Test Dataset:", test_dataset)
print("Validation Dataset:", val_dataset)



train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
val_dataset = val_dataset.map(formatting_prompts_func, batched = True)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)


val_dataset


train_dataset = train_dataset.remove_columns(['split', 'type', 'instruction', 'input', 'context', 'response'])
val_dataset = val_dataset.remove_columns(['split', 'type', 'instruction', 'input', 'context','response'])
test_dataset = test_dataset.remove_columns(['split', 'type', 'instruction', 'input', 'context','response']).to_pandas()



test_dataset


for idx in range(3,6):  # Print a few samples
    sample_text = val_dataset[idx]["text"]
    tokenized_sample = tokenizer(sample_text, truncation=True, max_length=max_seq_length)
    print(f"Original text: {sample_text}")
    print(f"Tokenized input IDs: {tokenized_sample['input_ids']}")



import numpy as np
def inspect_logits(eval_preds):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract logits if predictions is a tuple
        print("isinstance(predictions, tuple)")

    if predictions.ndim == 3:  # This means logits are of shape (batch_size, seq_length, vocab_size)
        print("Inspecting logits for the first few validation samples:")
        for i in range(min(2, predictions.shape[0])):  # Loop over the first two samples in the batch
            logits = predictions[i]
            print(f"Logits for validation sample {i}:")
            print(logits)  # Logits have shape (seq_length, vocab_size)

            max_logits = np.max(logits, axis=-1)  # Find the maximum logits across the vocab axis
            print(f"Maximum logits along sequence positions for sample {i}: {max_logits}")

            # Check if the logits are very low or mostly the same
            if np.allclose(max_logits, 0, atol=1e-3):
                print(f"All logits for sample {i} are close to zero.")
            elif np.allclose(max_logits, max_logits[0]):
                print(f"All logits for sample {i} are approximately the same, which may indicate an issue.")


def inspect_predictions(predictions):
    # Convert logits to token IDs if necessary
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    if predictions.ndim == 3:  # Convert logits to predicted token IDs
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions

    # Print predicted token IDs for the first few validation samples
    print("Inspecting predicted token IDs for the first few validation samples:")
    for i in range(min(2, len(predicted_ids))):
        print(f"Predicted token IDs for sample {i}: {predicted_ids[i]}")

        # Check if the prediction is empty (i.e., all tokens are pad tokens)
        if np.all(predicted_ids[i] == tokenizer.pad_token_id):
            print(f"Prediction {i} contains only pad tokens, indicating an empty prediction.")



import evaluate
import numpy as np
import torch
import mlflow

# Initialize evaluation metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def compute_metrics(eval_preds):
    print("in compute_metrics")
#     inspect_logits(eval_preds)
    predictions, labels = eval_preds
#     inspect_predictions(predictions)
    # Handle different output formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits to predicted token IDs if necessary
    if predictions.ndim == 3:
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions

    # Replace -100 in labels with pad_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is not set. Please ensure tokenizer.pad_token_id is defined.")
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up decoded texts
    decoded_predictions = [pred.strip() for pred in decoded_predictions]
    decoded_references = [ref.strip() for ref in decoded_references]

    # Enhanced check for empty predictions or references
    empty_predictions_indices = [i for i, pred in enumerate(decoded_predictions) if not pred]
    empty_references_indices = [i for i, ref in enumerate(decoded_references) if not ref]

    if empty_predictions_indices or empty_references_indices:
        print("Warning: Empty predictions or references found.")
        if empty_predictions_indices:
            print(f"Empty predictions at indices: {empty_predictions_indices}")
        if empty_references_indices:
            print(f"Empty references at indices: {empty_references_indices}")
        # Optionally, you can handle this case differently instead of raising an error
        # For example, setting empty predictions/references to a default value like "[EMPTY]"
        # decoded_predictions = ["[EMPTY]" if i in empty_predictions_indices else pred for i, pred in enumerate(decoded_predictions)]
        # decoded_references = ["[EMPTY]" if i in empty_references_indices else ref for i, ref in enumerate(decoded_references)]
#         for i in range(len(decoded_predictions)):
#             print(f"prediction at index {i}: Original text = {decoded_predictions[i]}")
        for i in range(len(decoded_predictions)):
            print(f"reference at index {i}: Original text = {decoded_references[i]}")

    # Prepare references for BLEU
    formatted_references = [[ref] for ref in decoded_references]

    # Compute BLEU metric
    bleu_results = bleu_metric.compute(predictions=decoded_predictions, references=formatted_references)

    # Compute ROUGE metric
    rouge_results = rouge_metric.compute(predictions=decoded_predictions, references=decoded_references)

    if all(decoded_predictions) and all(decoded_references):
        bertscore_results = bertscore_metric.compute(predictions=decoded_predictions, references=decoded_references, lang="en")
    else:
        print("Skipping BERTScore due to empty predictions or references.")
        bertscore_results = {"precision": [0], "recall": [0], "f1": [0]}


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



import subprocess
from transformers import TrainerCallback
class SaveAllCheckpointsCallback(TrainerCallback):
    def __init__(self, save_dir, save_steps=50):
        self.save_dir = save_dir
        self.save_steps = save_steps
#     def on_evaluate(self, args, state, control, **kwargs):
#         print("on_evaluate")
#         print("kwargs in on_evaluate\n", kwargs)
#         logits = kwargs['metrics']['eval_logits']
#         # Log the first logits tensor
#         print(f"Logits for the first sample: {logits[0]}")
    def on_step_end(self, args, state, control, **kwargs):
        print("step ", state.global_step)
        for name, param in kwargs['model'].named_parameters():
            print(f"param Name {name} {param.requires_grad} {param.grad==None}")
            if param.grad is not None:
                max_grad = param.grad.abs().max().item()
                print(f"Max gradient for {name} at step {state.global_step}: {max_grad}")

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            self._save_full_checkpoint(state.global_step)

    def _save_full_checkpoint(self, step):
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save checkpoint using DVC
        try:
            result = subprocess.run(['dvc', 'add', checkpoint_dir], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully added {checkpoint_dir} to DVC.")
            else:
                print(f"Error adding {checkpoint_dir} to DVC: {result.stderr}")
                subprocess.run(["dvc", "push"], check=True, cwd='.')
                print(f"Full training checkpoint saved and pushed to DagsHub: {checkpoint_dir}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
        per_device_train_batch_size = 4,#2,
        per_device_eval_batch_size = 2,
#         use_cpu = True,
        eval_accumulation_steps = 2,
#         eval_delay = 5,
        gradient_accumulation_steps = 8,#4,
        warmup_steps = 191,
        eval_strategy ='steps',
        max_grad_norm=1.0,
        eval_steps=3,
        save_steps=3,
        save_strategy='steps',
        save_total_limit=3,
#         max_steps = 60,
        num_train_epochs = 1,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 3,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine_with_restarts",#"linear",
        seed = 3407,
        output_dir = "outputs",
#         remove_unused_columns=False,
        report_to= 'mlflow',#'all'#"wandb"
    )
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    compute_metrics = compute_metrics,
#     data_collator = data_collator,
#     callbacks = [MlflowEvaluationCallback(val_dataset_df, tokenizer,15)],
    packing = False, # Can make training 5x faster for short sequences. 
    args = training_args
)


trainer.add_callback(SaveAllCheckpointsCallback(save_dir=training_args.output_dir,save_steps = training_args.save_steps ))


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Start training within an MLflow run
with mlflow.start_run():
    trainer_stats = trainer.train()


# %% [code]



import os
import torch
import mlflow
import dvc.api
from dvc.repo import Repo
from transformers import Trainer, TrainingArguments
from pathlib import Path

# Assuming your model is a PyTorch model
class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch):
        # Save the model state and optimizer state
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        # Log to MLflow
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        # Load the checkpoint to resume training
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']



# Initialize CheckpointManager
checkpoint_manager = CheckpointManager()

# Example TrainingArguments for HuggingFace Trainer
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=4,
    logging_steps=10,
    save_steps=50,  # Save checkpoints every 50 steps
    save_total_limit=3,  # Keep only the last 3 checkpoints
    load_best_model_at_end=True
)



# Start MLflow run
with mlflow.start_run() as run:
    for epoch in range(num_epochs):
        # Training loop - update model
        trainer.train()

        # Save checkpoint after each epoch
        checkpoint_manager.save_checkpoint(trainer.model, trainer.optimizer, epoch)

        # Use DVC API to add and push checkpoints to remote
        repo = Repo(".")
        repo.add("checkpoints")
        repo.push(remote="origin")

        # Log checkpoints to DagsHub via MLflow
        mlflow.log_artifact(checkpoint_manager.checkpoint_dir, artifact_path="checkpoints")



#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


