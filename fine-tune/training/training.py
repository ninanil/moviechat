import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import mlflow
from transformers import TrainingArguments
from trl import SFTTrainer
from data.datasets import load_datasets
from models.model import load_model
from utils.callbacks import SaveAllCheckpointsCallback
from utils.metrics import compute_metrics
from utils.login_scripts import login_huggingface, login_wandb
from transformers import AutoTokenizer

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Log in to external services
    login_huggingface(os.environ.get('HUGGINGFACE_TOKEN'))
    run = login_wandb(
        key=os.environ.get('WANDB_KEY'),
        project=cfg.wandb.project_name,
        entity=cfg.wandb.entity
    )

    # Initialize MLflow and DagsHub
    import dagshub
    dagshub.init(
        repo_owner=cfg.dagshub.repo_owner,
        repo_name=cfg.dagshub.repo_name,
        mlflow=cfg.dagshub.mlflow
    )
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model, tokenizer = load_model(tokenizer, cfg.model)

    # Load datasets
    artifact = run.use_artifact(cfg.data.dataset_artifact, type='dataset')
    artifact_dir = artifact.download()
    train_dataset, val_dataset = load_datasets(artifact_dir, tokenizer)

    # Prepare training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        logging_steps=cfg.training.logging_steps,
        optim=cfg.training.optim,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        seed=cfg.training.seed,
        output_dir=cfg.training.output_dir,
        report_to=cfg.training.report_to,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=2,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        packing=False,
        args=training_args
    )

    # Add callbacks
    trainer.add_callback(
        SaveAllCheckpointsCallback(
            save_dir=training_args.output_dir,
            save_steps=training_args.save_steps
        )
    )

    # Start training within an MLflow run
    with mlflow.start_run():
        trainer_stats = trainer.train()

    # Log training statistics
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
    print(f"Peak memory usage: {torch.cuda.max_memory_reserved() / 1e9} GB.")

if __name__ == "__main__":
    main()
