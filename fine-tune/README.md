
# MovieChat: Fine-Tuning LLaMA2 with QLoRA and Unsloth

MovieChat is a powerful project designed to fine-tune the LLaMA2 language model using QLoRA and [Unsloth](https://unsloth.ai/), enabling efficient and effective customization for movie-related conversational AI applications. This project leverages advanced tools like Hydra for configuration management, DagsHub and MLflow for experiment tracking, and [Weights & Biases](https://wandb.ai/site/) (W&B) for comprehensive monitoring.


## Table of Contents
- [Title and Description](#title-and-description)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Project Structure](#project-structure)
- [Install and Run](#instal-and-run)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Outputs](#outputs)
## Features

- **Efficient Fine-Tuning**: Utilize QLoRA and Unsloth to fine-tune the LLaMA2 model with reduced memory usage.  
- **Modular Configuration**: Manage configurations effortlessly using Hydra, allowing dynamic composition and overrides.  
- **Comprehensive Tracking**: Monitor experiments and metrics with DagsHub and MLflow, alongside real-time tracking via Weights & Biases.  
- **Version Control with DVC**: Handle large datasets and model checkpoints seamlessly using Data Version Control (DVC).  
- **Scalable Training**: Optimize training with gradient accumulation and mixed precision (FP16/BF16) support.  
- **LoRA Adapters**: Implement Low-Rank Adaptation (LoRA) for efficient parameter updates.  

## Project Structure

```
your_project/
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── data_config.yaml
│   ├── dagshub/
│   │   └── dagshub_config.yaml
│   ├── model/
│   │   └── model_config.yaml
│   ├── training/
│   │   └── training_config.yaml
├── data/
│   └── datasets.py
├── models/
│   └── model.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py
├── utils/
│   ├── callbacks.py
│   ├── metrics.py
│   ├── login_scripts.py
│   └── __init__.py
├── scripts/
│   └── install_packages.sh
├── requirements.txt
├── README.md
└── .gitignore
```
## Install and Run

### Prerequisites

- **Python 3.8+**   
- **CUDA-enabled GPU** (for training)  
- **DVC** installed for data versioning  

### Clone the Repository

```
git clone https://github.com/ninanil/moviechat.git
cd moviechat
```
### Set Up Environment Variables

for security, sensitive information like api keys should be managed via environment variables. create a `.env` file in the root directory and add your credentials:

```bash
HUGGINGFACE_TOKEN=your_huggingface_token
WANDB_KEY=your_wandb_api_key
```
**Note**: ensure that `.env` is included in your `.gitignore` to prevent accidental commits.

### Install Dependencies

1. **Make the installation script executable**:

```bash
pip install -r requirements.txt
```
### Running the Training Script
```
python training/train.py
```
## Configuration

### Using Hydra

Hydra is used for managing configurations, allowing you to override settings dynamically via command-line arguments or configuration files.

### Configuration Files

All configurations are located in the `configs/` directory, structured as follows:

- **configs/config.yaml**: Main configuration file with default settings.  
- **configs/data/data_config.yaml**: Data-related configurations.  
- **configs/dagshub/dagshub_config.yaml**: DagsHub and MLflow configurations.  
- **configs/model/model_config.yaml**: Model-specific settings.  
- **configs/training/training_config.yaml**: Training parameters.  

## Model Details
### LLaMA2 Fine-Tuning with Unsloth
The project leverages the LLaMA2 model, fine-tuned using [Unsloth](https://unsloth.ai/), which facilitates efficient loading and manipulation of large language models.
### QLoRA Optimization

**QLoRA** (Quantized Low-Rank Adaptation) is employed to optimize the fine-tuning process, significantly reducing memory usage while maintaining model performance.

### LoRA Adapters:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```
## Outputs

### Checkpoints

Model checkpoints are saved at specified intervals during training. these checkpoints are managed using dvc and pushed to dagshub for versioning and easy retrieval.

**Checkpoint Directory**:  
All checkpoints are stored in the `outputs/` directory, organized by timestamp.

### Metrics and Logs

Training metrics such as loss, BLEU, ROUGE, BERTScore are logged to mlflow and visualized in w&b. These logs provide insights into model performance and training progress.
