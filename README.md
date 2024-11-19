
# MovieChat: A Comprehensive Framework for Movie-Related Conversational AI

MovieChat is a project aimed at developing a movie-focused conversational AI by combining effective data preparation techniques with advanced fine-tuning strategies for Large Language Models (LLMs). This framework is designed to preprocess diverse datasets and fine-tune LLaMA2 models using QLoRA and Unsloth, enabling them to generate movie-related dialogues and answers with high efficiency and accuracy.


## Overview

### Data Preparation

The project integrates datasets like **[Cornell Movie Dialogues](https://convokit.cornell.edu/documentation/movie.html)** and **[MovieQA](https://github.com/makarandtapaswi/MovieQA_benchmark?tab=readme-ov-file)**. It preprocesses the data, enriching it with metadata from IMDb and tracking changes using **Weights & Biases (W&B)** and the **Hugging Face Hub**.

**Key Highlights**:

- Seamless integration with IMDb for metadata.  
- Modular YAML-based configuration using Hydra.  
- Data versioning and artifact management with W&B and DVC.  

For more details, refer to the [Data Preparation README](https://github.com/ninanil/moviechat/blob/master/data_preparation/README.md).  

## Fine-Tuning

The fine-tuning pipeline leverages **QLoRA** and **Unsloth** to efficiently adapt **LLaMA2** to movie-centric tasks. Training workflows include dynamic configuration, real-time logging, and version-controlled checkpoints using **DagsHub** and **MLflow**.

**Key Highlights**:

- Efficient low-memory optimization via QLoRA.  
- Comprehensive tracking and logging with DagsHub, MLflow, and W&B.  
- Scalable training with mixed precision (FP16/BF16).  

For more details, refer to the [Fine-Tuning README](https://github.com/ninanil/moviechat/blob/master/fine-tune/README.md).  

## Workflows

### Data Preparation
   1. Load datasets like Cornell Movie Dialogues and MovieQA.  
   2. Preprocess data with IMDb metadata.  
   3. Save datasets locally and upload them to the Hugging Face Hub.  

### Model Fine-Tuning
   1. Configure training parameters using Hydra.  
   2. Fine-tune LLaMA2 with QLoRA optimization.  
   3. Track experiments with DagsHub, MLflow, and W&B.  
   4. Save checkpoints for easy reproducibility.  


