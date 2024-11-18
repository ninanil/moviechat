from huggingface_hub import login
import wandb

def login_huggingface(token):
    if token is None:
        raise ValueError("Hugging Face token is not set.")
    login(token)

def login_wandb(key, project, entity):
    if key is None:
        raise ValueError("Weights & Biases API key is not set.")
    wandb.login(key=key)
    run = wandb.init(project=project, entity=entity)
    return run
