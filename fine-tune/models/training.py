from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizer

def load_model(tokenizer: PreTrainedTokenizer, model_cfg):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.model_name,
        max_seq_length=model_cfg.max_seq_length,
        dtype=model_cfg.dtype,
        load_in_4bit=model_cfg.load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # These could also be parameters in your config
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

    return model, tokenizer
