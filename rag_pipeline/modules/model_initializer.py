from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    AutoTokenizer, pipeline, BitsAndBytesConfig
)
import torch
import logging

class ModelInitializer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize_summarizer(self):
        model_id = self.config['models']['summarization']['model_id']
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        self.logger.info("Initialized summarization model")
        return summarizer

    def initialize_conversation_model(self):
        model_id = self.config['models']['conversation']['model_id']
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
        conversation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config['models']['conversation']['max_new_tokens'],
            temperature=self.config['models']['conversation']['temperature'],
            top_k=self.config['models']['conversation']['top_k']
        )
        self.logger.info("Initialized conversation model")
        return conversation_pipe
