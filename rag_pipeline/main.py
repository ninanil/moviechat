import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from modules.data_loader import DataLoader
from modules.data_preprocessor import DataPreprocessor
from modules.vector_store import MovieNameVectorStore, MovieInfoVectorStore
from modules.model_initializer import ModelInitializer
from modules.retriever import MovieNameRetriever, MovieInfoRetriever
from modules.prompt_templates import PromptTemplateManager
from modules.memory_manager import MemoryManager
from modules.conversation_chain import ConversationChain
from huggingface_hub import login
import sys

@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    # Setup logging
    logging.basicConfig(level=cfg.logging.level)
    logger = logging.getLogger(__name__)
    logger.info("Starting application")

    # Display the configuration (optional)
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate Hugging Face API Key
    if cfg.huggingface.api_key == "YOUR_HF_API_KEY":
        logger.error("Please set your Hugging Face API key in config/main.yaml")
        sys.exit(1)

    # Login to Hugging Face Hub
    login(cfg.huggingface.api_key)
    logger.info("Logged in to Hugging Face")

    # Data Loading
    data_loader = DataLoader(cfg.data_loader)
    datasets = data_loader.get_datasets()

    # Data Preprocessing
    preprocessor = DataPreprocessor(cfg.data_preprocessor)
    combined_dataset = preprocessor.preprocess_datasets(datasets)

    # Initialize Vector Stores and add documents
    movie_name_vector_store = MovieNameVectorStore(cfg.vector_store.movie_vector_store)
    movie_name_vector_store.add_documents(combined_dataset)

    movie_info_vector_store = MovieInfoVectorStore(cfg.vector_store.movie_info_vector_store)
    movie_info_vector_store.add_documents(combined_dataset)

    # Initialize Models
    model_initializer = ModelInitializer(cfg.model_initializer)
    summarizer = model_initializer.initialize_summarizer()
    conversation_pipe = model_initializer.initialize_conversation_model()

    # Initialize Retrievers
    # Initialize MovieNameRetriever with weights from config
    movie_name_retriever = MovieNameRetriever(
        config=cfg.retriever.movie_name_retriever,
        chroma_index=movie_name_vector_store.chroma_index
    )
    movie_name_ensemble_retriever = movie_name_retriever.initialize_retriever()

    # Initialize MovieInfoRetriever with weights from config
    movie_info_retriever = MovieInfoRetriever(
        config=cfg.retriever.movie_info_retriever,
        chroma_index=movie_info_vector_store.chroma_index
    )
    ensemble_retriever = movie_info_retriever.initialize_retriever()

    # Initialize Prompt Templates
    prompt_manager = PromptTemplateManager()
    chat_prompt_template = prompt_manager.get_chat_prompt_template(cfg.prompt_templates.instruction)

    # Initialize Memory
    memory_manager = MemoryManager(cfg.memory_manager)
    memory = memory_manager.initialize_memory()

    # Create Conversation Chain
    conversation_chain = ConversationChain(
        memory=memory,
        summarizer=summarizer,
        ensemble_retriever=ensemble_retriever,
        movie_name_retriever=movie_name_ensemble_retriever,
        prompt_template=chat_prompt_template,
        conversation_pipe=conversation_pipe
    )

    # Interactive Chat Loop
    user_input = input("User: ")
    while user_input.lower() != 'exit':
        response = conversation_chain.handle_conversation({
            'instruction': cfg.prompt_templates.instruction,
            'question': user_input
        })
        print(f"Assistant: {response}")
        user_input = input("User: ")

if __name__ == '__main__':
    main()
