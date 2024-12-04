
# RAG-MovieChat

A modular Retrieval-Augmented Generation (RAG) system for answering movie-related questions using Chroma, LangChain, and Hugging Face models.


## Features

- **Dynamic Retrieval**: Retrieves movie names first and uses the context to generate precise answers.
- **Two Specialized Retrievers**:
  - **MovieNameRetriever**: Focuses on extracting the most relevant movie name.
  - **MovieInfoRetriever**: Fetches detailed movie information based on context.
- **Summarization**: Summarizes conversation history to maintain context in long interactions.
- **Chroma Vector Stores**: Efficiently stores and retrieves documents with embeddings.
- **Hydra Configuration**: Flexible YAML-based configuration for easy customization.
- **Interactive Conversation Chain**: Handles user questions with memory and context-awareness.

## Technologies Used

- **LangChain**: For chaining LLM-based retrieval and conversational models.
- **Chroma**: Vector store for efficient document management.
- **Hugging Face Transformers**: Models for summarization and conversational pipelines.
- **Hydra**: For dynamic configuration management.
- **Python**: The backbone of the application.

---


## Project Structure

```
project/
│
├── config/                        # Hydra configuration files
│   ├── conversation_chain.yaml    # Configuration for the conversation pipeline
│   ├── hydra.yaml                 # Hydra-specific settings (e.g., logging, defaults)
│   ├── main.yaml                  # Main entry configuration file
│   ├── memory_manager.yaml        # Configuration for conversation memory
│   ├── model_initializer.yaml     # Configuration for Hugging Face models
│   ├── prompt_templates.yaml      # Configuration for dynamic LLM prompts
│   ├── retriever.yaml             # Settings for movie name/info retrievers
│   └── vector_store.yaml          # Configuration for vector store (Chroma)
│
├── modules/                       # Core application modules
│   ├── data_loader.py             # Handles data loading logic
│   ├── data_preprocessor.py       # Manages data cleaning and preprocessing
│   ├── vector_store.py            # Handles Chroma vector store management
│   ├── retriever.py               # Initializes retrievers for movies
│   ├── model_initializer.py       # Loads Hugging Face models
│   ├── prompt_templates.py        # Manages templates for LLM prompts
│   ├── memory_manager.py          # Manages memory for conversation context
│   └── conversation_chain.py      # Handles the pipeline for interactive user conversations

```
# Important Classes

## `VectorStoreInitializer` (Abstract Base Class)
Manages chroma vector stores for storing and retrieving document embeddings. Specialized child classes:

- `MovieNameVectorStore`: stores movie names and metadata.
- `MovieInfoVectorStore`: stores detailed movie information like plots and synopses.

## `Retriever` (Abstract Base Class)
Defines the interface for initializing retrievers. Specialized child classes:

- `MovieNameRetriever`: extracts relevant movie names from user input.
- `MovieInfoRetriever`: fetches detailed movie information using configurable filters.

---

## `ConversationChain`
The core class managing user interactions. It:

1. summarizes chat history for concise context.
2. retrieves relevant movie information using retrievers.
3. generates responses using hugging face conversational models.

## `PromptTemplateManager`
Manages dynamic prompt templates for interacting with llms, ensuring responses are contextually accurate.
