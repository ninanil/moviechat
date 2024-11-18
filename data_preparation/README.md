
# Data Preparation for Movie Datasets
This project handles data preparation for movie-related datasets such as [**Cornell Movie Dialogues**](https://convokit.cornell.edu/documentation/movie.html) and [**MovieQA**](https://github.com/makarandtapaswi/MovieQA_benchmark?tab=readme-ov-file). It includes functionality for loading, preprocessing, and saving datasets while integrating with external services like **Hugging Face** and **Weights & Biases**. The processed datasets are designed to serve as input for fine-tuning a **Large Language Model (LLM)**, enabling it to better understand and generate movie-related conversations and answers.


##  Table Of Contents
- [Title and Description](#title-and-description)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Example](#example)
- [Design Patterns Used](#design-patterns-used)

## Features

- **flexible configuration**:  
  - hydra integration for yaml-based configuration ensures seamless handling of diverse datasets with distinct preprocessing requirements.  

- **dataset handling**:  
  - loads **cornell movie dialogues**: focused on movie-based conversational exchanges, providing rich, natural dialogue examples for training.  
  - loads **movieqa**: designed for question-answering tasks with structured metadata like plot summaries, genres, and release years.  
  - fetches additional movie metadata from imdb using `IMDbPY` to enhance both datasets with information like imdb ratings, genres, and plot outlines.  

- **why diversity matters**:  
  - a diverse and comprehensive dataset is crucial to improving the model's understanding and generation capabilities.  
  - by exposing the model to varied contexts (e.g., conversational and structured data), it can perform better in generating contextually accurate responses.  

- **Integration**:
  - Supports logging and experiment tracking via Weights & Biases (wandb).
  - wandb is also used for dataset versioning.
  - Uploads processed datasets to Hugging Face's Hub.

## Setup Instructions
### 1. Install Dependencies
Install all the required libraries by running:

`pip install -r requirements.txt`

### 2. Set Up Environment Variables
Create a .env file in the root directory and add the following keys:
```
HF_API_TOKEN=your_huggingface_token
WANDB_API_TOKEN=your_wandb_token
```
### 3. Run the Script
Start the data preparation process by running:

`python main.py`

## Project Structure
```
data_preparation/
├── config/
│   ├── class_configs/
│   │   ├── __init__.py
│   │   ├── dataset_config.py         # Configuration classes for datasets
│   ├── config_loader/
│       ├── __init__.py
│       ├── config_loader.py          # Loads configuration objects
│
├── data_fetchers/
│   ├── __init__.py
│   ├── movie_fetcher.py              # Abstract base class for fetching movies
│   ├── cornell_movie_fetcher.py      # Handles Cornell Movie Dialogue data fetching
│   ├── movieqa_fetcher.py            # Handles MovieQA data fetching
│
├── data_loaders/
│   ├── __init__.py
│   ├── base_dataloader.py            # Abstract class for loading datasets
│   ├── cornell_loader.py             # DataLoader for Cornell Movie Dialogues
│   ├── movieqa_loader.py             # DataLoader for MovieQA
│   ├── config_loader.py              # Helper to load dataset configurations
│   ├── data_loader_factory.py        # Factory pattern for creating DataLoaders
│
├── datasets/                         # Directory for storing dataset files
│
├── service/
│   ├── __init__.py
│   ├── service_locator.py            # Implements Service Locator pattern
│   ├── hydra_config_locator.py       # Centralized access for Hydra configuration
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                     # Shared logger setup
│   ├── singleton_initializer.py      # Singleton for initializing configurations
├── main.py                           # Entry point for running the script

```
### Configuration 
this project uses hydra to manage flexible and modular configurations through yaml files. the following configuration files are integral to the project:

1. `config.yaml`:  
   this is the root configuration file that uses hydra's `defaults` mechanism to include other configuration files.  

2. `dataset.yaml`:  
   defines configurations for both the cornell movie dialogues and movieqa datasets, specifying file paths, columns, and preprocessing settings.  

3. `hf_config.yaml`:  
   handles integration with the hugging face hub, defining repository ids, file paths, and metadata for uploading datasets.  

4. `wandb_config.yaml`:  
   defines configurations for weights & biases (w&b) integration, including artifact names, descriptions, and project metadata for dataset versioning and logging.  

### Data Loader Overview
The `CornellDataLoader` is a concrete implementation of the `BaseDataLoader` abstract class. It is responsible for managing the entire lifecycle of the Cornell Movie dataset, including data loading, preprocessing, merging, and splitting.

#### **workflow:**

1. **loading**: raw data files are read into dataframes.  
2. **preprocessing**: each dataframe is cleaned, parsed, and standardized.  
3. **merging**: dataframes are combined into a single dataset with all relevant information.  
4. **splitting**: the merged dataset is split into training, validation, and testing sets.  
5. **saving**: the final dataset is saved locally and uploaded to external platforms.  
#### **integration with imdb and external services**

- **imdb metadata**: enriches the dataset with metadata like plot outlines, synapsis, genres, and imdb ratings using the `IMDbPY` library.  
- **weights & biases**: tracks dataset versions and stores processed data as artifacts for reproducibility.  
- **hugging face hub**: allows easy sharing and reuse of the dataset for fine-tuning llms.  

### Diagram Visualization for BaseMovieFetcher Class Diagram
```
BaseMovieFetcher (abstract)
    ├── process_movie_data()         # Calls load(), preprocess(), fetch_movie_details(), save_data()
    ├── fetch_movie_details()        # Implemented in base class
    ├── save_data()                  # Implemented in base class
    ├── load()                       # Abstract method
    └── preprocess()                 # Abstract method

CornellMovieFetcher (child class)
    ├── load()                       # Implemented in child class
    └── preprocess()                 # Implemented in child class

MovieQAFetcher (child class)
    ├── load()                       # Implemented in child class
    └── preprocess()                 # Implemented in child class
```
## Example
Below is an example of how data from the MovieQA dataset is structured in the JSON file. This structure is designed to fine-tune the model for question-answering tasks by including rich context and metadata.

```
{
    "split": "train",
    "type": "qa",
    "instruction": "Answer the following question:",
    "input": "What does Warren Worthington discover about his son?",
    "context": {
        "movie_name": "X-Men: The Last Stand",
        "genre": "Action, Adventure, Sci-Fi, Thriller",
        "year": 2006,
        "plot_outline": "When a cure is created, which apparently can turn any mutant into a normal human being, there is outrage amongst the mutant community. While some mutants do like the idea of a cure, including Rogue, many mutants find that there shouldn't be a cure. Magneto, who still believes a war is coming, recruits a large team of mutants to take down Warren Worthington II and his cure. Might seem easy for the X-Men to stop, but Magneto has a big advantage, which Wolverine doesn't have. Jean Grey has returned, and joined with Magneto. The Dark Phoenix has woken within her, which has the ability to destroy anything in her way, even if that anything is an X-Man.",
        "additional_information": null
    },
    "response": "His son is a mutant"
}
```
### Explanation

- `split`: specifies the dataset split (`train`, `val`, or `test`).  
- `type`: indicates the type of task (`qa` for question answering).  
- `instruction`: provides the task instruction for the model (e.g., "answer the following question").  
- `input`: contains the question to be answered.  
- `context`:  
  - `movie_name`: name of the movie.  
  - `genre`: genres of the movie.  
  - `year`: release year of the movie.  
  - `plot_outline`: a summary of the movie's plot to provide additional context.  
  - `additional_information`: reserved for any extra information, left `null` if not available.  
- `response`: the model's expected answer to the question.  

## Design Patterns Used

### 1. Singleton

- **Used for**:
  - `HydraConfigLocator` ensures only one configuration object is created and shared.
  - `SingletonInitializer` for centralized initialization of core dependencies (e.g., hydra, logger).

- **Why?** prevents duplication and ensures consistent configuration access throughout the project.

### 2. Factory Pattern

- **Used in** `DataLoaderFactory` to create specific `DataLoader` objects (`CornellLoader`, `MovieQALoader`).

- **Why?** simplifies object creation and makes it easier to add support for new datasets.

### 3.Service Locator

- **Used in** `service_locator.py` to register and retrieve external services like weights & biases (`wandb`) and hugging face hub (`HfApi`).

- **Why?** decouples service initialization from their usage, improving maintainability.

### 4. Template Method

- **Implemented in** `BaseMovieFetcher` with abstract methods (`load`, `preprocess`) and common methods (`fetch_movie_details`, `save_data`).

- **Why?** enforces a consistent structure while allowing customization for specific datasets.