import os
from huggingface_hub import HfApi, login, snapshot_download
from utils.singleton_initializer import SingletonInitializer

# Get the singleton instance
initializer = SingletonInitializer()

# Initialize Hydra
cfg = initializer.initialize_hydra(config_path="../config/yaml_configs", config_name="config")

# Initialize services
run, api = initializer.initialize_services(cfg)

# Use CornellMovieFetcher
cornell_fetcher_config = cfg.wandb.movie_fetcher.cornell_dataset
cornell_fetcher = CornellMovieFetcher(cornell_fetcher_config, "datasets/cornell/movie_titles_metadata.txt")

# Load, preprocess, and save movie data
cornell_fetcher.process_movie_data()

# Use MovieQAFetcher
movieqa_fetcher_config = cfg.wandb.movie_fetcher.movieqa_dataset
movieqa_fetcher = MovieQAFetcher(movieqa_fetcher_config, "datasets/movieqa/movieqa_metadata.csv")

# Load, preprocess, fetch movie details, and save movie data
# Load, preprocess, and save movie data
movieqa_fetcher.process_movie_data()


# Download the entire repository (including folders)
local_dir = snapshot_download(repo_id = cfg.hf.repo_id, repo_type="dataset", allow_patterns= cfg.hf.allow_patterns, local_dir = '/kaggle/working/')

logger.info(f"Downloaded repository to {local_dir}")

cornell_loader = DataLoaderFactory.create_dataloader("cornell", cfg.cornell_dataset)
file_path = cornell_loader.process_movie_data()

movieqa_loader = DataLoaderFactory.create_dataloader("movieqa", cfg.movieqa_dataset)

file_path2 = movieqa_loader.process_movie_data()

combined_data = BaseDataLoader.concat_json_files(file_path, file_path2)

    
