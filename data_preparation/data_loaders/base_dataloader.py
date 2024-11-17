import os
import json
import logging
from abc import ABC, abstractmethod
from service.service_locator import ServiceLocator
from service.hydra_config_locator import HydraConfigLocator
from utils.logger import get_logger

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def load_data(self):
        """
        Load data from the specified source.
        """
        pass

    @abstractmethod
    def preprocess_data(self):
        """
        Preprocess the loaded data.
        """
        pass
    
    @abstractmethod
    def train_val_test_split(self):
        pass
    
    @abstractmethod
    def merge_dataframes(self):
        pass

    @abstractmethod
    def convert_to_json(self):
        """
        Convert the processed data to JSON format.
        """
        pass

    @abstractmethod
    def save_data(self):
        """
        Save the JSON data to a file.
        """
        pass

    def process_movie_data(self):
        """
        High-level workflow for processing movie data.
        This method calls abstract and child class-specific methods in a consistent sequence.
        """
        # Load data
        data_frames = self.load()

        # Preprocess data
        data_frames = self.preprocess_data(data_frames)

        # Merge dataframes
        merged_data = self.merge_dataframes(data_frames)

        # Train test Split
        self.train_val_test_split()

        # Convert data to JSON format
        json_data = self.convert_to_json(merged_data)

        # Save the JSON data
        file_path = self.save_data(json_data)
        return file_path

    @staticmethod
    def concat_json_files(file_path1, file_path2):
        try:
            api = ServiceLocator.get_service("hf_api")
            wandb  = ServiceLocator.get_service("wandb")
            cfg = HydraConfigLocator.get_config()  # Retrieve the config from the locator 
            # Open and read the first JSON file
            with open(file_path1, "r") as file1:
                data1 = json.load(file1)
            self.logger.info(f"Successfully loaded data from {file_path1}")

            # Open and read the second JSON file
            with open(file_path2, "r") as file2:
                data2 = json.load(file2)
            self.logger.info(f"Successfully loaded data from {file_path2}")

            # Concatenate the data from both files
            data = data1 + data2
            self.logger.info(f"Successfully concatenated data from {file_path1} and {file_path2}")
            full_output_path = os.path.join(".",cfg.hf.combined_dataset.file_name)
            with open(full_output_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
            if cfg.hf.combined_dataset.to_hf:
                api.upload_file(
                path_or_fileobj=cfg.hf.combined_dataset.file_name,
                repo_id= cfg.hf.repo_id,
                path_in_repo = f"{cfg.hf.combined_dataset.path_in_repo}/{cfg.hf.combined_dataset.file_name}",
                repo_type="dataset",
                commit_message=cfg.hf.combined_dataset.commit_message,
                commit_description=cfg.hf.combined_dataset.commit_description
                    )
                self.logger.info(f"File {cfg.hf.combined_dataset.file_name}  logged to Huggingface successfully.")
                if cfg.wandb.combined_dataset.to_wandb:
                    artifact = wandb.Artifact(name=cfg.wandb.combined_dataset.json_artifact_name, 
                                          description= cfg.wandb.combined_dataset.description, type='dataset')  # Name and type for the artifact
                    artifact.add_file(full_output_path)  # Add the saved JSON file to the artifact
                    artifact.save()
                    self.logger.info(f"File {cfg.hf.combined_dataset.file_name}  logged to WandB successfully.")
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            raise

        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
