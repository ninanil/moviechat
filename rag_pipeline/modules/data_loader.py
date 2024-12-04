import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import logging

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = './data'

    def download_datasets(self):
        file_paths = []
        for filename in self.config['data']['filenames']:
            file_path = hf_hub_download(
                repo_id=self.config['data']['repo_id'],
                subfolder=self.config['data']['subfolder'],
                filename=filename,
                local_dir=self.data_dir,
                repo_type="dataset"
            )
            file_paths.append(file_path)
            self.logger.info(f"Downloaded {filename} to {file_path}")
        return file_paths

    def load_datasets(self, file_paths):
        datasets = []
        for file_path in file_paths:
            dataset = load_dataset('csv', data_files=file_path, split='train')
            datasets.append(dataset)
            self.logger.info(f"Loaded dataset from {file_path}")
        return datasets

    def get_datasets(self):
        file_paths = self.download_datasets()
        datasets = self.load_datasets(file_paths)
        return datasets
