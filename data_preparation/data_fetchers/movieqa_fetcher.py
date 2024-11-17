from data_fetchers.movie_fetcher import BaseMovieFetcher
import pandas as pd
from utils.logger import get_logger
import os
from pathlib import Path

class MovieQAFetcher(BaseMovieFetcher):
    def __init__(self, config, file_path):
        """
        Initialize CornellMovieFetcher with specific config.
        
        Parameters:
        config: Hydra configuration object for the Cornell dataset.
        """
        super().__init__(config)  # Pass the config to the base class
        self.file_path =  file_path
        self.logger = get_logger(self.__class__.__name__)
        
    def load(self):
        working_directory =  Path.cwd()
        path =  os.path.join(working_directory, self.file_path)
        return movie_df
        
    def preprocess(self, movie_metadata_df):
        try:
            movie_df = movie_df[[ 'name', 'year']].rename(columns={'name':'movie_name','year':'release_year'})
            self.logger.info("MovieQA dataset preprocessing completed.")
            return movie_df
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise
