from data_fetchers.movie_fetcher import BaseMovieFetcher
import os
from pathlib import Path
from utils.logger import get_logger

class CornellMovieFetcher(BaseMovieFetcher):
    def __init__(self, config, file_name):
        """
        Initialize CornellMovieFetcher with specific config.
        
        Parameters:
        config: Hydra configuration object for the Cornell dataset.
        """
        super().__init__(config)  # Pass the config to the base class
        self.file_name = file_name
        self.logger = get_logger(self.__class__.__name__)
        
    def load(self):
        """
        Load the Cornell movie metadata from the file.

        Returns:
        DataFrame: Loaded metadata as a pandas DataFrame.
        """
        try:
            working_directory = Path.cwd() 
            # Construct the file path
            file_path = os.path.join(working_directory, self.file_name)
            
            # Load the movie metadata into a DataFrame
            movie_metadata_df = pd.read_csv(
                file_path,
                engine='python',
                sep=' \+\+\+\$\+\+\+ ',  # Custom separator used in Cornell movie metadata
                names=['movie_id', 'movie_name', 'release_year', 'imdb_rating', 'num_imdb_votes', 'genre'],
                encoding='iso-8859-1'
            )
            self.logger.info(f"Cornell movie metadata loaded successfully from {file_path}.")
            return movie_metadata_df
        except Exception as e:
            self.logger.error(f"Error loading Cornell movie metadata: {e}")
            raise
        
    def preprocess(self, movie_df):
        """
        Preprocess the Cornell dataset.
        
        Parameters:
        movie_df (DataFrame): The DataFrame containing movie data.
        
        Returns:
        DataFrame: The preprocessed DataFrame.
        """
        try:
            movie_year_df = movie_df[['movie_name','release_year']].drop_duplicates()

            movie_year_df['release_year'] = movie_year_df.release_year.str.replace(r'\D',"", regex = True).astype(int)
            self.logger.info("Cornell dataset preprocessing completed.")
            return movie_year_df
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise
