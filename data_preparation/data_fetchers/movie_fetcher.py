import time
from imdb import IMDb
from imdb.Character import Character
import pandas as pd
from abc import ABC, abstractmethod
from utils.logger import get_logger
from pathlib import Path
from service.hydra_config_locator import HydraConfigLocator

class BaseMovieFetcher(ABC):
    def __init__(self, config, csv_file_name='my_file.csv', batch_size=50, delay=5):
        """
        Initialize MovieFetcher with IMDb instance and batch settings.
        
        Parameters:
        csv_file_name (str): Name of the CSV file to save movie details.
        batch_size (int): Number of movies to fetch in one batch.
        delay (int): Number of seconds to wait between batches.
        config: Hydra configuration object for the dataset.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.ia = IMDb()
        self.csv_file_name = csv_file_name
        self.batch_size = batch_size
        self.delay = delay
        self.movie_data = []
        self.not_found_movies = []
    def load(self):
        """
        Load the dataset from a CSV file.
        """
        try:
            working_directory = HydraConfigLocator().working_directory
            csv_file_path = os.path.join(working_directory, self.csv_file_name)
            movie_df = pd.read_csv(csv_file_path)
            self.logger.info(f"Dataset loaded successfully from {path}.")
            return movie_df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def fetch_movie_details(self, movie_df):
        """
        Fetch movie details from IMDb in batches to avoid rate limits.
        
        Parameters:
        movie_df (DataFrame): DataFrame containing 'title' and optionally 'year'.
        
        Returns:
        movie_data (list): List of dictionaries with movie details.
        """
        for i, row in enumerate(movie_df.itertuples(index=False)):
            title = row.movie_name
            if title == "Nirgendwo in Afrika":
                title = "Nirgendwo in Africa"
            year = row.release_year if hasattr(row, 'release_year') else None

            try:
                movie = self._fetch_single_movie(title, year)
                if movie:
                    self.movie_data.append(movie)
                    print(f"Fetched details for: {title}")
                else:
                    self.not_found_movies.append({'movie_name': title, 'year': year})
                    print(f"Movie not found: {title}")
            except Exception as e:
                print(f"Error fetching details for movie '{title}': {e}")

            # Batch processing logic
            if (i + 1) % self.batch_size == 0:
                print(f"Batch {i // self.batch_size + 1} complete. Waiting {self.delay} seconds...")
                time.sleep(self.delay)

        self._save_data()

    def _fetch_single_movie(self, title, year):
        """
        Fetch details for a single movie by title and year.
        
        Parameters:
        title (str): Movie title.
        year (int or None): Release year of the movie, if available.
        
        Returns:
        dict or None: Dictionary of movie details or None if not found.
        """
        search_results = self.ia.search_movie(title)
        if title == "Nirgendwo in Africa":
            title = "Nirgendwo in Afrika"

        found_movie = None
        for movie in search_results:
            if year and 'year' in movie.data and movie.data['year'] == year:
                found_movie = movie
                break
            elif not year:
                found_movie = movie
                break

        if found_movie:
            self.ia.update(found_movie)
            plot_outline = found_movie.get('plot outline')
            if not plot_outline and found_movie.get('plot') and found_movie.get('synopsis'):
                plot_outline = max(found_movie.get('synopsis'), found_movie.get('plot'), key=len)
            
            return {
                'movie_name': title,
                'year': found_movie.get('year', ''),
                'kind': found_movie.get('kind', ''),
                'cover_url': found_movie.get('cover url', ''),
                'original_title': found_movie.get('original title', ''),
                'localized_title': found_movie.get('localized title', ''),
                'genres': found_movie.get('genres', []),
                'runtimes': found_movie.get('runtimes', []),
                'countries': found_movie.get('countries', []),
                'language_codes': found_movie.get('language codes', []),
                'rating': found_movie.get('rating', ''),
                'votes': found_movie.get('votes', ''),
                'imdbID': found_movie.get('imdbID', ''),
                'plot_outline': plot_outline,
                'languages': found_movie.get('languages', []),
                'director': [director['name'] for director in found_movie.get('director', []) if director.get('name')],
                'writer': [writer['name'] for writer in found_movie.get('writer', []) if writer.get('name')],
                'cast': [actor['name'] for actor in found_movie.get('cast', []) if actor.get('name')],
                'character_names': [
                    actor.currentRole.get('name')  
                    for actor in found_movie.get('cast', []) if actor.currentRole and isinstance(actor.currentRole, Character)
                ],
                'box_office': found_movie.get('box office', {}),
                'plot': found_movie.get('plot', ""),
                'synopsis': found_movie.get('synopsis', ""),
            }
        return None
    def process_movie_data(self):
        """
        A reusable function to handle repetitive operations
        like loading, preprocessing, fetching details, and saving data.
        """
        movie_df = self.load()
        processed_movie_df = self.preprocess(movie_df)
        self.fetch_movie_details(processed_movie_df)
        self.save_data()
        return processed_movie_df

    def save_data(self):
        """
        Save the processed movie data to a CSV file and log it to WandB.
        """
        try:
            output_path = os.path.join(self.folder_path, self.config.artifact_name)
            movie_data_df = pd.DataFrame(self.movie_data)
            movie_data_df.to_csv(output_path, index=False)
            self.logger.info(f"Movie data saved to {output_path}.")

            # Save as WandB artifact using the ServiceLocator for the wandb service
            wandb = ServiceLocator.get_service("wandb")
            if self.config.to_wandb:
                artifact = wandb.Artifact(
                    name=self.config.artifact_name,
                    type=self.config.artifact_type,
                    description=self.config.artifact_description,
                    metadata={
                        'size': len(self.movie_data),
                        'columns': movie_data_df.columns.to_list(),
                    }
                )
                artifact.add_file(output_path)
                artifact.save()
                self.logger.info(f"Artifact saved to WandB: {self.config.artifact_name}.")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
        
    @abstractmethod
    def preprocess(self, movie_df):
        """
        Abstract method for preprocessing data.
        Subclasses must implement this method.
        """
        self.logger.info("Preprocessing method is not implemented.")
        pass
