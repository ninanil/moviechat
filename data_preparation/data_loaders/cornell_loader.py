import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from service.service_locator import ServiceLocator
from service.hydra_config_locator import HydraConfigLocator

class CornellDataLoader(BaseDataLoader):
    """
    Concrete data loader for Cornell movie datasets.
    """

    def __init__(self, config):
        
        self.config = config
        self.movie_dataset_df = pd.DataFrame()
        self.movie_dataset = []
        self.metadata_questions = pd.DataFrame()
        self.api = ServiceLocator.get_service("hf_api")
        self.wandb  = ServiceLocator.get_service("wandb")
        self.cfg = HydraConfigLocator.get_config()
        
    def load_data(self):
        """
        Load dataset files into pandas DataFrames.

        :return: List of pandas DataFrames.
        """
        try:
            movie_df_list = [self._read_csv_file(file_key) for file_key in self.config.files ]
            logger.info("Data loaded successfully.")
            return movie_df_list
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _read_csv_file(self, file_kay):
        """
        Read a single CSV file into a pandas DataFrame.

        :return: pandas DataFrame.
        """
        try:
            file_config = self.config.files[file_kay]
            path = os.path.join(self.config.folder_path, file_config['file_name'])
            column_names = file_config['columns']
            df = pd.read_csv(path, engine='python', sep = file_config['sep'], 
                             names = column_names, encoding=file_config['encoding'])
            logger.debug(f"Loaded data from {path} with columns {column_names}.")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Parser error while reading {path}: {e}")
            raise
            
    def preprocess_data(self, df_list):
        """
        Preprocesses the given dataframes by cleaning and formatting specific columns.

        """
        try:
            # Unpack dataframes from the input list
            movie_conversation_df, movie_utterances_df, movie_metadata_df, characters_df, imdb_details_df = df_list

            # Title case the 'character_name' column in both utterances and characters dataframes
            movie_utterances_df['character_name'] = movie_utterances_df['character_name'].str.title()
            characters_df['character_name'] = characters_df['character_name'].str.title()

            # Clean 'release_year' by removing any non-digit characters and converting to integer
            movie_metadata_df['release_year'] = movie_metadata_df['release_year'].str.replace(r'\D', '', regex=True)
            movie_metadata_df['release_year'] = movie_metadata_df['release_year'].astype(int)

            # Extract genres from the 'genre' column and join multiple genres with commas
            movie_metadata_df['genre'] = movie_metadata_df['genre'].str.extractall(r"'(.*?)'")[0].groupby(level=0).apply(', '.join)

            # Round 'imdb_rating' to one decimal place and ensure the type is float
            movie_metadata_df['imdb_rating'] = movie_metadata_df['imdb_rating'].map(lambda r: round(r, 1)).astype(float)

            # Parse the 'line_id_list' column into actual lists using a helper method
            movie_conversation_df['line_id_list'] = movie_conversation_df['line_id_list'].apply(self._parse_list_string)
            
            
            imdb_details_df = imdb_details_df[['movie_name', 'plot_outline']]
            print("preprocess imdb_details_df", imdb_details_df.columns)
            logger.info("Data preprocessing completed successfully.")

            return movie_conversation_df, movie_utterances_df, movie_metadata_df, characters_df, imdb_details_df

        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise  # Re-raise the exception after logging
    
    def _row_to_json(self, row):
        """
        Convert a DataFrame row to a JSON-like dictionary.

        :param row: pandas Series representing a row in the DataFrame.
        :return: Dictionary representing the JSON object.
        """
        

        json_obj = {
            "split": row.get('split', 'train'),
            "type":"dialogue",
            "instruction": row.get('instruction', 'Continue the conversation between the characters.'),
            "input": row.get('utterance', ''),
            "context": {
                "movie_name": row.get('movie_name', 'Unknown'),
                "character_names": row.get('character_names', 'Unknown'),
                "genre": row.get('genre', ''),
                "year": row.get('release_year', 'Unknown'),
                "imdb_rating": row.get('imdb_rating', 0),
                "num_imdb_votes": row.get('num_imdb_votes', 0),
                'plot_outline': row.get('plot_outline', 'Unknown'),
                "additional_information": None
            },
            "response": row.get('response', 'Unknown')
        }

        return json_obj

    def convert_to_json(self):
        """
        Convert the list of DataFrames into a JSON-compatible list of dictionaries.

        :param df_list: List of pandas DataFrames.
        """
        
        try:
            
            # Convert the main dataset to JSON using the existing method
            self.movie_dataset = self.movie_dataset_df.apply(self._row_to_json, axis=1).tolist()
            
        
            logger.info("Data converted to JSON format successfully.")
        except Exception as e:
            logger.error(f"Error converting data to JSON: {e}")
            raise
    def _generate_samples(self, row):
        """
        Generate new samples by modifying the question and response of a given row.

        Args:
            row (namedtuple): A row from the DataFrame with movie details and dialogue.

        Returns:
            pd.DataFrame: A DataFrame with the modified question-answer samples.
        """
        # Define question templates
        genre_questions = [
            'What genre is the movie {movie_name}?',
            'Which genres does {movie_name} belong to?',
            'Can you tell me the genre of {movie_name}?',
            'What are the main genres of {movie_name}?',
            'Under which genres is {movie_name} classified?',
            'What type of film is {movie_name}?',
            'What genres categorize {movie_name}?'
        ]

        release_year_questions = [
            'In what year was {movie_name} released?',
            'When did {movie_name} come out?',
            'What is the release year of {movie_name}?',
            'Which year did {movie_name} premiere?',
            'When was {movie_name} first released?',
            'What year did {movie_name} hit theaters?',
            'Can you tell me the release year of {movie_name}?'
        ]

        imdb_rating_questions = [
            'What is the IMDb rating of {movie_name}?',
            'How is {movie_name} rated on IMDb?',
            'Can you tell me the IMDb score for {movie_name}?',
            'What rating did {movie_name} receive on IMDb?',
            'What is the IMDb rating for {movie_name}?'
        ]

        imdb_votes_questions = [
            'How many votes does {movie_name} have on IMDb?',
            'What is the total number of IMDb votes for {movie_name}?',
            'How many people rated {movie_name} on IMDb?',
            'What is the number of votes for {movie_name} on IMDb?'
        ]
        # List to collect new samples
        new_samples = []

        # Extract values from the row
        movie_name = getattr(row, 'movie_name', 'Unknown')
        genre = getattr(row, 'genre', 'Unknown')
        year = getattr(row, 'release_year', 'Unknown')
        imdb_rating = getattr(row, 'imdb_rating', 'Unknown')
        imdb_votes = getattr(row, 'num_imdb_votes', 'Unknown')
        
        # Helper function to create a new modified row
        def add_modified_row(new_question, new_response):
            # Convert row back to a dictionary and modify question and response
            new_row = row._asdict()  # Convert namedtuple row to a dictionary
            new_row['utterance'] = new_question
            new_row['response'] = new_response
            new_row['instruction'] = 'Answer the following question:'
            new_samples.append(new_row)  # Add the modified row to the samples list

        # Modify question and response based on attributes and add the new row to the samples list
        if genre != 'Unknown':
            new_question = random.choice(genre_questions).format(movie_name=movie_name)
            add_modified_row(new_question, genre)

        if year != 'Unknown':
            new_question = random.choice(release_year_questions).format(movie_name=movie_name)
            add_modified_row(new_question, year)

        if imdb_rating != 'Unknown':
            new_question = random.choice(imdb_rating_questions).format(movie_name=movie_name)
            add_modified_row(new_question, imdb_rating)

        if imdb_votes != 'Unknown':
            new_question = random.choice(imdb_votes_questions).format(movie_name=movie_name)
            add_modified_row(new_question, imdb_votes)

        return new_samples

    def _parse_list_string(self, value):
        """
        Parse a string representation of a list into an actual list.

        :param value: String to parse.
        :return: List of extracted strings.
        """
        clean_string = re.findall(r"'(.*?)'", value)

        if clean_string:
            return clean_string
        else:
            logger.warning(f"Could not parse: {value}")
            return []

    def merge_dataframes(self, df_list):
        """
        Merge multiple DataFrames into a single DataFrame for processing.

        :param df_list: List of pandas DataFrames.
        :return: Merged pandas DataFrame.
        """
        try:
            if len(df_list) != 5:
                raise ValueError("df_list must contain exactly four DataFrames.")

            movie_conversation_df, movie_utterances_df, movie_metadata_df, characters_df, imdb_details_df = df_list

            # Concatenate character name with utterance
            movie_utterances_df['utterance'] = movie_utterances_df['character_name'].str.cat(
                movie_utterances_df['utterance'], sep=': '
            )
            
            # Merge conversation and metadata DataFrames on 'movie_id'
            movie_conversation_metadata_df = pd.merge(
                left=movie_conversation_df,
                right=movie_metadata_df,
                on='movie_id',
                how='inner'
            )
            
            # Merge with character details
            movie_conversation_metadata_df = pd.merge(
                left=movie_conversation_metadata_df,
                right=characters_df,
                left_on='character_id1',
                right_on='character_id',
                how='inner'
            )
            
            
            # Drop redundant columns
            movie_conversation_metadata_df.drop(
                columns=['character_id1', 'movie_id_x', 'movie_id_y', 'character_id', 'gender', 'position'],
                inplace=True
            )
            
            # Map character IDs to names
            characters_dict = characters_df.set_index('character_id')['character_name'].to_dict()
            movie_conversation_metadata_df['character_name2'] = movie_conversation_metadata_df['character_id2'].map(characters_dict)

            # Rename columns for clarity
            movie_conversation_metadata_df.rename(
                columns={'character_name': 'character_name1', 'movie_name_x': 'movie_name'},
                inplace=True
            )
            movie_conversation_metadata_df.drop(columns=['character_id2'], inplace=True)
           

            # Save the original DataFrame index for grouping
            movie_conversation_metadata_df['original_line_index'] = movie_conversation_metadata_df.index

            # Explode the 'line_id_list' to have one row per line ID
            movie_conversation_metadata_df = movie_conversation_metadata_df.explode('line_id_list')

            # Merge with utterances DataFrame using 'line_id_list' as the key
            movie_conversation_metadata_df = pd.merge(
                left=movie_conversation_metadata_df,
                right=movie_utterances_df,
                left_on='line_id_list',
                right_index=True,
                how='inner'
            )

            # Concatenate character names
            movie_conversation_metadata_df['character_names'] = (
                movie_conversation_metadata_df['character_name1'] + ', ' +
                movie_conversation_metadata_df['character_name2']
            )
            # Create 'response' column as a copy of 'utterance'
            movie_conversation_metadata_df['response'] = movie_conversation_metadata_df['utterance'].copy(deep=True)
            
            # Aggregate the data
            movie_dataset_df = movie_conversation_metadata_df.groupby('original_line_index').agg({
                'character_names': 'first',
                'movie_name': 'first',
                'release_year': 'first',
                'imdb_rating': 'mean',
                'num_imdb_votes': 'first',
                'genre': 'first',
                'utterance': lambda u: '\n '.join(map(str, u[:-1])) if len(u) > 1 else u.iloc[0],
                'response': 'last'
            }).reset_index().drop(columns=['original_line_index'])
            
            # print("in merge_dataframe imdb_details_df", imdb_details_df.columns, imdb_details_df.size)
            # print("imdb_details_df", imdb_details_df.head(5))
            movie_dataset_df = pd.merge(left = movie_dataset_df, right = imdb_details_df, on = 'movie_name', how='left')
            # print("movie_dataset_df", movie_dataset_df.head(5), movie_dataset_df.size)
            # Remove duplicate rows based on all columns
            movie_dataset_df = movie_dataset_df.drop_duplicates()
            # Remove rows where 'utterance' or 'response' are NaN
            self.movie_dataset_df = movie_dataset_df.dropna(subset=['utterance', 'response'])
            self.movie_dataset_df.loc[:, 'instruction'] = 'Continue the conversation between the characters.'
            logger.debug("DataFrames merged successfully.")
            # Generate new samples if flagged
            if self.config.generate_new_questions:
                # Sample a fraction of the data
                sampled_df = self.movie_dataset_df.groupby('movie_name').agg({'character_names':'first','movie_name':'first','genre':'first', 'release_year':'first','imdb_rating':'mean', 'num_imdb_votes':'first'}).sample(frac=self.config.sample_fraction, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled {len(sampled_df)} data successfully.")

                # Generate new samples from the sampled DataFrame
                #new_sample_list = sampled_df.apply(self._generate_json_sample, axis=1).tolist()
                new_sample_list = [sample for row in sampled_df.itertuples(index=False) 
                                            for sample in self._generate_samples(row)]
                new_sample_df = pd.DataFrame(new_sample_list)
                
              
            self.movie_dataset_df = pd.concat([self.movie_dataset_df, new_sample_df], axis = 0).reset_index(drop = True)
            self.movie_dataset_df['response'] = self.movie_dataset_df['response'].astype(str)    
            if self.config.data_prune_enabled:
                self._data_pruning()
            self._save_df()
            logger.debug(f"Total number of rows found in movie_dataset_df: {len(self.movie_dataset_df)}")
            return self.movie_dataset_df
        except Exception as e:
            logger.error(f"Error merging DataFrames: {e}")
            raise
    def _data_pruning(self):

        logger.info(f"Cornell dataset size before pruning: {self.movie_dataset_df.shape[0]} rows")
        filtered_movies = self.movie_dataset_df.groupby('movie_name').size()
        filtered_movies = filtered_movies[filtered_movies > self.config.frequent_sample_ratio].index

        # Filter the main dataset to include only those movies
        self.movie_dataset_df = self.movie_dataset_df[self.movie_dataset_df['movie_name'].isin(filtered_movies)].reset_index(drop = True)
        logger.info(f"Cornell dataset size after pruning: {self.movie_dataset_df.shape[0]} rows")
        
    def train_val_test_split(self):
        # Filtering movies that appear more than 50 times
        # filtered_movies = self.movie_dataset_df.groupby('movie_name').size()
        # filtered_movies = filtered_movies[filtered_movies > 50].index

        # Filter the main dataset to include only those movies
        # filtered_dataset = self.movie_dataset_df[self.movie_dataset_df['movie_name'].isin(filtered_movies)]

        # Calculating sample sizes for training, validation, and test
        total_records = len(self.movie_dataset_df)
        train_size = int(total_records * self.config.train_ratio)  # 90% training
        test_size = int(total_records * self.config.test_ratio)    # 0.05% test
        val_size = total_records - train_size - test_size  # remaining for validation
        
        # Split the data into train, validation, and test
        train_data, temp_data = train_test_split(
            self.movie_dataset_df, 
            train_size=train_size, 
            stratify=self.movie_dataset_df['movie_name'], 
            random_state=42
        )

        try:
             test_data,val_data = train_test_split(
                temp_data, 
                train_size=test_size, 
                stratify=temp_data['movie_name'], 
                random_state=42
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
            test_data,val_data = train_test_split(
                temp_data, 
                train_size=test_size, 
                random_state=42
            )
        # Assign the split column
        train_data['split'] = 'train'
        val_data['split'] = 'val'
        test_data['split'] = 'test'

        # Combine the data back into a single dataset
        self.movie_dataset_df = pd.concat([train_data, val_data, test_data], ignore_index=True)

        # Logging the size of each dataset
        logger.info(f"Training set size: {len(train_data)}")
        logger.info(f"Validation set size: {len(val_data)}")
        logger.info(f"Test set size: {len(test_data)}")
        
        return self.movie_dataset_df

    def save_data(self):
        """
        Save the converted JSON data to the specified output file.
        """
        try:
            full_output_path  = os.path.join(self.config.folder_path, self.config.output_path)
            with open(full_output_path , 'w', encoding='utf-8') as f:
                json.dump(self.movie_dataset, f, ensure_ascii=False, indent=4)
            logger.info(f"Data saved to {full_output_path }")
            if self.cfg.wandb.cornell_dataset.to_wandb:
                artifact = self.wandb.Artifact(name = self.cfg.wandb.cornell_dataset.json_artifact_name,
                                          description = self.cfg.wandb.cornell_dataset.description,type='dataset')  # Name and type for the artifact
                artifact.add_file(full_output_path)  # Add the saved JSON file to the artifact
                artifact.save()
                logger.info(f"JSON file '{self.config.output_path}' uploaded to WandB successfully.")
            if self.cfg.hf.cornell_dataset.to_hf:
                 self.api.upload_file(
                    path_or_fileobj=full_output_path,
                    repo_id=self.cfg.hf.repo_id,
                    path_in_repo=f"{self.cfg.hf.cornell_dataset.path_in_repo}{self.cfg.hf.cornell_dataset.file_name}",
                    repo_type="dataset")
                 logger.info(f"File {self.config.output_path}  logged to Huggingface successfully.")
            return full_output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def _save_df(self):
        """
        Save the converted JSON data to the specified output file.
        """
        try:
            output_path = os.path.join(self.config.folder_path,'movie_dataset.csv')
            # Save the DataFrame to a CSV file
            self.movie_dataset_df.to_csv(output_path, index=False)
            logger.info(f"DataFrame saved as CSV file at '{output_path}'.")
            if self.cfg.wandb.cornell_dataset.to_wandb:
                # Create a WandB artifact
                artifact = self.wandb.Artifact(name = self.cfg.wandb.cornell_dataset.csv_artifact_name, type= 'dataset',
                                          description = self.cfg.wandb.cornell_dataset.description,
                                         metadata = {'size':len(self.movie_dataset_df),
                                                    'columns': self.movie_dataset_df.columns.to_list()
    })
    
                # Add the saved CSV file to the artifact
                artifact.add_file(output_path)
                logger.info(f"CSV file '{output_path}' added to WandB artifact cornell_movie_df.")
    
                # Save the artifact
                wandb.log_artifact(artifact)
                logger.info(f"Artifact cornell_movie_df logged to WandB successfully.")

        except Exception as e:
            logger.error(f"Error saving DataFrame or uploading to WandB: {e}")
