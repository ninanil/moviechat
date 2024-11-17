import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from service.service_locator import ServiceLocator
from service.hydra_config_locator import HydraConfigLocator
from utils.logger import get_logger

class MovieQADataLoader(BaseDataLoader):
    def __init__(self,config):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config  # Retrieve the config from the locator
        self.qa_dataset = []
        self.qa_movie_metadata_df = pd.DataFrame()
        self.api = ServiceLocator.get_service("hf_api")
        self.wandb  = ServiceLocator.get_service("wandb")
        self.cfg = HydraConfigLocator.get_config()
        
    def load_data(self):
        """
        Load Q&A dataset.
        """
        try:
            qa_df_list = []
            for file,file_config in self.config.files.items():
                path = os.path.join(self.config.folder_path,file_config['file_name'])
                if path.endswith('.json'):
                    qa_df_list.append(self._read_json_file(path))
                else:
                    qa_df_list.append(self._read_csv_file(file_config))
            
            self.logger.info("Data loaded successfully.")
            return qa_df_list
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    def _read_json_file(self, data_path):
        try:
            qa_df = pd.read_json(data_path)
            self.logger.info("Q&A JSON file loaded successfully.")
            return qa_df
        except Exception as e:
            self.logger.error(f"Error loading Q&A JSON file: {e}")
            raise
    def _read_csv_file(self, file):
        try:
            path = os.path.join(self.config.folder_path, file['file_name'])
            qa_df = pd.read_csv(path, usecols=file['columns'], 
                                sep = file['sep'], encoding=file['encoding'])
            self.logger.info("Q&A CSV file loaded successfully.")
            return qa_df
        except Exception as e:
            self.logger.error(f"Error loading Q&A JSON file: {e}")
            raise
    def preprocess_data(self,df_list):
        qa_df, movie_df, imdb_details_df = df_list
        movie_df = movie_df[['genre', 'imdb_key', 'name', 'year']]
        qa_df = qa_df[qa_df['correct_index'].notna()]
        qa_df['response'] = qa_df.apply(lambda row: row['answers'][int(row['correct_index'])], axis=1)
        qa_df = qa_df[['question','imdb_key' ,'response']]
        return qa_df, movie_df, imdb_details_df
    def merge_dataframes(self, df_list): 
        qa_df, movie_df, imdb_details_df = df_list
        self.qa_movie_metadata_df = pd.merge(left = qa_df, right = movie_df, on = 'imdb_key', how='left')
        self.qa_movie_metadata_df = pd.merge(left = self.qa_movie_metadata_df , right = imdb_details_df, left_on = 'name', right_on = 'title', how= 'left').drop(columns=['title']).rename(columns={'name':'movie_name'})
        self.qa_movie_metadata_df.drop(columns=['imdb_key'], inplace = True)
        self._save_df()
        return self.qa_movie_metadata_df
    def convert_to_json(self):
        self.qa_dataset = self.qa_movie_metadata_df.apply(self._row_to_json, axis=1).tolist()
        return self.qa_dataset
    def _row_to_json(self, row):
        return {
            "split": row.get('split', 'train'),
            "type":"qa",
            "instruction": "Answer the following question:",
            "input": row['question'],
            "context": {
                "movie_name": row.get('movie_name', 'Unknown'),
                "genre": row.get('genre', 'Unknown'),
                "year": row.get('year', 'Unknown'),
                "plot_outline": row.get('plot_outline', 'Unknown'),
                "additional_information": None
            },
            "response": row['response']
        }
    def train_val_test_split(self):
        # Filtering movies that appear more than 50 times
        filtered_qa_movies = self.qa_movie_metadata_df.groupby('movie_name').size()
        filtered_qa_movies = filtered_qa_movies[filtered_qa_movies > 50].index

        # Filter the main dataset to include only those movies
        filtered_dataset = self.qa_movie_metadata_df[self.qa_movie_metadata_df['movie_name'].isin(filtered_qa_movies)]

        # Calculating sample sizes for training, validation, and test
        total_records = len(filtered_dataset)
        train_size = int(total_records * self.config.train_ratio)  # 90% training
        test_size = int(total_records * self.config.test_ratio)    # 0.05% test
        val_size = total_records - train_size - test_size  # remaining for validation

        # Split the data into train, validation, and test
        train_data, temp_data = train_test_split(
            filtered_dataset, 
            train_size=train_size, 
            stratify=filtered_dataset['movie_name'], 
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
            self.logger.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
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
        self.qa_movie_metadata_df = pd.concat([train_data, val_data, test_data], ignore_index=True)

        # Logging the size of each dataset
        self.logger.info(f"Training set size: {len(train_data)}")
        self.logger.info(f"Validation set size: {len(val_data)}")
        self.logger.info(f"Test set size: {len(test_data)}")
        return self.qa_movie_metadata_df
    
    def save_data(self):
         try:
            full_output_path = os.path.join(self.config.folder_path ,self.config.output_path)#os.path.join(self.folder_path,self.output_path)
            with open(full_output_path, 'w', encoding='utf-8') as f:
                json.dump(self.qa_dataset, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Q&A Movie dataset saved to {self.config.output_path}")
            if self.cfg.wandb.movieqa_dataset.to_wandb:
                artifact = self.wandb.Artifact(name=self.cfg.wandb.movieqa_dataset.json_artifact_name, 
                                          description= self.cfg.wandb.movieqa_dataset.description, type='dataset')  # Name and type for the artifact
                artifact.add_file(full_output_path)  # Add the saved JSON file to the artifact
                #wandb.log_artifact(artifact)  # Log the artifact to WandB
                artifact.save()
            if self.cfg.hf.movieqa_dataset.to_hf:
                self.api.upload_file(
                    path_or_fileobj=full_output_path,
                    repo_id= self.cfg.hf.repo_id,
                    path_in_repo = self.cfg.hf.movieqa_dataset.path_in_repo,
                    repo_type="dataset",
                    )
                self.logger.info(f"Artifact {self.config.output_path}  logged to Huggingface successfully.")
            return full_output_path
         except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
    def _save_df(self):
        """
        Save the converted JSON data to the specified output file.
        """
        try:
            output_path = os.path.join(self.config.folder_path ,'qa_movie_df.csv')
            # Save the DataFrame to a CSV file
            self.qa_movie_metadata_df.to_csv(output_path, index=False)
            self.logger.info(f"DataFrame saved as CSV file at '{output_path}'.")
            
            if self.cfg.wandb.movieqa_dataset.to_wandb:
                # Create a WandB artifact
                artifact = self.wandb.Artifact(name = self.cfg.wandb.movieqa_dataset.csv_artifact_name, type = 'dataset',
                                          description= self.cfg.wandb.movieqa_dataset.description,
                                        metadata = {'size':len(self.qa_movie_metadata_df),
                                                    'columns':self.qa_movie_metadata_df.columns.to_list()},
                                         )
    
                # Add the saved CSV file to the artifact
                artifact.add_file(output_path)
                self.logger.info(f"CSV file '{output_path}' added to WandB artifact qa_movie_df.")
    
                # Save the artifact
                artifact.save()
                self.logger.info(f"Artifact qa_movie_df  logged to WandB successfully.")

        except Exception as e:
            self.logger.error(f"Error saving DataFrame or uploading to WandB: {e}")
