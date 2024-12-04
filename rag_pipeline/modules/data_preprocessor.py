from datasets import concatenate_datasets, Value
import logging
import ast

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess_datasets(self, datasets):
        dataset1, dataset2 = datasets

        # Filter and clean dataset1
        filtered_dataset1 = dataset1.filter(
            lambda row: (
                row['plot_outline'] is not None and row['movie_name'] != 'Cherry Falls'
            )
        )
        self.logger.info("Filtered dataset1")

        # Fill NaN values in 'synopsis' with 'plot_outline'
        def fill_movie_details(example):
            if example['synopsis'] is None and example['plot_outline'] is not None:
                example['synopsis'] = example['plot_outline']
            if example['movie_name'] == 'spare me':
                example['votes'] = 31
                example['rating'] = 6.8
            return example

        dataset1 = filtered_dataset1.map(fill_movie_details)
        dataset1 = dataset1.cast_column('votes', Value('int64'))
        self.logger.info("Filled missing details in dataset1")

        # Process dataset2 if needed
        def update_dataset2(example):
            if example['movie_name'] == 'Men in Black 3: Gag Reel':
                example['plot_outline'] = (
                    "The Men in Black 3: Gag Reel is a behind-the-scenes compilation..."
                )
            return example

        dataset2 = dataset2.map(update_dataset2)
        self.logger.info("Updated dataset2")

        # Combine datasets
        combined_dataset = concatenate_datasets([dataset1, dataset2])

        # Convert list strings to actual lists and join them
        def convert_list_string(example):
            for field in ['genres', 'languages', 'director', 'writer', 'cast', 'character_names']:
                if example[field]:
                    list_data = ast.literal_eval(example[field])
                    example[field] = ', '.join(list_data)
            return example

        combined_dataset = combined_dataset.map(convert_list_string)
        self.logger.info("Converted list strings to actual strings in combined dataset")

        return combined_dataset
