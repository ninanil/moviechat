from omegaconf import OmegaConf
from config.class_configs.dataset_config import CornellDatasetConfig, MovieQADatasetConfig

class ConfigLoader:
    @staticmethod
    def load_cornell_config(cfg):
        cornell_config_dict = OmegaConf.to_object(cfg.cornell_dataset)
        return CornellDatasetConfig(**cornell_config_dict)

    @staticmethod
    def load_movieqa_config(cfg):
        movieqa_config_dict = OmegaConf.to_object(cfg.movieqa_dataset)
        return MovieQADatasetConfig(**movieqa_config_dict)

