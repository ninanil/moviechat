from config.config_loader.config_loader import ConfigLoader

class DataLoaderFactory:
    """
    Factory class to create DataLoader instances.
    """
    @staticmethod
    def create_dataloader(loader_type: str, cfg):
        if loader_type == "cornell":
            cornell_config = ConfigLoader.load_cornell_config(cfg)
            return CornellDataLoader(cornell_config)
        elif loader_type == "movieqa":
            movieqa_config = ConfigLoader.load_movieqa_config(cfg)
            return MovieQADataLoader(movieqa_config)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

