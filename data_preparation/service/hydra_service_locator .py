class HydraConfigLocator:
    _config = None  # Class-level attribute to store Hydra config

    @staticmethod
    def set_config(cfg):
        """
        Register the Hydra config with the locator.
        """
        HydraConfigLocator._config = cfg

    @staticmethod
    def get_config():
        """
        Retrieve the Hydra config from the locator.
        """
        if HydraConfigLocator._config is None:
            raise ValueError("Hydra config is not set. Please set it in the main script.")
        return HydraConfigLocator._config
