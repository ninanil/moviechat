from pathlib import Path

class HydraConfigLocator:
    _config = None  # Class-level attribute to store Hydra config
    _working_directory = None  # Class-level attribute to store the working directory
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of HydraConfigLocator exists.
        """
        if cls._instance is None:
            cls._instance = super(HydraConfigLocator, cls).__new__(cls)
        return cls._instance
    @property
    def config(self):
        """
        Getter for Hydra config.
        """
        if self._config is None:
            raise ValueError("Hydra config is not set. Please set it using the setter.")
        return self._config

    @config.setter
    def config(self, cfg):
        """
        Setter for Hydra config.
        """
        self._config = cfg

    @property
    def working_directory(self):
        """
        Getter for the working directory.
        """
        if self._working_directory is None:
            raise ValueError("Working directory is not set. Please set it using the setter.")
        return self._working_directory

    @working_directory.setter
    def working_directory(self, directory):
        """
        Setter for the working directory.
        Accepts a Path object or a string and converts it to a Path.
        """
        if isinstance(directory, str):
            directory = Path(directory)
        if not isinstance(directory, Path):
            raise TypeError("Working directory must be a Path object or a string.")
        self._working_directory = directory
