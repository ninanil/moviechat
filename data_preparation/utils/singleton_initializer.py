import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
import wandb
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pathlib import Path
from service_locator import ServiceLocator
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

class SingletonInitializer:
    """
    A Singleton class to initialize Hydra, WandB, and Hugging Face API services.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonInitializer, cls).__new__(cls)
        return cls._instance

    def initialize_hydra(self, config_path: str, config_name: str):
        """
        Initialize Hydra and compose configuration.
        """
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra.initialize(config_path=config_path, version_base="1.1")
        cfg = hydra.compose(config_name=config_name)
        logger.info("Hydra initialized successfully.")
        return cfg

    def initialize_services(self, cfg):
        """
        Initialize external services like WandB and Hugging Face API.
        """
        # Load environment variables
        load_dotenv(Path.cwd() / "config/yaml_configs/.env")
        logger.info("Environment variables loaded.")

        # Initialize WandB
        wandb.login(key=cfg.wandb.api_token)
        run = wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity)
        ServiceLocator.register_service("wandb_run", run)

        # Initialize Hugging Face API
        login(cfg.hf.api_token)
        api = HfApi()
        ServiceLocator.register_service("hf_api", api)

        logger.info("WandB and Hugging Face API initialized successfully.")
        return run, api
