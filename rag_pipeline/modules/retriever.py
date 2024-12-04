from abc import ABC, abstractmethod
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import ConfigurableField
import logging

class Retriever(ABC):
    """
    Abstract base class for retrievers.
    """
    def __init__(self, config, chroma_index):
        self.config = config
        self.chroma_index = chroma_index
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def initialize_retriever(self) -> EnsembleRetriever:
        """
        Initialize the EnsembleRetriever. To be implemented by child classes.
        """
        pass

class MovieNameRetriever(Retriever):
    """
    Specialized retriever for extracting movie names.
    """
    def initialize_retriever(self) -> EnsembleRetriever:
        """
        Initialize the ensemble retriever with specialized weights.
        """
        # Define individual retrievers
        retriever_vanilla = self.chroma_index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config['k_vanilla']}
        )
        retriever_mmr = self.chroma_index.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.config['k_mmr']}
        )

        # Initialize EnsembleRetriever with specified weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vanilla, retriever_mmr],
            weights=self.config['weights']
        )
        self.logger.info(f"Initialized MovieNameRetriever with weights: {self.config['weights']}")
        return ensemble_retriever

class MovieInfoRetriever(Retriever):
    """
    Specialized retriever for fetching detailed movie information.
    """
    def initialize_retriever(self) -> EnsembleRetriever:
        """
        Initialize the ensemble retriever with specialized weights.
        """
        # Define ConfigurableFields
        vanilla_filter = ConfigurableField(
            id="search_kwargs_vanilla",
            name="Vanilla Filter",
            description="Filter for vanilla retriever"
        )
        mmr_filter = ConfigurableField(
            id="search_kwargs_mmr",
            name="MMR Filter",
            description="Filter for MMR retriever"
        )

        # Initialize individual retrievers with ConfigurableFields
        retriever_vanilla = self.chroma_index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config['k_vanilla']}
        ).configurable_fields(search_kwargs=vanilla_filter)

        retriever_mmr = self.chroma_index.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.config['k_mmr']}
        ).configurable_fields(search_kwargs=mmr_filter)

        # Initialize EnsembleRetriever with specified weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vanilla, retriever_mmr],
            weights=self.config['weights']
        )
        self.logger.info(f"Initialized MovieInfoRetriever with weights: {self.config['weights']}")
        return ensemble_retriever
