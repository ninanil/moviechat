from abc import ABC, abstractmethod
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
import logging

class VectorStoreInitializer(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model_name = self.config['embedding']['model_name']
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": self.device}
        )

        self.chroma_index = Chroma(
            embedding_function=self.embedding_function,
            collection_name=self.config['chroma']['collection_name'],
            persist_directory=self.config['chroma']['persist_directory']
        )
        self.logger.info(f"Initialized Chroma vector store '{self.config['chroma']['collection_name']}' at '{self.config['chroma']['persist_directory']}'")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['text_splitter']['chunk_size'],
            chunk_overlap=self.config['text_splitter']['chunk_overlap']
        )

    @abstractmethod
    def process_movie(self, row):
        """
        Process a single movie record and return full_text and metadata.
        """
        pass

    def add_documents(self, combined_dataset):
        self.logger.info("Starting ChromaDB integration...")
        for row in combined_dataset:
            full_text, metadata = self.process_movie(row)
            chunks = self.text_splitter.split_text(full_text)
            metadatas = [metadata] * len(chunks)
            self.chroma_index.add_texts(chunks, metadatas=metadatas)
            self.logger.info(f"Adding {len(chunks)} chunks to Chroma for movie '{metadatas[0].get('movie_name', 'Unknown')}'")

class MovieNameVectorStore(VectorStoreInitializer):
    def __init__(self, config):
        super().__init__(config)

    def process_movie(self, row):
        # Check for missing metadata
        if not row.get('movie_name') or not row.get('year'):
            self.logger.warning(f"Missing metadata for row: {row}")

        full_text = (
            f"Movie Name: {row.get('movie_name', 'Unknown')}\n"
            f"Year: {row.get('year', 'Unknown')}\n"
            f"Rating: {row.get('rating', 'Unknown')}\n"
            f"Votes: {row.get('votes', 'Unknown')}\n"
            f"Genre: {row.get('genre', 'Unknown')}\n"
            f"Director: {row.get('director', 'Unknown')}\n"
            f"Writer: {row.get('writer', 'Unknown')}\n"
            f"Cast: {row.get('cast', 'Unknown')}\n"
            f"Character Names: {row.get('character_names', 'Unknown')}\n"
            f"Languages: {row.get('languages', 'Unknown')}\n"
            f"Countries: {row.get('countries', 'Unknown')}\n"
            f"Plot: {row.get('plot', 'Unknown')}\n"
            f"Plot Outline: {row.get('plot_outline', 'Unknown')}\n"
            f"Synopsis: {row.get('synopsis', 'Unknown')}"
        )
        metadata = {
            'movie_name': row.get('movie_name', "Unknown"),
            'year': row.get('year', "Unknown")
        }
        return full_text, metadata

class MovieInfoVectorStore(VectorStoreInitializer):
    def __init__(self, config):
        super().__init__(config)

    def process_movie(self, row):
        # Check for missing metadata
        if not row.get('movie_name') or not row.get('year'):
            self.logger.warning(f"Missing metadata for row: {row}")

        full_text = (
            f"Movie Name: {row.get('movie_name', 'Unknown')}\n"
            f"Synopsis: {row.get('synopsis', 'Unknown')}"
        )
        metadata = {
            'movie_name': row.get('movie_name', "Unknown"),
            'year': row.get('year', "Unknown")
        }
        return full_text, metadata
