from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FileConfig:
    file_name: str
    sep: str
    encoding: str
    columns: List[str]

@dataclass
class CornellDatasetConfig:
    folder_path: str
    output_path: str
    sample_fraction: float
    train_ratio: float
    test_ratio: float
    frequent_sample_ratio: int
    generate_new_questions: bool
    data_prune_enabled: bool
    files: Dict[str, FileConfig]

@dataclass
class MovieQADatasetConfig:
    folder_path: str
    output_path: str
    train_ratio: float
    test_ratio: float
    files: Dict[str, FileConfig]
