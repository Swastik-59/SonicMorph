from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Dict, Any


class BaseCollector(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def discover(self, artist: str) -> Iterable[Dict[str, Any]]:
        """Yield candidate track metadata dicts for the given artist."""
        raise NotImplementedError()

    @abstractmethod
    def download(self, candidate: Dict[str, Any], out_dir: Path) -> Path:
        """Download the candidate and return the file path."""
        raise NotImplementedError()

