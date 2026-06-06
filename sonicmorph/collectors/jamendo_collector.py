from .base_collector import BaseCollector
from pathlib import Path
from typing import Iterable, Dict, Any


class JamendoCollector(BaseCollector):
    def discover(self, artist: str) -> Iterable[Dict[str, Any]]:
        return []

    def download(self, candidate: Dict[str, Any], out_dir: Path) -> Path:
        raise NotImplementedError()
