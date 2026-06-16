from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATASET_DIR = ROOT / "dataset"


def load_yaml(name: str):
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    def __init__(self):
        self.artists = load_yaml("artists.yaml")
        self.pipeline = load_yaml("pipeline.yaml")
        self.sources = load_yaml("sources.yaml")


config = Config()
