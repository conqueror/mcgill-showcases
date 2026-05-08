"""Project configuration constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CORPUS_PATH = RAW_DIR / "research_corpus.csv"
QUERY_PATH = RAW_DIR / "research_queries.json"

DEFAULT_RANDOM_STATE = 7
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QA_MODEL = "deepset/roberta-base-squad2"
DEFAULT_SUMMARY_MODEL = "google/flan-t5-base"
