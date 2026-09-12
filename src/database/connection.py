from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_DATABASE_NAME = "multiagent_cdss"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def get_client(uri: str | None = None) -> Any:
    """Create a MongoDB client from MONGODB_URI without exposing credentials."""
    _load_environment()
    connection_uri = uri or os.getenv("MONGODB_URI")
    if not connection_uri:
        raise RuntimeError("MONGODB_URI is required to connect to MongoDB")
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pymongo is required for MongoDB connections") from exc
    return MongoClient(connection_uri)


def get_database(uri: str | None = None, database_name: str | None = None) -> Any:
    """Create a MongoDB database handle using MONGODB_URI or an explicit URI."""
    _load_environment()
    client = get_client(uri)
    return client[database_name or os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE_NAME)]
