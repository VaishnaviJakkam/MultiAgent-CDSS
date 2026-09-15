from __future__ import annotations

import os

from src.database.connection import _load_environment, get_client


def _safe_category(exc: Exception) -> str:
    try:
        from pymongo.errors import (
            ConfigurationError,
            InvalidURI,
            OperationFailure,
            ServerSelectionTimeoutError,
            PyMongoError,
        )
    except ImportError:
        return "other connection error (pymongo is not installed)"

    if isinstance(exc, InvalidURI):
        return "invalid URI"
    if isinstance(exc, OperationFailure):
        return "authentication failure"
    if isinstance(exc, ServerSelectionTimeoutError):
        return "DNS/server selection failure or network/timeout failure"
    if isinstance(exc, ConfigurationError):
        return "TLS/SSL failure or invalid MongoDB configuration"
    if isinstance(exc, PyMongoError):
        return "MongoDB ping failure"
    return "other connection error"


def main() -> int:
    try:
        _load_environment()
    except Exception:
        print("MongoDB connection failed.")
        print("Category: other connection error")
        return 1

    uri_present = bool(os.getenv("MONGODB_URI"))
    database_present = bool(os.getenv("MONGODB_DATABASE"))

    if not uri_present:
        print("MongoDB connection failed.")
        print("Category: MONGODB_URI missing")
        return 1
    if not database_present:
        print("MongoDB connection failed.")
        print("Category: MONGODB_DATABASE missing")
        return 1

    client = None
    try:
        client = get_client()
        client.admin.command("ping")
    except Exception as exc:
        print("MongoDB connection failed.")
        print(f"Category: {_safe_category(exc)}")
        return 1
    finally:
        if client is not None:
            client.close()
    print("MongoDB connection successful.")
    print("Ping: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
