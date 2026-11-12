from __future__ import annotations

import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


class MongoDatabase:
    """
    Small MongoDB Atlas adapter.

    It intentionally exposes __getitem__ so that the existing
    PatientRepository can work with it exactly like InMemoryDatabase.

    Example:
        database["patients"]
        database["observations"]
    """

    def __init__(
        self,
        uri: str | None = None,
        database_name: str | None = None,
    ) -> None:

        load_dotenv()

        self.uri = uri or os.getenv("MONGODB_URI")

        self.database_name = (
            database_name
            or os.getenv(
                "MONGODB_DATABASE",
                "multiagent_cdss",
            )
        )

        if not self.uri:
            raise ValueError(
                "MONGODB_URI is missing. "
                "Add it to your .env file."
            )

        self.client = MongoClient(
            self.uri,
            serverSelectionTimeoutMS=10000,
        )

        self.database: Database = self.client[
            self.database_name
        ]

    def ping(self) -> bool:
        """
        Verify MongoDB Atlas connection.
        """

        result = self.client.admin.command("ping")

        return result.get("ok") == 1.0

    def __getitem__(self, name: str):
        """
        Makes MongoDatabase compatible with PatientRepository.

        Example:
            database["patients"]
        """

        return self.database[name]

    def close(self) -> None:
        self.client.close()