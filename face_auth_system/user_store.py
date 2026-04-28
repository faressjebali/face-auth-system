"""
User profile storage.

Maps UUID user_id → {first_name, surname, dob} in a JSON file.
No biometric data is stored here — embeddings live in the gallery pickle.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Optional

import config

_USERS_FILE = config.DATA_DIR / "users.json"


class UserStore:
    """
    Simple JSON-backed user profile database.

    Parameters
    ----------
    path : Path
        Path to the JSON file (default: ``data/users.json``).
    """

    def __init__(self, path: Path = _USERS_FILE) -> None:
        self.path = Path(path)
        self._data: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    def create(self, first_name: str, surname: str, dob: str) -> str:
        """
        Create a new user profile and return the generated user_id.

        Parameters
        ----------
        first_name, surname : str
        dob : str
            Date of birth as a string (e.g. ``"1990-05-21"``).

        Returns
        -------
        str
            UUID4 user identifier.
        """
        uid = str(uuid.uuid4())
        self._data[uid] = {
            "first_name": first_name.strip(),
            "surname": surname.strip(),
            "dob": dob.strip(),
        }
        self._save()
        return uid

    # ------------------------------------------------------------------
    def get(self, user_id: str) -> Optional[dict]:
        """Return profile dict or ``None`` if not found."""
        return self._data.get(user_id)

    def display_name(self, user_id: str) -> str:
        """Return ``"First Last"`` for display, or ``"Unknown"``."""
        u = self.get(user_id)
        return f"{u['first_name']} {u['surname']}" if u else "Unknown"

    def delete(self, user_id: str) -> bool:
        """Remove a user profile. Returns ``True`` if it existed."""
        if user_id in self._data:
            del self._data[user_id]
            self._save()
            return True
        return False

    def list_ids(self) -> list:
        return list(self._data.keys())

    # ------------------------------------------------------------------
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
