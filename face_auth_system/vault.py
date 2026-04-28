"""
Encrypted embedding vault using AES-256-GCM.

Each user's face template is encrypted with a per-user random salt and a
PBKDF2-derived key.  The master password is never stored on disk.

File format per user:  salt (16 B) | nonce (12 B) | AES-GCM ciphertext
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import config

logger = logging.getLogger(__name__)


def _derive_key(password: bytes, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=config.KEY_DERIVATION_ITERATIONS,
    )
    return kdf.derive(password)


class EmbeddingVault:
    """
    AES-256-GCM encrypted store for face embeddings / protected templates.

    Parameters
    ----------
    vault_dir : Path
        Directory where per-user ``.vault`` files are written.
    """

    def __init__(self, vault_dir: Path = config.VAULT_DIR) -> None:
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _user_path(self, user_id: str) -> Path:
        safe = hashlib.sha256(user_id.encode()).hexdigest()[:24]
        return self.vault_dir / f"{safe}.vault"

    # ------------------------------------------------------------------
    def store(
        self,
        user_id: str,
        embedding: np.ndarray,
        password: str,
    ) -> None:
        """Encrypt and persist *embedding* for *user_id*."""
        salt = os.urandom(config.VAULT_SALT_SIZE)
        nonce = os.urandom(config.VAULT_NONCE_SIZE)
        key = _derive_key(password.encode(), salt)

        payload = {
            "embedding": embedding.astype(np.float32).tobytes().hex(),
            "shape": list(embedding.shape),
        }
        plaintext = json.dumps(payload).encode()

        ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)

        with open(self._user_path(user_id), "wb") as f:
            f.write(salt + nonce + ciphertext)

        logger.info("Stored encrypted embedding for user '%s'.", user_id)

    # ------------------------------------------------------------------
    def retrieve(self, user_id: str, password: str) -> Optional[np.ndarray]:
        """Decrypt and return the stored embedding, or ``None`` on failure."""
        path = self._user_path(user_id)
        if not path.exists():
            logger.warning("No vault entry for user '%s'.", user_id)
            return None
        try:
            data = path.read_bytes()
            s = config.VAULT_SALT_SIZE
            n = config.VAULT_NONCE_SIZE
            salt, nonce, ct = data[:s], data[s:s + n], data[s + n:]
            key = _derive_key(password.encode(), salt)
            plaintext = AESGCM(key).decrypt(nonce, ct, None)
            payload = json.loads(plaintext.decode())
            raw = bytes.fromhex(payload["embedding"])
            return np.frombuffer(raw, dtype=np.float32).reshape(payload["shape"])
        except Exception as exc:
            logger.error("Vault retrieval failed for '%s': %s", user_id, exc)
            return None

    # ------------------------------------------------------------------
    def verify(
        self,
        user_id: str,
        probe: np.ndarray,
        password: str,
    ) -> Tuple[bool, float]:
        """
        Compare *probe* against the stored template via cosine similarity.

        Returns
        -------
        Tuple[bool, float]
            ``(is_match, similarity)`` using ``config.VAULT_COSINE_THRESHOLD``.
        """
        stored = self.retrieve(user_id, password)
        if stored is None:
            return False, 0.0
        from deep_model import cosine_similarity
        sim = cosine_similarity(probe.flatten(), stored.flatten())
        return sim >= config.VAULT_COSINE_THRESHOLD, float(sim)

    # ------------------------------------------------------------------
    def delete(self, user_id: str) -> bool:
        """Remove vault entry (supports template revocation)."""
        path = self._user_path(user_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted vault entry for user '%s'.", user_id)
            return True
        return False

    def exists(self, user_id: str) -> bool:
        return self._user_path(user_id).exists()

    def list_users(self) -> list:
        return [p.stem for p in self.vault_dir.glob("*.vault")]
