"""
Authentication event logger.

Writes privacy-preserving JSON-lines log entries to data/logs/.
No plaintext PII is stored; user identifiers are SHA-256 hashed.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import config

# Module-level stdlib logger for internal diagnostics
_internal = logging.getLogger(__name__)


def _hash_user_id(user_id: str) -> str:
    """
    Return a truncated SHA-256 hex digest of *user_id*.

    Parameters
    ----------
    user_id : str
        Raw user identifier (PII).

    Returns
    -------
    str
        16-character hex digest (64-bit pseudonym).
    """
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _get_log_path() -> Path:
    """
    Return path to today's log file, creating parent directories if needed.

    Returns
    -------
    Path
        Absolute path of the current log file.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return config.LOGS_DIR / f"{config.LOG_FILENAME_PREFIX}_{date_str}.jsonl"


def log_auth_event(
    user_id: str,
    result: str,
    confidence: float,
    classical_score: Optional[float] = None,
    deep_score: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a single authentication event to the daily JSON-lines log.

    Parameters
    ----------
    user_id : str
        Raw user identifier; stored only as a pseudonymous hash.
    result : str
        Authentication outcome, e.g. ``"GRANTED"`` or ``"DENIED"``.
    confidence : float
        Fused confidence score in [0, 1].
    classical_score : float, optional
        Score from the classical pipeline.
    deep_score : float, optional
        Score from the deep-learning pipeline.
    extra : dict, optional
        Additional metadata (must be JSON-serialisable, no PII).

    Returns
    -------
    None
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "user_hash": _hash_user_id(user_id),
        "result": result,
        "confidence": round(float(confidence), 4),
    }
    if classical_score is not None:
        record["classical_score"] = round(float(classical_score), 4)
    if deep_score is not None:
        record["deep_score"] = round(float(deep_score), 4)
    if extra:
        record["extra"] = extra

    log_path = _get_log_path()
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        _internal.error("Failed to write auth log entry: %s", exc)


def log_security_alert(
    user_id: str,
    alert_type: str,
    detail: Optional[str] = None,
) -> None:
    """
    Append a security-anomaly alert to the daily log.

    Parameters
    ----------
    user_id : str
        Raw user identifier (hashed before storage).
    alert_type : str
        Short label such as ``"LOCKOUT"`` or ``"RATE_LIMIT"``.
    detail : str, optional
        Human-readable description (must not contain PII).

    Returns
    -------
    None
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "user_hash": _hash_user_id(user_id),
        "event": "SECURITY_ALERT",
        "alert_type": alert_type,
    }
    if detail:
        record["detail"] = detail

    log_path = _get_log_path()
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        _internal.error("Failed to write security alert: %s", exc)


def read_logs(date_str: Optional[str] = None) -> list:
    """
    Read and parse log entries for a given date.

    Parameters
    ----------
    date_str : str, optional
        Date in ``YYYY-MM-DD`` format.  Defaults to today (UTC).

    Returns
    -------
    list of dict
        Parsed log records.
    """
    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    log_path = config.LOGS_DIR / f"{config.LOG_FILENAME_PREFIX}_{date_str}.jsonl"
    if not log_path.exists():
        return []
    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        _internal.warning("Skipping malformed log line: %s", exc)
    except OSError as exc:
        _internal.error("Cannot read log file %s: %s", log_path, exc)
    return records
