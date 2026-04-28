"""
Security monitor: rate limiting and account lockout.

Enforces two policies:

  1. **Sliding-window rate limit** — at most N attempts in a rolling window.
  2. **Progressive lockout** — after M consecutive failures the account is
     locked for a configurable duration.

State is kept in memory and flushed to a JSON file so lockouts survive
process restarts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import config

logger = logging.getLogger(__name__)

_DEFAULT_STATE_FILE = config.DATA_DIR / "security_state.json"


class SecurityMonitor:
    """
    Per-user rate limiting and lockout enforcement.

    Parameters
    ----------
    max_failed : int
        Consecutive failures before lockout.
    lockout_sec : int
        Lockout duration in seconds.
    rate_window_sec : int
        Rolling time window for rate counting.
    rate_max : int
        Maximum attempts permitted within *rate_window_sec*.
    state_file : Path, optional
        JSON file for persisting state across restarts.
    """

    def __init__(
        self,
        max_failed: int = config.MAX_FAILED_ATTEMPTS,
        lockout_sec: int = config.LOCKOUT_DURATION_SEC,
        rate_window_sec: int = config.RATE_LIMIT_WINDOW_SEC,
        rate_max: int = config.RATE_LIMIT_MAX_ATTEMPTS,
        state_file: Optional[Path] = _DEFAULT_STATE_FILE,
    ) -> None:
        self.max_failed = max_failed
        self.lockout_sec = lockout_sec
        self.rate_window_sec = rate_window_sec
        self.rate_max = rate_max
        self.state_file = state_file
        # user_id → {"failed": int, "lockout_until": float, "attempts": [ts, ...]}
        self._state: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    def _record(self, user_id: str) -> dict:
        if user_id not in self._state:
            self._state[user_id] = {"failed": 0, "lockout_until": 0.0, "attempts": []}
        return self._state[user_id]

    # ------------------------------------------------------------------
    def is_locked_out(self, user_id: str) -> Tuple[bool, float]:
        """
        Returns
        -------
        Tuple[bool, float]
            ``(is_locked, seconds_remaining)``
        """
        st = self._record(user_id)
        now = time.time()
        if now < st["lockout_until"]:
            return True, st["lockout_until"] - now
        return False, 0.0

    def is_rate_limited(self, user_id: str) -> bool:
        st = self._record(user_id)
        now = time.time()
        st["attempts"] = [t for t in st["attempts"] if now - t < self.rate_window_sec]
        return len(st["attempts"]) >= self.rate_max

    # ------------------------------------------------------------------
    def check(self, user_id: str) -> Tuple[bool, str]:
        """
        Decide whether a new attempt is allowed.

        Side-effect: records the current timestamp in the sliding window.

        Returns
        -------
        Tuple[bool, str]
            ``(allowed, reason)`` — *reason* is ``""`` when allowed.
        """
        locked, remaining = self.is_locked_out(user_id)
        if locked:
            msg = f"Account locked for {remaining:.0f}s more."
            logger.warning("Lockout block: user='%s' remaining=%.0fs", user_id, remaining)
            return False, msg

        if self.is_rate_limited(user_id):
            msg = (
                f"Rate limit: max {self.rate_max} attempts "
                f"per {self.rate_window_sec}s."
            )
            logger.warning("Rate limit block: user='%s'", user_id)
            return False, msg

        self._record(user_id)["attempts"].append(time.time())
        return True, ""

    # ------------------------------------------------------------------
    def record_failure(self, user_id: str) -> None:
        """Increment failure counter; apply lockout when threshold is reached."""
        st = self._record(user_id)
        st["failed"] += 1
        logger.info(
            "Failed attempt %d/%d for user '%s'.",
            st["failed"], self.max_failed, user_id,
        )
        if st["failed"] >= self.max_failed:
            st["lockout_until"] = time.time() + self.lockout_sec
            st["failed"] = 0
            logger.warning(
                "Locked out user '%s' for %ds after %d failures.",
                user_id, self.lockout_sec, self.max_failed,
            )
        self._save()

    def record_success(self, user_id: str) -> None:
        """Reset failure counter on successful authentication."""
        self._record(user_id)["failed"] = 0
        self._save()

    def reset_user(self, user_id: str) -> None:
        """Admin reset: clear all security state for *user_id*."""
        self._state.pop(user_id, None)
        self._save()
        logger.info("Security state reset for user '%s'.", user_id)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self.state_file is None:
            return
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            logger.error("Could not save security state: %s", exc)

    def _load(self) -> None:
        if self.state_file is None or not Path(self.state_file).exists():
            return
        try:
            with open(self.state_file) as f:
                self._state = json.load(f)
        except Exception as exc:
            logger.warning("Could not load security state: %s", exc)
