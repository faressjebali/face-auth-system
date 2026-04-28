"""Unit tests for security_monitor.py."""

import time

import pytest

from security_monitor import SecurityMonitor


@pytest.fixture
def monitor():
    # In-memory only (no state file) for isolated tests
    return SecurityMonitor(
        max_failed=3,
        lockout_sec=5,
        rate_window_sec=10,
        rate_max=3,
        state_file=None,
    )


# ---------------------------------------------------------------------------
# check / rate limiting
# ---------------------------------------------------------------------------

def test_first_attempt_allowed(monitor):
    allowed, reason = monitor.check("alice")
    assert allowed
    assert reason == ""


def test_rate_limit_blocks_after_max(monitor):
    for _ in range(3):
        monitor.check("alice")
    allowed, reason = monitor.check("alice")
    assert not allowed
    assert "Rate limit" in reason


def test_rate_window_expires(monitor):
    # Use a very short window
    mon = SecurityMonitor(rate_window_sec=0, rate_max=1, state_file=None)
    mon.check("bob")
    # Immediately check again — window is 0s so old timestamp is pruned
    allowed, _ = mon.check("bob")
    assert allowed


# ---------------------------------------------------------------------------
# is_locked_out
# ---------------------------------------------------------------------------

def test_not_locked_by_default(monitor):
    locked, remaining = monitor.is_locked_out("alice")
    assert not locked
    assert remaining == 0.0


def test_lockout_after_failures(monitor):
    monitor.record_failure("alice")
    monitor.record_failure("alice")
    monitor.record_failure("alice")  # 3rd → lockout triggers
    locked, remaining = monitor.is_locked_out("alice")
    assert locked
    assert remaining > 0.0


def test_lockout_blocks_check(monitor):
    for _ in range(3):
        monitor.record_failure("alice")
    allowed, reason = monitor.check("alice")
    assert not allowed
    assert "locked" in reason.lower()


# ---------------------------------------------------------------------------
# record_success / reset
# ---------------------------------------------------------------------------

def test_success_resets_failure_counter(monitor):
    monitor.record_failure("alice")
    monitor.record_failure("alice")
    monitor.record_success("alice")
    # Should NOT be locked now (only 2 failures, then success reset counter)
    monitor.record_failure("alice")
    monitor.record_failure("alice")
    locked, _ = monitor.is_locked_out("alice")
    assert not locked


def test_reset_user_clears_all(monitor):
    for _ in range(3):
        monitor.record_failure("alice")
    monitor.reset_user("alice")
    locked, _ = monitor.is_locked_out("alice")
    assert not locked


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------

def test_users_are_isolated(monitor):
    for _ in range(3):
        monitor.record_failure("alice")
    locked_a, _ = monitor.is_locked_out("alice")
    locked_b, _ = monitor.is_locked_out("bob")
    assert locked_a
    assert not locked_b


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_state_persisted_and_loaded(tmp_path):
    state_file = tmp_path / "state.json"
    m1 = SecurityMonitor(max_failed=2, lockout_sec=60, rate_max=10, state_file=state_file)
    m1.record_failure("alice")
    m1.record_failure("alice")  # triggers lockout

    m2 = SecurityMonitor(max_failed=2, lockout_sec=60, rate_max=10, state_file=state_file)
    locked, remaining = m2.is_locked_out("alice")
    assert locked
    assert remaining > 0.0
