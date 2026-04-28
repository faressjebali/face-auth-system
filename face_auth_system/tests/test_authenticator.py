"""Unit tests for authenticator.py."""

import pytest

from authenticator import AuthResult, Authenticator


@pytest.fixture
def auth():
    return Authenticator(classical_weight=0.4, deep_weight=0.6, threshold=0.6)


# ---------------------------------------------------------------------------
# fuse_scores
# ---------------------------------------------------------------------------

def test_fuse_scores_weighted(auth):
    # 0.4 * 1.0 + 0.6 * 1.0 = 1.0
    assert abs(auth.fuse_scores(1.0, 1.0) - 1.0) < 1e-6


def test_fuse_scores_all_zero(auth):
    assert auth.fuse_scores(0.0, 0.0) == 0.0


def test_fuse_scores_clamps_inputs(auth):
    # Values outside [0,1] are clipped
    score = auth.fuse_scores(-1.0, 2.0)
    assert 0.0 <= score <= 1.0


def test_fuse_scores_midpoint(auth):
    fused = auth.fuse_scores(0.0, 1.0)
    assert abs(fused - auth.deep_weight) < 1e-6


# ---------------------------------------------------------------------------
# authenticate — basic decisions
# ---------------------------------------------------------------------------

def test_granted_above_threshold(auth):
    result = auth.authenticate(1.0, 1.0)
    assert result.decision == "GRANTED"


def test_denied_below_threshold(auth):
    result = auth.authenticate(0.0, 0.0)
    assert result.decision == "DENIED"


def test_result_scores_stored(auth):
    result = auth.authenticate(0.7, 0.8)
    assert 0.0 <= result.combined_score <= 1.0
    assert result.classical_score == pytest.approx(0.7, abs=1e-3)
    assert result.deep_score == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# authenticate — identity consistency
# ---------------------------------------------------------------------------

def test_identity_mismatch_penalises_score(auth):
    result_no_claim = auth.authenticate(0.9, 0.9)
    result_mismatch = auth.authenticate(
        0.9, 0.9, claimed_identity="alice", deep_identity="bob"
    )
    assert result_mismatch.combined_score < result_no_claim.combined_score


def test_identity_match_no_penalty(auth):
    result = auth.authenticate(
        0.9, 0.9, claimed_identity="alice", deep_identity="alice"
    )
    assert result.decision == "GRANTED"


def test_identity_unknown_deep_no_penalty(auth):
    result = auth.authenticate(
        0.9, 0.9, claimed_identity="alice", deep_identity="unknown"
    )
    # No penalty when deep returns "unknown"
    assert result.decision == "GRANTED"


# ---------------------------------------------------------------------------
# update_threshold
# ---------------------------------------------------------------------------

def test_update_threshold(auth):
    auth.update_threshold(0.9)
    assert auth.threshold == pytest.approx(0.9)
    result = auth.authenticate(0.7, 0.7)
    assert result.decision == "DENIED"


def test_update_threshold_clamps():
    auth = Authenticator()
    auth.update_threshold(5.0)
    assert auth.threshold == 1.0
    auth.update_threshold(-1.0)
    assert auth.threshold == 0.0


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------

def test_weights_normalised():
    auth = Authenticator(classical_weight=1.0, deep_weight=3.0)
    assert abs(auth.classical_weight + auth.deep_weight - 1.0) < 1e-6


def test_zero_weights_raises():
    with pytest.raises(ValueError):
        Authenticator(classical_weight=0.0, deep_weight=0.0)
