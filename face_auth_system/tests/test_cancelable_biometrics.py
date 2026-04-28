"""Unit tests for cancelable_biometrics.py."""

import numpy as np
import pytest

from cancelable_biometrics import CancelableBiometrics, _orthogonal_matrix, _seed_from_token


INPUT_DIM = 512
OUTPUT_DIM = 256


@pytest.fixture
def cb():
    return CancelableBiometrics(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)


@pytest.fixture
def embedding():
    rng = np.random.default_rng(0)
    e = rng.random(INPUT_DIM).astype(np.float32)
    return e / np.linalg.norm(e)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def test_seed_from_token_deterministic():
    assert _seed_from_token("abc") == _seed_from_token("abc")


def test_seed_from_token_different_tokens():
    assert _seed_from_token("abc") != _seed_from_token("xyz")


def test_matrix_shape():
    M = _orthogonal_matrix(42, INPUT_DIM, OUTPUT_DIM)
    assert M.shape == (OUTPUT_DIM, INPUT_DIM)


def test_matrix_rows_orthonormal():
    M = _orthogonal_matrix(7, INPUT_DIM, OUTPUT_DIM)
    gram = M @ M.T   # should be close to identity
    np.testing.assert_allclose(gram, np.eye(OUTPUT_DIM), atol=1e-4)


# ---------------------------------------------------------------------------
# CancelableBiometrics.protect
# ---------------------------------------------------------------------------

def test_protect_output_shape(cb, embedding):
    template = cb.protect(embedding, "token1")
    assert template.shape == (OUTPUT_DIM,)


def test_protect_binary(cb, embedding):
    template = cb.protect(embedding, "token1")
    assert set(np.unique(template)).issubset({0.0, 1.0})


def test_protect_same_token_same_result(cb, embedding):
    t1 = cb.protect(embedding, "tok")
    t2 = cb.protect(embedding, "tok")
    np.testing.assert_array_equal(t1, t2)


def test_protect_different_tokens_different_templates(cb, embedding):
    t1 = cb.protect(embedding, "token_A")
    t2 = cb.protect(embedding, "token_B")
    # With high probability, templates from different tokens are not identical
    assert not np.array_equal(t1, t2)


def test_protect_wrong_dim_raises(cb):
    bad = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError):
        cb.protect(bad, "tok")


# ---------------------------------------------------------------------------
# CancelableBiometrics.compare
# ---------------------------------------------------------------------------

def test_compare_identical_templates(cb, embedding):
    t = cb.protect(embedding, "tok")
    assert cb.compare(t, t) == 1.0


def test_compare_different_templates_range(cb, embedding):
    rng = np.random.default_rng(1)
    other = rng.random(INPUT_DIM).astype(np.float32)
    other /= np.linalg.norm(other)
    t1 = cb.protect(embedding, "tok")
    t2 = cb.protect(other, "tok")
    sim = cb.compare(t1, t2)
    assert 0.0 <= sim <= 1.0


def test_compare_shape_mismatch_raises(cb):
    with pytest.raises(ValueError):
        cb.compare(np.zeros(128), np.zeros(64))


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------

def test_revoke_clears_cache(cb, embedding):
    cb.protect(embedding, "old_token")
    assert "old_token" in cb._cache
    cb.revoke("old_token")
    assert "old_token" not in cb._cache


def test_revoke_nonexistent_token_silent(cb):
    cb.revoke("does_not_exist")  # must not raise
