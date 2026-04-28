"""Unit tests for vault.py (AES-256-GCM encrypted embedding store)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vault import EmbeddingVault


@pytest.fixture
def vault(tmp_path):
    return EmbeddingVault(vault_dir=tmp_path)


@pytest.fixture
def embedding():
    rng = np.random.default_rng(42)
    emb = rng.random(256).astype(np.float32)
    return emb / np.linalg.norm(emb)


def test_store_creates_file(vault, embedding, tmp_path):
    vault.store("alice", embedding, "password123")
    assert len(list(tmp_path.glob("*.vault"))) == 1


def test_retrieve_roundtrip(vault, embedding):
    vault.store("alice", embedding, "secret")
    recovered = vault.retrieve("alice", "secret")
    assert recovered is not None
    np.testing.assert_allclose(recovered, embedding, rtol=1e-5)


def test_wrong_password_returns_none(vault, embedding):
    vault.store("alice", embedding, "correct_pw")
    recovered = vault.retrieve("alice", "wrong_pw")
    assert recovered is None


def test_missing_user_returns_none(vault):
    assert vault.retrieve("nobody", "pw") is None


def test_delete(vault, embedding):
    vault.store("alice", embedding, "pw")
    assert vault.exists("alice")
    assert vault.delete("alice")
    assert not vault.exists("alice")


def test_delete_nonexistent_returns_false(vault):
    assert not vault.delete("ghost")


def test_verify_correct(vault, embedding):
    # Store the embedding itself; cosine similarity = 1.0
    vault.store("alice", embedding, "pw")
    match, sim = vault.verify("alice", embedding, "pw")
    assert sim > 0.9


def test_verify_wrong_password(vault, embedding):
    vault.store("alice", embedding, "correct")
    match, sim = vault.verify("alice", embedding, "wrong")
    assert not match
    assert sim == 0.0


def test_different_users_isolated(vault, embedding):
    bob_emb = np.flip(embedding).copy()
    vault.store("alice", embedding, "pw")
    vault.store("bob", bob_emb, "pw")
    alice_rec = vault.retrieve("alice", "pw")
    bob_rec = vault.retrieve("bob", "pw")
    np.testing.assert_allclose(alice_rec, embedding, rtol=1e-5)
    np.testing.assert_allclose(bob_rec, bob_emb, rtol=1e-5)


def test_list_users(vault, embedding):
    vault.store("alice", embedding, "pw")
    vault.store("bob", embedding, "pw")
    users = vault.list_users()
    assert len(users) == 2
