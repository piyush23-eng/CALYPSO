"""Tests for Verifiable Reward Functions."""

import pytest
from src.training.rewards import GATEVerifiableRewards


def test_mcq_reward_correct():
    rewards = GATEVerifiableRewards()
    completion = "Step 1 derivation...\n**Correct Option**: **(A)**"
    r = rewards.mcq_reward(completion, "A")
    assert r == 1.0


def test_mcq_reward_incorrect():
    rewards = GATEVerifiableRewards()
    completion = "Step 1 derivation...\n**Correct Option**: **(B)**"
    r = rewards.mcq_reward(completion, "A")
    assert r == 0.0


def test_msq_reward_full_match():
    rewards = GATEVerifiableRewards()
    completion = "Options (A) and (C) are undecidable.\n**Correct Options**: **(A, C)**"
    r = rewards.msq_reward(completion, ["A", "C"])
    assert r == 1.0


def test_msq_reward_partial_match():
    rewards = GATEVerifiableRewards()
    completion = "Option (A) is undecidable.\n**Correct Option**: **(A)**"
    r = rewards.msq_reward(completion, ["A", "C"])
    assert 0.0 < r < 1.0


def test_nat_reward_exact():
    rewards = GATEVerifiableRewards()
    completion = "Computation yields: **Numerical Answer**: **118.5**"
    r = rewards.nat_reward(completion, 118.5)
    assert r == 1.0


def test_format_reward():
    rewards = GATEVerifiableRewards()
    completion = (
        "### Phase 1: Conceptual Framework\nFormulas here\n"
        "### Phase 2: Step-by-Step Derivation\nProof steps here\n"
        "### Phase 3: Option Elimination\nEliminating options\n"
        "### Phase 4: Final Answer\n(A) is correct"
    )
    score = rewards.format_reward(completion)
    assert score == 1.0
