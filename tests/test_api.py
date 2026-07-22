"""Testes da API com TestClient — mockando o pipeline para não gastar tokens."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api.main as main
from core.optimizer import OptimizationResult


client = TestClient(main.app)


def _fake_result(prompt: str, **kwargs) -> OptimizationResult:
    return OptimizationResult(
        original_prompt=prompt,
        optimized_prompt="Summarize:",
        original_tokens=50,
        optimized_tokens=5,
        tokens_saved=45,
        reduction_pct=90.0,
        intention_score=0.9,
        output_similarity=1.0,
        original_output="",
        optimized_output="",
        techniques_applied=["Imperative form"],
        grade="A",
        provider=kwargs.get("provider", "openai"),
        model=kwargs.get("model") or "gpt-4o-mini",
    )


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "openai_configured" in body and "anthropic_configured" in body


def test_techniques():
    r = client.get("/techniques")
    assert r.status_code == 200
    assert len(r.json()["techniques"]) == 7


def test_models():
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "openai" in body["providers"] and "anthropic" in body["providers"]


def test_optimize_mockado(monkeypatch):
    monkeypatch.setattr(main, "run_optimization", _fake_result)
    monkeypatch.setattr(main, "estimate_cost", lambda tokens, model: 0.0001)

    r = client.post("/optimize", json={"prompt": "please summarize this text for me", "test_outputs": False})
    assert r.status_code == 200
    body = r.json()
    assert body["grade"] == "A"
    assert body["tokens_saved"] == 45
    assert body["provider"] == "openai"


def test_optimize_validacao_prompt_curto():
    r = client.post("/optimize", json={"prompt": "curto"})
    assert r.status_code == 422  # min_length=10


def test_batch_mockado(monkeypatch):
    monkeypatch.setattr(main, "run_optimization", _fake_result)
    monkeypatch.setattr(main, "estimate_cost", lambda tokens, model: 0.0001)

    r = client.post("/batch", json={"prompts": ["please summarize this text", "kindly review this code"]})
    assert r.status_code == 200
    body = r.json()
    assert body["total_prompts"] == 2
    assert body["total_tokens_saved"] == 90
