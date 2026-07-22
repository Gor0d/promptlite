"""Testes unitários puros — sem chamadas de API (sem custo de tokens)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimizer import count_tokens, estimate_cost, compute_grade
from core.providers import _extract_json, get_provider, PROVIDER_MODELS, DEFAULT_MODEL
from data.benchmark_prompts import get_prompt_by_id, get_prompts_by_domain


def test_count_tokens_basico():
    assert count_tokens("Hello world") > 0
    assert count_tokens("") == 0


def test_count_tokens_modelo_desconhecido_usa_fallback():
    # Não deve lançar KeyError para modelos não mapeados no tiktoken.
    assert count_tokens("teste", model="claude-opus-4-8") > 0


def test_estimate_cost_por_modelo():
    assert estimate_cost(1_000_000, "gpt-4o-mini") == 0.15
    assert estimate_cost(1_000_000, "claude-opus-4-8") == 5.0
    # Modelo desconhecido cai no default.
    assert estimate_cost(1_000_000, "modelo-x") == 0.15


def test_compute_grade_output_ruim_e_D():
    assert compute_grade(80, 0.9, 0.5) == "D"  # output_similarity < 0.75


def test_compute_grade_excelente_e_A():
    assert compute_grade(80, 0.95, 0.98) == "A"


def test_compute_grade_faixas():
    assert compute_grade(0, 0.5, 0.8) in {"C", "D"}


def test_extract_json_tolerante_a_cercas():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('lixo {"b": 2} mais lixo') == {"b": 2}
    assert _extract_json("sem json") == {}


def test_provider_invalido_lanca_erro():
    import pytest
    with pytest.raises(ValueError):
        get_provider("gemini")


def test_tabela_de_modelos_consistente():
    for prov, default in DEFAULT_MODEL.items():
        assert default in PROVIDER_MODELS[prov]


def test_benchmark_helpers():
    p = get_prompt_by_id("healthcare_001")
    assert p is not None and p["domain"] == "healthcare"
    assert get_prompt_by_id("inexistente") is None
    assert len(get_prompts_by_domain("healthcare")) >= 1
