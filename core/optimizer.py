"""
PromptLite — Core Optimizer
Analisa, otimiza e avalia prompts para redução de tokens com preservação de intenção.
Agnóstico de provider: funciona com OpenAI e Anthropic (Claude).
"""

import json
from dataclasses import dataclass

from dotenv import load_dotenv

from core.providers import (
    LLMProvider,
    get_provider,
    embed_similarity,
    PRICING_PER_INPUT_TOKEN,
)

# Carrega .env logo no import para que OPENAI_API_KEY / ANTHROPIC_API_KEY
# estejam disponíveis antes de qualquer client ser instanciado.
load_dotenv()


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    reduction_pct: float
    intention_score: float        # 0-1: quão bem a intenção foi preservada
    output_similarity: float      # 0-1: similaridade dos outputs do LLM
    original_output: str
    optimized_output: str
    techniques_applied: list
    grade: str                    # A, B, C, D
    provider: str = "openai"
    model: str = ""


# ─────────────────────────────────────────────
# 1. TOKENIZAÇÃO / CUSTO
# ─────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Conta tokens usando o tokenizador apropriado ao modelo."""
    from core.providers import _tiktoken_count
    return _tiktoken_count(text, model)


def estimate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estima custo em USD (preço de input) para um dado modelo."""
    rate = PRICING_PER_INPUT_TOKEN.get(model, 0.15 / 1_000_000)
    return tokens * rate


# ─────────────────────────────────────────────
# 2. ANÁLISE DE INTENÇÃO
# ─────────────────────────────────────────────

INTENTION_SYSTEM_PROMPT = """Analyze the given prompt and extract its core components.
Return ONLY a JSON object with these fields:
- task: the main task being requested (1 sentence)
- context: essential context provided (1-2 sentences or null)
- constraints: any constraints or requirements (list or null)
- output_format: expected output format (1 sentence or null)
- redundancies: list of redundant or verbose parts that could be removed
- techniques_to_apply: list of optimization techniques applicable

Be concise and precise."""


def extract_intention(prompt: str, provider: LLMProvider) -> dict:
    """Usa o LLM para extrair a intenção core do prompt (com fallback seguro)."""
    return provider.chat_json(INTENTION_SYSTEM_PROMPT, f"Prompt to analyze:\n\n{prompt}")


# ─────────────────────────────────────────────
# 3. OTIMIZAÇÃO DO PROMPT
# ─────────────────────────────────────────────

OPTIMIZATION_SYSTEM_PROMPT = """You are PromptLite, an expert prompt engineer specialized in token optimization.

Your goal: rewrite prompts to be as concise as possible while PERFECTLY preserving the original intention, constraints and expected output format.

Techniques to apply:
1. REMOVE redundant phrases ("Please", "I would like you to", "As an AI", "It's important that")
2. CONSOLIDATE repeated instructions into single clear directives
3. USE imperative form ("Summarize" instead of "Can you summarize")
4. ELIMINATE filler words and unnecessary context
5. COMPRESS examples — keep only the most illustrative one if multiple exist
6. USE structured format (bullet points) instead of prose when listing requirements
7. REMOVE obvious statements the model already knows

Rules:
- NEVER remove essential constraints, context or output requirements
- NEVER change the meaning or intent
- NEVER add new instructions not in the original
- Output ONLY the optimized prompt, nothing else
- Write in the same language as the input prompt"""


def optimize_prompt(prompt: str, provider: LLMProvider) -> tuple[str, list]:
    """Otimiza o prompt. Retorna (prompt_otimizado, técnicas_aplicadas)."""
    intention = extract_intention(prompt, provider)
    techniques = intention.get("techniques_to_apply", []) or []

    user = f"""Optimize this prompt for maximum token reduction:

ORIGINAL PROMPT:
{prompt}

IDENTIFIED REDUNDANCIES:
{json.dumps(intention.get('redundancies', []), indent=2, ensure_ascii=False)}

Return ONLY the optimized prompt."""

    optimized = provider.chat(OPTIMIZATION_SYSTEM_PROMPT, user).strip()
    return optimized, techniques


# ─────────────────────────────────────────────
# 4. TESTE DE EQUIVALÊNCIA DE OUTPUT
# ─────────────────────────────────────────────

def get_llm_output(prompt: str, provider: LLMProvider) -> str:
    """Executa o prompt no LLM e retorna o output."""
    return provider.chat("", prompt) if provider.name == "openai" else provider.chat(
        "You are a helpful assistant.", prompt
    )


def compute_similarity(text1: str, text2: str) -> float:
    """Similaridade semântica entre dois textos (via embeddings ou fallback)."""
    return embed_similarity(text1, text2)


# ─────────────────────────────────────────────
# 5. SCORE E GRADE
# ─────────────────────────────────────────────

def compute_intention_score(original: str, optimized: str) -> float:
    """Preservação de intenção via similaridade semântica dos prompts."""
    return compute_similarity(original, optimized)


def compute_grade(reduction_pct: float, intention_score: float,
                  output_similarity: float) -> str:
    """
    Grade composta: redução de tokens + preservação de intenção + equivalência de output.
    """
    if output_similarity < 0.75:
        return "D"  # output mudou demais — não usar

    score = (reduction_pct / 100 * 0.4) + (intention_score * 0.3) + (output_similarity * 0.3)

    if score >= 0.70:
        return "A"
    elif score >= 0.55:
        return "B"
    elif score >= 0.40:
        return "C"
    else:
        return "D"


# ─────────────────────────────────────────────
# 6. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def run_optimization(prompt: str, test_outputs: bool = True,
                     provider: str = "openai", model: str = None) -> OptimizationResult:
    """
    Pipeline completo de otimização, parametrizado por provider e modelo.
    """
    llm = get_provider(provider, model)
    model = llm.model

    original_tokens = llm.count_tokens(prompt)

    optimized_prompt, techniques = optimize_prompt(prompt, llm)
    optimized_tokens = llm.count_tokens(optimized_prompt)

    tokens_saved = original_tokens - optimized_tokens
    reduction_pct = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0

    original_output = ""
    optimized_output = ""
    output_similarity = 1.0

    if test_outputs:
        original_output = get_llm_output(prompt, llm)
        optimized_output = get_llm_output(optimized_prompt, llm)
        output_similarity = compute_similarity(original_output, optimized_output)

    intention_score = compute_intention_score(prompt, optimized_prompt)
    grade = compute_grade(reduction_pct, intention_score, output_similarity)

    return OptimizationResult(
        original_prompt=prompt,
        optimized_prompt=optimized_prompt,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        tokens_saved=tokens_saved,
        reduction_pct=round(reduction_pct, 1),
        intention_score=round(intention_score, 3),
        output_similarity=round(output_similarity, 3),
        original_output=original_output,
        optimized_output=optimized_output,
        techniques_applied=techniques,
        grade=grade,
        provider=llm.name,
        model=model,
    )
