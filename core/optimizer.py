"""
PromptLite — Core Optimizer
Analisa, otimiza e avalia prompts para redução de tokens com preservação de intenção.
"""

import os
import json
import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from typing import Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    techniques_applied: list[str]
    grade: str                    # A, B, C, D


# ─────────────────────────────────────────────
# 1. TOKENIZAÇÃO
# ─────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Conta tokens usando tiktoken — biblioteca oficial da OpenAI."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estima custo em USD baseado nos tokens."""
    # gpt-4o-mini: $0.15 per 1M input tokens
    rates = {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "gpt-4o": 2.50 / 1_000_000,
        "gpt-4-turbo": 10.00 / 1_000_000,
    }
    rate = rates.get(model, 0.15 / 1_000_000)
    return tokens * rate


# ─────────────────────────────────────────────
# 2. ANÁLISE DE INTENÇÃO
# ─────────────────────────────────────────────

def extract_intention(prompt: str) -> dict:
    """
    Usa LLM para extrair a intenção core do prompt.
    Retorna: task, context, constraints, output_format
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """Analyze the given prompt and extract its core components.
Return ONLY a JSON object with these fields:
- task: the main task being requested (1 sentence)
- context: essential context provided (1-2 sentences or null)
- constraints: any constraints or requirements (list or null)  
- output_format: expected output format (1 sentence or null)
- redundancies: list of redundant or verbose parts that could be removed
- techniques_to_apply: list of optimization techniques applicable

Be concise and precise."""
            },
            {"role": "user", "content": f"Prompt to analyze:\n\n{prompt}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


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


def optimize_prompt(prompt: str) -> tuple[str, list[str]]:
    """
    Otimiza o prompt para redução máxima de tokens.
    Retorna: (prompt_otimizado, técnicas_aplicadas)
    """
    # Primeiro extrai a intenção para guiar a otimização
    intention = extract_intention(prompt)
    techniques = intention.get("techniques_to_apply", [])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": OPTIMIZATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Optimize this prompt for maximum token reduction:

ORIGINAL PROMPT:
{prompt}

IDENTIFIED REDUNDANCIES:
{json.dumps(intention.get('redundancies', []), indent=2)}

Return ONLY the optimized prompt."""
            }
        ]
    )

    optimized = response.choices[0].message.content.strip()
    return optimized, techniques


# ─────────────────────────────────────────────
# 4. TESTE DE EQUIVALÊNCIA DE OUTPUT
# ─────────────────────────────────────────────

def get_llm_output(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Executa o prompt no LLM e retorna o output."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def get_embedding(text: str) -> list[float]:
    """Gera embedding para calcular similaridade semântica."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]  # limite de segurança
    )
    return response.data[0].embedding


def compute_similarity(text1: str, text2: str) -> float:
    """
    Calcula similaridade semântica entre dois textos via embeddings.
    1.0 = idênticos semanticamente | 0.0 = completamente diferentes
    """
    emb1 = np.array(get_embedding(text1)).reshape(1, -1)
    emb2 = np.array(get_embedding(text2)).reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)


# ─────────────────────────────────────────────
# 5. SCORE E GRADE
# ─────────────────────────────────────────────

def compute_intention_score(original: str, optimized: str) -> float:
    """
    Score de preservação de intenção baseado em similaridade
    semântica dos próprios prompts.
    """
    return compute_similarity(original, optimized)


def compute_grade(reduction_pct: float, intention_score: float,
                  output_similarity: float) -> str:
    """
    Grade composta: redução de tokens + preservação de intenção + equivalência de output.

    A: excelente — reduziu muito e preservou perfeitamente
    B: bom — reduziu bem com pequena perda de fidelidade
    C: aceitável — reduziu pouco ou perdeu alguma intenção
    D: problemático — output mudou significativamente
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

def run_optimization(prompt: str, test_outputs: bool = True) -> OptimizationResult:
    """
    Pipeline completo de otimização:
    1. Conta tokens originais
    2. Otimiza o prompt
    3. Conta tokens otimizados
    4. Testa outputs nos dois prompts
    5. Calcula similaridade semântica
    6. Computa score e grade
    """
    print(f"[1/5] Contando tokens originais...")
    original_tokens = count_tokens(prompt)

    print(f"[2/5] Otimizando prompt...")
    optimized_prompt, techniques = optimize_prompt(prompt)
    optimized_tokens = count_tokens(optimized_prompt)

    tokens_saved = original_tokens - optimized_tokens
    reduction_pct = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0

    original_output = ""
    optimized_output = ""
    output_similarity = 1.0

    if test_outputs:
        print(f"[3/5] Testando output original...")
        original_output = get_llm_output(prompt)

        print(f"[4/5] Testando output otimizado...")
        optimized_output = get_llm_output(optimized_prompt)

        print(f"[5/5] Calculando similaridade...")
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
        grade=grade
    )
