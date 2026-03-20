"""
PromptLite — API REST com FastAPI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="PromptLite API",
    description="Otimizador de prompts para redução de tokens com preservação de intenção",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=10000,
                        example="Please help me summarize this text in a clear and concise way...")
    test_outputs: bool = Field(True, description="Testar equivalência de outputs no LLM (usa mais tokens)")
    model: str = Field("gpt-4o-mini", description="Modelo LLM para testes")


class OptimizeResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    reduction_pct: float
    intention_score: float
    output_similarity: float
    original_output: str
    optimized_output: str
    techniques_applied: list
    grade: str
    cost_saved_usd: float


class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., max_length=10)
    test_outputs: bool = False


@app.get("/")
async def root():
    return {
        "name": "PromptLite",
        "description": "Otimizador de prompts — reduz tokens, preserva intenção",
        "author": "Emerson Guimarães — github.com/Gor0d",
        "stack": ["FastAPI", "OpenAI", "tiktoken", "scikit-learn"],
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "healthy",
        "openai_configured": bool(api_key),
        "version": "1.0.0"
    }


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Endpoint principal: otimiza um prompt e retorna métricas completas.

    Pipeline:
    1. Conta tokens originais com tiktoken
    2. Extrai intenção core com LLM
    3. Otimiza removendo redundâncias
    4. Conta tokens otimizados
    5. Testa outputs nos dois prompts (opcional)
    6. Calcula similaridade semântica via embeddings
    7. Computa grade final (A/B/C/D)
    """
    try:
        from core.optimizer import run_optimization, estimate_cost
        result = run_optimization(request.prompt, test_outputs=request.test_outputs)
        cost_saved = estimate_cost(result.tokens_saved, request.model)

        return OptimizeResponse(
            original_prompt=result.original_prompt,
            optimized_prompt=result.optimized_prompt,
            original_tokens=result.original_tokens,
            optimized_tokens=result.optimized_tokens,
            tokens_saved=result.tokens_saved,
            reduction_pct=result.reduction_pct,
            intention_score=result.intention_score,
            output_similarity=result.output_similarity,
            original_output=result.original_output,
            optimized_output=result.optimized_output,
            techniques_applied=result.techniques_applied,
            grade=result.grade,
            cost_saved_usd=round(cost_saved, 6)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_optimize(request: BatchRequest):
    """
    Otimiza múltiplos prompts de uma vez.
    Útil para processar datasets de treinamento de LLMs.
    """
    from core.optimizer import run_optimization, estimate_cost

    results = []
    total_saved = 0

    for i, prompt in enumerate(request.prompts):
        try:
            result = run_optimization(prompt, test_outputs=request.test_outputs)
            total_saved += result.tokens_saved
            results.append({
                "index": i,
                "status": "success",
                "original_tokens": result.original_tokens,
                "optimized_tokens": result.optimized_tokens,
                "tokens_saved": result.tokens_saved,
                "reduction_pct": result.reduction_pct,
                "grade": result.grade,
                "optimized_prompt": result.optimized_prompt
            })
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})

    return {
        "total_prompts": len(request.prompts),
        "total_tokens_saved": total_saved,
        "results": results
    }


@app.get("/benchmark")
async def run_benchmark():
    """Executa otimização nos prompts do dataset de benchmark (sem testar outputs)."""
    from core.optimizer import run_optimization
    from data.benchmark_prompts import BENCHMARK_PROMPTS

    results = []
    for p in BENCHMARK_PROMPTS:
        try:
            result = run_optimization(p["prompt"], test_outputs=False)
            results.append({
                "id": p["id"],
                "domain": p["domain"],
                "description": p["description"],
                "original_tokens": result.original_tokens,
                "optimized_tokens": result.optimized_tokens,
                "reduction_pct": result.reduction_pct,
                "intention_score": result.intention_score,
                "grade": result.grade
            })
        except Exception as e:
            results.append({"id": p["id"], "error": str(e)})

    avg_reduction = sum(r.get("reduction_pct", 0) for r in results) / len(results)
    return {"results": results, "avg_reduction_pct": round(avg_reduction, 1)}


@app.get("/techniques")
async def list_techniques():
    """Lista as técnicas de otimização implementadas."""
    return {
        "techniques": [
            {"name": "Remove filler phrases", "example": "'Please help me' → removed"},
            {"name": "Imperative form", "example": "'Can you summarize' → 'Summarize'"},
            {"name": "Remove AI acknowledgments", "example": "'As an AI' → removed"},
            {"name": "Consolidate instructions", "example": "3 similar rules → 1 rule"},
            {"name": "Remove obvious context", "example": "'It's important that' → removed"},
            {"name": "Compress examples", "example": "3 examples → 1 best example"},
            {"name": "Structured format", "example": "prose requirements → bullet list"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
