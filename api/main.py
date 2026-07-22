"""
PromptLite — API REST com FastAPI
Multi-provider (OpenAI + Anthropic/Claude).
"""

import os
import sys
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from core.optimizer import run_optimization, estimate_cost  # noqa: E402
from core.providers import PROVIDER_MODELS, DEFAULT_MODEL  # noqa: E402

app = FastAPI(
    title="PromptLite API",
    description="Otimizador de prompts para redução de tokens com preservação de intenção",
    version="2.0.0",
)

# CORS: por padrão liberado; restrinja via env PROMPTLITE_CORS_ORIGINS (lista separada por vírgula).
_origins = os.getenv("PROMPTLITE_CORS_ORIGINS", "*")
allow_origins = ["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

Provider = Literal["openai", "anthropic"]


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=10000,
                        examples=["Please help me summarize this text in a clear and concise way..."])
    test_outputs: bool = Field(True, description="Testar equivalência de outputs no LLM (usa mais tokens)")
    provider: Provider = Field("openai", description="Provider do LLM")
    model: str | None = Field(None, description="Modelo específico (default por provider se omitido)")


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
    provider: str
    model: str
    cost_saved_usd: float


class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., max_length=10)
    test_outputs: bool = False
    provider: Provider = "openai"
    model: str | None = None


@app.get("/")
async def root():
    return {
        "name": "PromptLite",
        "description": "Otimizador de prompts — reduz tokens, preserva intenção",
        "author": "Emerson Guimarães — github.com/Gor0d",
        "stack": ["FastAPI", "OpenAI", "Anthropic", "tiktoken", "scikit-learn"],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "version": "2.0.0",
    }


@app.get("/models")
async def list_models():
    """Lista os modelos disponíveis por provider e o default de cada um."""
    return {"providers": PROVIDER_MODELS, "defaults": DEFAULT_MODEL}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """Otimiza um prompt e retorna métricas completas."""
    try:
        result = await run_in_threadpool(
            run_optimization,
            request.prompt,
            test_outputs=request.test_outputs,
            provider=request.provider,
            model=request.model,
        )
        cost_saved = estimate_cost(result.tokens_saved, result.model)

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
            provider=result.provider,
            model=result.model,
            cost_saved_usd=round(cost_saved, 6),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_optimize(request: BatchRequest):
    """Otimiza múltiplos prompts. Útil para datasets de treinamento."""
    results = []
    total_saved = 0

    for i, prompt in enumerate(request.prompts):
        try:
            result = await run_in_threadpool(
                run_optimization,
                prompt,
                test_outputs=request.test_outputs,
                provider=request.provider,
                model=request.model,
            )
            total_saved += result.tokens_saved
            results.append({
                "index": i,
                "status": "success",
                "original_tokens": result.original_tokens,
                "optimized_tokens": result.optimized_tokens,
                "tokens_saved": result.tokens_saved,
                "reduction_pct": result.reduction_pct,
                "grade": result.grade,
                "optimized_prompt": result.optimized_prompt,
            })
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})

    return {
        "total_prompts": len(request.prompts),
        "total_tokens_saved": total_saved,
        "results": results,
    }


@app.get("/benchmark")
async def run_benchmark(provider: Provider = "openai", model: str | None = None):
    """Executa otimização nos prompts do dataset de benchmark (sem testar outputs)."""
    from data.benchmark_prompts import BENCHMARK_PROMPTS

    results = []
    for p in BENCHMARK_PROMPTS:
        try:
            result = await run_in_threadpool(
                run_optimization, p["prompt"], test_outputs=False,
                provider=provider, model=model,
            )
            results.append({
                "id": p["id"],
                "domain": p["domain"],
                "description": p["description"],
                "original_tokens": result.original_tokens,
                "optimized_tokens": result.optimized_tokens,
                "reduction_pct": result.reduction_pct,
                "intention_score": result.intention_score,
                "grade": result.grade,
            })
        except Exception as e:
            results.append({"id": p["id"], "error": str(e)})

    valid = [r for r in results if "error" not in r]
    avg_reduction = (sum(r["reduction_pct"] for r in valid) / len(valid)) if valid else 0
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
