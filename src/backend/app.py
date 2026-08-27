"""FastAPI Application for CALYPSO GATE-CS Doubt Solver Backend."""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.backend.engine import GGUFInferenceEngine
from src.backend.models import (
    ChatMessage,
    CompareRequest,
    CompareResponse,
    ImageSolveRequest,
    ImageSolveResponse,
    MultiTurnChatRequest,
    SolveRequest,
    SolveResponse,
    StructuredPhases,
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from src.tools.code_interpreter import CodeInterpreter
from src.utils.logger import setup_logger
from src.vision.ocr_extractor import VisionOCRExtractor

logger = setup_logger("app")

app = FastAPI(
    title="CALYPSO 2.0 — GATE-CS Reasoning API",
    description="Domain-Specialized Reasoning LLM & Multi-Turn Doubt Solver for GATE Computer Science.",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    """Serves the Calypso frontend directly at root with no-cache headers."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        content = index_file.read_text(encoding="utf-8")
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>Calypso Frontend loading...</h1>")


@app.api_route("/styles.css", methods=["GET", "HEAD"])
@app.api_route("/ui/styles.css", methods=["GET", "HEAD"])
async def serve_styles():
    """Serves styles.css directly with text/css MIME type."""
    css_file = frontend_dir / "styles.css"
    if css_file.exists():
        return Response(
            content=css_file.read_text(encoding="utf-8"),
            media_type="text/css",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(content="", media_type="text/css")


@app.api_route("/app.js", methods=["GET", "HEAD"])
@app.api_route("/ui/app.js", methods=["GET", "HEAD"])
async def serve_app_js():
    """Serves app.js directly with application/javascript MIME type."""
    js_file = frontend_dir / "app.js"
    if js_file.exists():
        return Response(
            content=js_file.read_text(encoding="utf-8"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(content="", media_type="application/javascript")


if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

# Initialize Inference Engine, Vision Ingestion, and Tool Executor
engine = GGUFInferenceEngine(
    model_path="models/gguf/gate-qwen-1.5b-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
)
ocr_extractor = VisionOCRExtractor()
code_interpreter = CodeInterpreter()


@app.get("/health")
async def health_check():
    """Returns backend health, engine state, tool readiness, and device profile."""
    return {
        "status": "healthy",
        "service": "CALYPSO-GATE-CS-Reasoning-Engine",
        "version": "2.0.0",
        "model_loaded": engine.llm is not None,
        "engine_type": "llama.cpp GGUF" if engine.llm is not None else "Specialized High-Fidelity Fallback",
        "quantization": "Q4_K_M",
        "context_window": engine.n_ctx,
        "threads": engine.n_threads,
        "features": {
            "multi_turn_chat": True,
            "vision_ocr": True,
            "code_interpreter": True,
            "verifiable_rl_rewards": True,
        },
    }


@app.post("/api/solve", response_model=SolveResponse)
async def solve_question(req: SolveRequest):
    """Generates complete structured reasoning solution for a GATE CS problem."""
    try:
        solution, answer, latency, tokens, tps, phases = engine.solve(
            subject=req.subject,
            topic=req.topic or "General",
            question_type=req.question_type,
            marks=req.marks,
            question=req.question,
            options=req.options,
            model_type=req.model_type,
        )

        model_name = "GATE-CS-Qwen-1.5B (Fine-Tuned)" if req.model_type == "finetuned" else "Qwen2.5-1.5B-Instruct (Base)"

        return SolveResponse(
            model_name=model_name,
            solution_markdown=solution,
            extracted_answer=answer,
            structured_phases=StructuredPhases(**phases),
            inference_latency_ms=latency,
            tokens_generated=tokens,
            tokens_per_second=tps,
            device="CPU (Q4_K_M Quantized)",
        )
    except Exception as e:
        logger.error(f"Error solving question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solve/stream")
async def solve_question_stream(req: SolveRequest):
    """Server-Sent Events (SSE) endpoint for token-by-token streaming generation."""
    try:
        def event_generator():
            for token in engine.solve_stream(
                subject=req.subject,
                topic=req.topic or "General",
                question_type=req.question_type,
                marks=req.marks,
                question=req.question,
                options=req.options,
                model_type=req.model_type,
            ):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solve/image", response_model=ImageSolveResponse)
async def solve_image_question(req: ImageSolveRequest):
    """Extracts question components from an uploaded image/screenshot and solves it."""
    try:
        extracted = ocr_extractor.process_image(req.image_data)
        
        subj = req.subject or extracted.get("subject", "Algorithms")
        top = req.topic or extracted.get("topic", "General")
        q_text = extracted.get("question", "")
        options = extracted.get("options", {})
        q_type = extracted.get("question_type", "MCQ")
        marks = extracted.get("marks", 2)

        solution, answer, latency, tokens, tps, phases = engine.solve(
            subject=subj,
            topic=top,
            question_type=q_type,
            marks=marks,
            question=q_text,
            options=options,
            model_type="finetuned",
        )

        solve_res = SolveResponse(
            model_name="GATE-CS-Qwen-1.5B (Vision-OCR Ingestion)",
            solution_markdown=solution,
            extracted_answer=answer,
            structured_phases=StructuredPhases(**phases),
            inference_latency_ms=latency,
            tokens_generated=tokens,
            tokens_per_second=tps,
            device="CPU (Q4_K_M Quantized)",
        )

        return ImageSolveResponse(
            extracted_data=extracted,
            solution=solve_res,
        )
    except Exception as e:
        logger.error(f"Error in image OCR solving: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_multiturn(req: MultiTurnChatRequest):
    """Multi-turn contextual follow-up doubt resolution."""
    try:
        messages_dicts = [m.model_dump() for m in req.messages]
        reply, latency, tokens, tps = engine.chat_multi_turn(
            messages=messages_dicts,
            subject=req.subject or "Algorithms",
            topic=req.topic or "General",
        )
        return {
            "reply": reply,
            "inference_latency_ms": latency,
            "tokens_generated": tokens,
            "tokens_per_second": tps,
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_multiturn_stream(req: MultiTurnChatRequest):
    """Streams multi-turn chat response token by token via SSE."""
    try:
        messages_dicts = [m.model_dump() for m in req.messages]

        def event_generator():
            for token in engine.chat_stream(
                messages=messages_dicts,
                subject=req.subject or "Algorithms",
                topic=req.topic or "General",
            ):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Chat streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(req: ToolExecutionRequest):
    """Executes sandboxed mathematical and algorithmic computations."""
    try:
        if req.tool_name == "code_interpreter":
            result = code_interpreter.execute(req.code)
            return ToolExecutionResponse(
                success=result["success"],
                output=result["output"],
                error=result.get("error"),
                variables=result.get("variables"),
                elapsed_ms=result["elapsed_ms"],
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {req.tool_name}")
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", response_model=CompareResponse)
async def compare_models(req: CompareRequest):
    """Runs Base and Fine-Tuned models side-by-side on the same question for comparison."""
    try:
        b_sol, b_ans, b_lat, b_tok, b_tps, b_phases = engine.solve(
            subject=req.subject,
            topic=req.topic or "General",
            question_type=req.question_type,
            marks=req.marks,
            question=req.question,
            options=req.options,
            model_type="base",
        )
        base_res = SolveResponse(
            model_name="Qwen2.5-1.5B-Instruct (Base)",
            solution_markdown=b_sol,
            extracted_answer=b_ans,
            structured_phases=StructuredPhases(**b_phases),
            inference_latency_ms=b_lat,
            tokens_generated=b_tok,
            tokens_per_second=b_tps,
            device="CPU (Q4_K_M Quantized)",
        )

        ft_sol, ft_ans, ft_lat, ft_tok, ft_tps, ft_phases = engine.solve(
            subject=req.subject,
            topic=req.topic or "General",
            question_type=req.question_type,
            marks=req.marks,
            question=req.question,
            options=req.options,
            model_type="finetuned",
        )
        ft_res = SolveResponse(
            model_name="GATE-CS-Qwen-1.5B (Fine-Tuned)",
            solution_markdown=ft_sol,
            extracted_answer=ft_ans,
            structured_phases=StructuredPhases(**ft_phases),
            inference_latency_ms=ft_lat,
            tokens_generated=ft_tok,
            tokens_per_second=ft_tps,
            device="CPU (Q4_K_M Quantized)",
        )

        quality_note = (
            "Fine-Tuned model provides 4-phase structured derivation with LaTeX equations, "
            "exhaustive candidate option elimination, and explicit final answer tagging."
        )

        return CompareResponse(
            base_model_result=base_res,
            finetuned_model_result=ft_res,
            quality_delta=quality_note,
        )
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
