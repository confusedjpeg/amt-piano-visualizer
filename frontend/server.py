"""
Pianoforge — FastAPI server for the AI Piano Arranger frontend.

Serves the frontend and exposes an API to run the pipeline asynchronously.
"""

from __future__ import annotations

import sys
import time
import asyncio
import uuid
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator

app = FastAPI(title="Pianoforge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent
PUBLIC_DIR = STATIC_DIR / "public"
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"

for d in [INPUT_DIR, OUTPUT_DIR, INTERMEDIATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=3)
_active_runs: dict[str, dict] = {}
_run_futures: dict[str, Future] = {}
_runs_lock = threading.Lock()


@app.on_event("shutdown")
def shutdown():
    executor.shutdown(wait=False)


# ── Serve frontend ──

@app.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")


app.mount("/css", StaticFiles(directory=str(PUBLIC_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(PUBLIC_DIR / "js")), name="js")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── API: Upload + Run ──

@app.post("/api/run")
async def run_pipeline(
    file: UploadFile = File(...),
    include_vocals: str = Form("true"),
    has_piano: str = Form("true"),
    pattern: str = Form("pop_ballad"),
):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename).suffix if file.filename else ".mp3"
    input_path = INPUT_DIR / f"{run_id}{ext}"

    content = await file.read()
    input_path.write_bytes(content)

    config = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    config.include_vocals = include_vocals.lower() == "true"
    config.has_piano = has_piano.lower() == "true"
    if pattern:
        config.arranger.default_pattern = pattern

    _active_runs[run_id] = {
        "run_id": run_id,
        "status": "pending",
        "progress": 0,
        "step": "",
        "steps_completed": [],
        "midi_path": None,
        "video_path": None,
        "duration": None,
        "warnings": [],
        "error": None,
        "filename": file.filename or "unknown",
        "created_at": datetime.now().isoformat(),
        "pattern": pattern,
        "include_vocals": include_vocals.lower() == "true",
        "has_piano": has_piano.lower() == "true",
    }

    asyncio.create_task(_execute_pipeline(run_id, input_path, config))

    print(f"  New run created: {run_id} (file={file.filename}, pattern={pattern})")
    return {"run_id": run_id, "status": "pending"}


_TOTAL_EXPECTED_STEPS = 5


async def _execute_pipeline(run_id: str, input_path: Path, config: PipelineConfig):
    print(f"  [{run_id}] Pipeline starting...")
    print(f"  [{run_id}] Config: include_vocals={config.include_vocals}, has_piano={config.has_piano}, pattern={config.arranger.default_pattern}")
    _active_runs[run_id]["status"] = "running"

    def progress_callback(info):
        step = info.get("step", "")
        completed = info.get("steps_completed", [])
        total = info.get("total_expected", _TOTAL_EXPECTED_STEPS)
        print(f"  [{run_id}] Step: {step} | Progress: {len(completed)}/{total} completed")
        with _runs_lock:
            _active_runs[run_id].update({
                "step": step,
                "steps_completed": completed,
                "progress": min(int(len(completed) / total * 100), 99),
            })

    def _run():
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run(
            audio_path=input_path,
            include_vocals=config.include_vocals,
            has_piano=config.has_piano,
            progress_callback=progress_callback,
        )
        return result

    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(executor, _run)
        with _runs_lock:
            _run_futures[run_id] = future
        result = await future

        print(f"  [{run_id}] Pipeline completed successfully in {result.duration_seconds:.1f}s")
        with _runs_lock:
            _active_runs[run_id].update({
                "status": "completed",
                "progress": 100,
                "midi_path": str(result.midi_path) if result.midi_path else None,
                "video_path": str(result.video_path) if result.video_path and str(result.video_path) else None,
                "duration": result.duration_seconds,
                "steps_completed": result.steps_completed,
                "warnings": result.warnings,
            })
    except Exception as exc:
        import traceback
        print(f"  [{run_id}] ERROR: {exc}")
        traceback.print_exc()
        with _runs_lock:
            _active_runs[run_id].update({
                "status": "failed",
                "error": str(exc),
            })
    finally:
        with _runs_lock:
            _run_futures.pop(run_id, None)


# ── API: Poll status ──

@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    run = _active_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# ── API: Download results ──

@app.get("/api/download/{run_id}/{file_type}")
async def download_result(run_id: str, file_type: str):
    run = _active_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    key = "midi_path" if file_type == "midi" else "video_path"
    path_str = run.get(key)
    if not path_str:
        raise HTTPException(status_code=404, detail=f"No {file_type} file available")

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    media_type = "audio/midi" if file_type == "midi" else "video/mp4"
    return FileResponse(path, media_type=media_type, filename=path.name)


# ── API: List runs ──

@app.get("/api/runs")
async def list_runs():
    with _runs_lock:
        runs = list(_active_runs.values())
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    for i, run in enumerate(runs):
        if run["status"] == "pending":
            run["queue_position"] = i + 1
    return runs


# ── API: Cancel run ──

@app.post("/api/cancel/{run_id}")
async def cancel_run(run_id: str):
    with _runs_lock:
        run = _active_runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run["status"] not in ("pending", "running"):
            raise HTTPException(status_code=400, detail=f"Cannot cancel run in '{run['status']}' state")

        future = _run_futures.get(run_id)
        if future and not future.done():
            future.cancel()

        run["status"] = "cancelled"
        run["error"] = "Cancelled by user"

    print(f"  [{run_id}] Run cancelled by user")
    return {"run_id": run_id, "status": "cancelled"}


@app.head("/api/runs")
async def head_runs():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Pianoforge FastAPI Server starting...")
    print(f"  Frontend + API: http://localhost:8765")
    print("  Pipeline logs appear below as they run.")
    print("=" * 60)
    uvicorn.run("server:app", host="127.0.0.1", port=8765, reload=True, log_level="info")
