from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


async def run_paper2poster_generate_wf_api(
    *,
    result_path: Path,
    paper_file: str,
    chat_api_url: str,
    api_key: str,
    model: str,
    vision_model: str,
    poster_width: float,
    poster_height: float,
    logo_path: str = "",
    aff_logo_path: str = "",
    url: str = "",
    email: str = "",
) -> Dict[str, Any]:
    """通过独立子进程执行 paper2poster 工作流，隔离 cwd/env/sys.path 变更。"""
    project_root = get_project_root()
    result_root = Path(result_path).resolve()
    worker_dir = result_root / ".worker" / uuid4().hex
    worker_dir.mkdir(parents=True, exist_ok=True)
    input_json = worker_dir / "input.json"
    output_json = worker_dir / "output.json"
    worker_script = project_root / "script" / "paper2poster_worker.py"

    payload = {
        "result_path": str(result_root),
        "paper_file": paper_file,
        "chat_api_url": chat_api_url,
        "api_key": api_key,
        "model": model,
        "vision_model": vision_model,
        "poster_width": poster_width,
        "poster_height": poster_height,
        "logo_path": logo_path,
        "aff_logo_path": aff_logo_path,
        "url": url,
        "email": email,
    }
    input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        str(worker_script),
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    log.info("[paper2poster-worker] 正在执行...")
    
    async def stream_output(stream, log_func, prefix):
        buffer = ""
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break

            text = chunk.decode("utf-8", errors="ignore").replace("\r", "\n")
            buffer += text

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    log_func("%s %s", prefix, line)

        if buffer.strip():
            log_func("%s %s", prefix, buffer)


    stdout_task = asyncio.create_task(
        stream_output(proc.stdout, log.info, "[worker stdout]")
    )
    stderr_task = asyncio.create_task(
        stream_output(proc.stderr, log.warning, "[worker stderr]")
    )

    await proc.wait()
    log.info("[paper2poster-worker] 执行完成")
    await asyncio.gather(stdout_task, stderr_task)
    
    log.info("[paper2poster-worker] 开始读取输出文件...")
    if not output_json.is_file():
        message = f"paper2poster worker exited with code {proc.returncode} and produced no output json"
        log.error(message)
        return {"success": False, "message": message}

    try:
        result = json.loads(output_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"paper2poster worker returned invalid JSON: {exc}"
        log.error(message)
        return {"success": False, "message": message}

    if proc.returncode != 0 and result.get("success", False):
        result = {
            "success": False,
            "message": f"paper2poster worker exited with code {proc.returncode}",
        }
    return result
