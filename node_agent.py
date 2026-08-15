"""
node_agent.py — runs on EACH Mac mini (MM7 and MM4).

Responsibilities:
  - Report host + inference metrics (/status)
  - Accept live tokens/sec pushed from your existing monitor script (/report)
  - Start / stop / restart YOUR ring-launch script as a subprocess (/control)
  - List models available under your models directory (/models)
  - Change which model this node will launch with next (/config)
  - Tail this node's launch-process log (/logs)
  - Run a quick one-off generation benchmark (/bench)

First-time setup (on MM7 AND MM4):
    cd ~/mlx-dashboard          # put this file + dashboard_server.py here
    source /Users/genefsbng3/venvs/bin/activate
    pip install fastapi uvicorn psutil requests

Run:
    python node_agent.py --rank 0 --port 9100      # on MM7
    python node_agent.py --rank 1 --port 9100      # on MM4

Config lives at ~/.mlx_cluster/agent_config.json and is created with sane
defaults on first run — edit it directly, or edit it from the dashboard's
Settings tab (it proxies to /config on this agent).

IMPORTANT — launch_cmd_template:
This is the one thing you MUST point at your real launch script. It defaults
to a guess. Edit it (via /config or the file) to match how you actually
start a rank on this node, e.g.:

    "source {venv}/bin/activate && python /Users/genefsbng3/mlx-dashboard/launch_ring.py --rank {rank} --model {model_path}"

Available template variables: {venv} {rank} {model_path} {models_dir}
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

CONFIG_DIR = Path.home() / ".mlx_cluster"
CONFIG_PATH = CONFIG_DIR / "agent_config.json"
LOG_PATH = CONFIG_DIR / "logs" / "node.log"

DEFAULT_CONFIG = {
    "venv": "/Users/genefsbng3/venvs",
    "models_dir": "/Users/genefsbng3/Documents/mlx",
    "current_model": "Llama-3.1-8B-Instruct-4bit",
    "launch_cmd_template": (
        "source {venv}/bin/activate && python "
        "/Users/genefsbng3/mlx-dashboard/launch_ring.py "
        "--rank {rank} --model {model_path}"
    ),
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {
    "hostname": socket.gethostname(),
    "rank": None,
    "started_at": time.time(),
    "last_report": {},
    "last_report_at": None,
    "proc": None,          # subprocess.Popen of the running ring launch, if any
    "proc_started_at": None,
}


def load_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    cfg = json.loads(CONFIG_PATH.read_text())
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class Report(BaseModel):
    tokens_per_sec: float | None = None
    shard: str | None = None
    status: str | None = None
    extra: dict | None = None


class ConfigUpdate(BaseModel):
    venv: str | None = None
    models_dir: str | None = None
    current_model: str | None = None
    launch_cmd_template: str | None = None


class ControlAction(BaseModel):
    action: str  # "start" | "stop" | "restart"


@app.get("/status")
def status():
    cfg = load_config()
    vm = psutil.virtual_memory()
    proc = STATE["proc"]
    running = proc is not None and proc.poll() is None
    return {
        "hostname": STATE["hostname"],
        "rank": STATE["rank"],
        "uptime_sec": round(time.time() - STATE["started_at"], 1),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "mem_used_gb": round(vm.used / 1e9, 2),
        "mem_total_gb": round(vm.total / 1e9, 2),
        "mem_percent": vm.percent,
        "current_model": cfg["current_model"],
        "running": running,
        "proc_uptime_sec": (
            round(time.time() - STATE["proc_started_at"], 1)
            if running and STATE["proc_started_at"] else None
        ),
        "inference": STATE["last_report"],
        "last_report_age_sec": (
            round(time.time() - STATE["last_report_at"], 1)
            if STATE["last_report_at"] else None
        ),
    }


@app.post("/report")
def report(r: Report):
    """Point your existing monitor script at POST /report with these fields."""
    STATE["last_report"] = {k: v for k, v in r.model_dump().items() if v is not None}
    STATE["last_report_at"] = time.time()
    return {"ok": True}


@app.get("/config")
def get_config():
    return load_config()


@app.post("/config")
def update_config(c: ConfigUpdate):
    cfg = load_config()
    for k, v in c.model_dump().items():
        if v is not None:
            cfg[k] = v
    save_config(cfg)
    return cfg


@app.get("/models")
def list_models():
    cfg = load_config()
    d = Path(cfg["models_dir"])
    if not d.exists():
        return {"models": [], "error": f"{d} does not exist on this node"}
    models = sorted([p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")])
    return {"models": models, "current": cfg["current_model"]}


def _stop_proc():
    proc = STATE["proc"]
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    STATE["proc"] = None
    STATE["proc_started_at"] = None


def _start_proc():
    cfg = load_config()
    model_path = str(Path(cfg["models_dir"]) / cfg["current_model"])
    cmd = cfg["launch_cmd_template"].format(
        venv=cfg["venv"],
        rank=STATE["rank"],
        model_path=model_path,
        models_dir=cfg["models_dir"],
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_PATH, "a")
    log_f.write(f"\n\n----- launching at {time.ctime()} -----\ncmd: {cmd}\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    STATE["proc"] = proc
    STATE["proc_started_at"] = time.time()
    return cmd


@app.post("/control")
def control(c: ControlAction):
    if c.action == "stop":
        _stop_proc()
        return {"ok": True, "running": False}
    if c.action == "start":
        if STATE["proc"] is not None and STATE["proc"].poll() is None:
            return {"ok": False, "error": "already running"}
        cmd = _start_proc()
        return {"ok": True, "running": True, "cmd": cmd}
    if c.action == "restart":
        _stop_proc()
        time.sleep(1)
        cmd = _start_proc()
        return {"ok": True, "running": True, "cmd": cmd}
    return {"ok": False, "error": f"unknown action {c.action}"}


@app.get("/logs")
def logs(lines: int = 200):
    if not LOG_PATH.exists():
        return {"lines": []}
    with open(LOG_PATH) as f:
        content = f.readlines()
    return {"lines": content[-lines:]}


@app.post("/bench")
def bench(max_tokens: int = 50):
    """Quick one-off generation to sanity-check speed on this node's model."""
    cfg = load_config()
    model_path = str(Path(cfg["models_dir"]) / cfg["current_model"])
    cmd = (
        f"source {cfg['venv']}/bin/activate && "
        f"python -m mlx_lm.generate --model {model_path} "
        f"--prompt 'Hello, how are you?' --max-tokens {max_tokens}"
    )
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=120,
        )
        elapsed = time.time() - t0
        out = result.stdout + result.stderr
        m = re.search(r"([\d.]+)\s*tokens-per-sec", out)
        tps = float(m.group(1)) if m else round(max_tokens / elapsed, 2)
        return {"ok": True, "tokens_per_sec": tps, "elapsed_sec": round(elapsed, 2), "raw": out[-800:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "benchmark timed out after 120s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    STATE["rank"] = args.rank
    load_config()
    uvicorn.run(app, host=args.host, port=args.port)
