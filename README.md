# Ringmaster

A lightweight web dashboard for controlling and monitoring an MLX ring
cluster running across your Mac minis. Built for a 2-node setup (Mac Mini 1 as
rank 0, Mac Mini 2 as rank 1) but the node list is editable from the UI, so it
grows with your cluster.

Ringmaster doesn't replace your ring-launch or monitor scripts — it wraps
them. Point it at your existing launch command once, and it gives you a
browser UI to start/stop/restart the cluster, switch models, watch logs,
run quick benchmarks, and chat with whatever's currently loaded — instead
of SSH-ing into two machines and running scripts by hand.

## How it's put together

```
   ┌─────────────────────┐         HTTP          ┌─────────────────────┐
   │   dashboard_server   │ ───────────────────▶  │     node_agent       │
   │  (runs on Mac Mini 1) │ ◀───────────────────  │  (runs on Mac Mini 1) │
   │                       │                        │   rank 0              │
   │   serves the web UI   │         HTTP           └─────────────────────┘
   │   at :8080             │ ───────────────────▶  ┌─────────────────────┐
   │                       │ ◀───────────────────  │     node_agent       │
   └─────────────────────┘                        │  (runs on Mac Mini 2) │
                                                     │   rank 1              │
                                                     └─────────────────────┘
```

**`node_agent.py`** runs on every Mac in the cluster (including Mac Mini 1 itself).
It's the only thing that touches that machine directly:
- reports CPU/memory and whatever your monitor script pushes to it
- starts/stops/restarts *your* ring-launch command as a subprocess
- lists model folders on that machine and can switch which one launches next
- tails the launch process's log
- runs a quick one-off generation benchmark

**`dashboard_server.py`** runs once, on Mac Mini 1. It holds the node registry
(which machines exist, their IP/port/rank), polls every agent, and serves
the web UI. It never touches MLX or your models directly — every action
it takes is really just an HTTP call to the right node's agent, in rank
order (0 first, then 1, ...), which is exactly what running your scripts
by hand does today.

**Important:** MLX's ring backend can't hot-join a node mid-run. Adding or
removing a node from the dashboard edits the registry — it takes effect
on the next Restart, not instantly.

## What you need to edit before this works

Both agents read `~/.mlx_cluster/agent_config.json`, created with these
defaults on first run:

```json
{
  "venv": "/Users/genefsbng3/venvs",
  "models_dir": "/Users/genefsbng3/Documents/mlx",
  "current_model": "Llama-3.1-8B-Instruct-4bit",
  "launch_cmd_template": "source {venv}/bin/activate && python /Users/genefsbng3/mlx-dashboard/launch_ring.py --rank {rank} --model {model_path}"
}
```

**`launch_cmd_template` is a placeholder.** Edit it on both Mac Mini 1 and Mac Mini 2 to
match the actual command you currently run by hand to bring up a rank —
same script, same flags, nothing new to learn. Available variables:
`{venv}` `{rank}` `{model_path}` `{models_dir}`.

To add a new model, drop its MLX-format folder into `models_dir` on every
node — it'll show up in the Models tab automatically. Works for Llama,
Qwen, GPT-OSS, or anything else `mlx_lm` supports.

## Setup

On **every** node (Mac Mini 1 and Mac Mini 2), same folder, e.g. `~/mlx-dashboard/`:

```bash
source /Users/genefsbng3/venvs/bin/activate
pip install fastapi uvicorn psutil requests
```

## Starting it

**On each node**, start its agent (pick the right rank per machine):

```bash
python node_agent.py --rank 0 --port 9100      # Mac Mini 1
python node_agent.py --rank 1 --port 9100      # Mac Mini 2
```

**On Mac Mini 1 only**, start the dashboard:

```bash
python dashboard_server.py --port 8080
```

Then open **http://192.168.1.29:8080** (or whatever Mac Mini 1's IP is) in a
browser.

## Wiring up your existing monitor script

Instead of replacing it, have it push numbers to the local agent:

```python
import requests
requests.post("http://localhost:9100/report", json={
    "tokens_per_sec": measured_tps,
    "shard": "layers 0-19",      # optional
    "status": "generating",      # optional
})
```

Do this once per node inside your monitor loop (every second or two, not
per-token) and the dashboard's live numbers start reflecting reality.

## Tabs

| Tab | What it does |
|---|---|
| **Status** | Live cluster stats, per-node memory/CPU/throughput, Start/Restart/Stop |
| **Models** | Lists model folders, switch + auto-restart the ring on a new one |
| **Settings** | Add/remove nodes (with optional SSH bootstrap of a new agent) |
| **Logs** | Tail any node's launch-process output |
| **Bench** | One-click 50-token generation benchmark per node |
| **Chat** | Talk to whatever's currently loaded, via an OpenAI-compatible endpoint you configure |

## Files

- `node_agent.py` — per-node process control, metrics, model discovery
- `dashboard_server.py` — orchestrator + web UI (serves everything at `/`)
- Config/state written at runtime to `~/.mlx_cluster/` on each machine
  (`agent_config.json`, `nodes.json`, `chat_config.json`, `logs/node.log`)
  — nothing here needs to ship with the repo.
