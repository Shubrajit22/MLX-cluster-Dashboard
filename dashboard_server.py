"""
dashboard_server.py — runs on MM7 (rank 0).

Talks to node_agent.py running on MM7 and MM4 (and any nodes you add later)
over plain HTTP. Adding/removing a node here does NOT hot-join a live ring —
it edits the node list and, on your next "Restart Cluster", relaunches the
ring across whatever nodes are currently in the list, in rank order. That
mirrors what you do by hand today (rank 0 first, then rank 1, ...).

First-time setup (on MM7 only, alongside node_agent.py):
    source /Users/genefsbng3/venvs/bin/activate
    pip install fastapi uvicorn requests

Run:
    python dashboard_server.py --port 8080

Then open http://10.43.110.29:8080

Node registry persists to ~/.mlx_cluster/nodes.json (created with MM7/MM4
defaults on first run). SSH bootstrap for newly added nodes assumes:
  - passwordless SSH key auth to the new node (same requirement mlx ring
    already has)
  - node_agent.py already copied to the same path on that node
    (~/mlx-dashboard/node_agent.py by default — override in the add-node form)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

CONFIG_DIR = Path.home() / ".mlx_cluster"
NODES_PATH = CONFIG_DIR / "nodes.json"
CHAT_CFG_PATH = CONFIG_DIR / "chat_config.json"

DEFAULT_NODES = {
    "MM7": {"ip": "10.43.110.29", "port": 9100, "rank": 0, "ssh_user": "genefsbng3",
            "agent_path": "~/mlx-dashboard/node_agent.py"},
    "MM4": {"ip": "10.43.110.8", "port": 9100, "rank": 1, "ssh_user": "genefsbng3",
             "agent_path": "~/mlx-dashboard/node_agent.py"},
}
DEFAULT_CHAT_CFG = {"endpoint": "http://10.43.110.29:8000/v1/chat/completions"}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_nodes():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not NODES_PATH.exists():
        NODES_PATH.write_text(json.dumps(DEFAULT_NODES, indent=2))
    return json.loads(NODES_PATH.read_text())


def save_nodes(nodes):
    NODES_PATH.write_text(json.dumps(nodes, indent=2))


def load_chat_cfg():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CHAT_CFG_PATH.exists():
        CHAT_CFG_PATH.write_text(json.dumps(DEFAULT_CHAT_CFG, indent=2))
    return json.loads(CHAT_CFG_PATH.read_text())


def save_chat_cfg(cfg):
    CHAT_CFG_PATH.write_text(json.dumps(cfg, indent=2))


def node_url(n):
    return f"http://{n['ip']}:{n['port']}"


def ranked_nodes(nodes):
    return sorted(nodes.items(), key=lambda kv: kv[1].get("rank", 999))


# ---------- Node registry ----------

class NodeAdd(BaseModel):
    name: str
    ip: str
    port: int = 9100
    rank: int
    ssh_user: str = "genefsbng3"
    agent_path: str = "~/mlx-dashboard/node_agent.py"
    bootstrap: bool = False   # attempt to SSH-start the agent on that node


@app.get("/api/nodes")
def get_nodes():
    return load_nodes()


@app.post("/api/nodes")
def add_node(n: NodeAdd):
    nodes = load_nodes()
    nodes[n.name] = {
        "ip": n.ip, "port": n.port, "rank": n.rank,
        "ssh_user": n.ssh_user, "agent_path": n.agent_path,
    }
    save_nodes(nodes)

    boot_result = None
    if n.bootstrap:
        cmd = (
            f"ssh {n.ssh_user}@{n.ip} "
            f"\"nohup python3 {n.agent_path} --rank {n.rank} --port {n.port} "
            f"> ~/mlx-dashboard/agent.log 2>&1 & disown\""
        )
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
            boot_result = {"cmd": cmd, "returncode": r.returncode, "stderr": r.stderr[-500:]}
        except Exception as e:
            boot_result = {"cmd": cmd, "error": str(e)}

    return {"ok": True, "nodes": nodes, "bootstrap": boot_result}


@app.delete("/api/nodes/{name}")
def remove_node(name: str):
    nodes = load_nodes()
    if name not in nodes:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    n = nodes[name]
    try:
        requests.post(f"{node_url(n)}/control", json={"action": "stop"}, timeout=5)
    except Exception:
        pass
    del nodes[name]
    save_nodes(nodes)
    return {"ok": True, "nodes": nodes, "note": "Restart the cluster to apply the new node set."}


# ---------- Cluster status ----------

@app.get("/api/cluster")
def cluster_status():
    nodes = load_nodes()
    out = []
    for name, n in ranked_nodes(nodes):
        entry = {"name": name, "url": f"{n['ip']}:{n['port']}", "rank": n.get("rank")}
        try:
            t0 = time.time()
            resp = requests.get(f"{node_url(n)}/status", timeout=1.5)
            resp.raise_for_status()
            entry.update(resp.json())
            entry["reachable"] = True
            entry["latency_ms"] = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            entry["reachable"] = False
            entry["error"] = str(e)
        out.append(entry)

    total_tps = sum(n.get("inference", {}).get("tokens_per_sec", 0) or 0 for n in out if n.get("reachable"))
    return {"polled_at": time.time(), "nodes": out, "cluster_tokens_per_sec": round(total_tps, 2)}


class ClusterControl(BaseModel):
    action: str  # start | stop | restart


@app.post("/api/cluster/control")
def cluster_control(c: ClusterControl):
    nodes = load_nodes()
    order = ranked_nodes(nodes) if c.action != "stop" else list(reversed(ranked_nodes(nodes)))
    results = []
    for name, n in order:
        try:
            r = requests.post(f"{node_url(n)}/control", json={"action": c.action}, timeout=10)
            results.append({"node": name, **r.json()})
        except Exception as e:
            results.append({"node": name, "ok": False, "error": str(e)})
        if c.action in ("start", "restart"):
            time.sleep(2)   # rank 0 connects first, then rank 1, ... matches manual launch order
    return {"results": results}


# ---------- Models ----------

@app.get("/api/models")
def get_models():
    nodes = load_nodes()
    ranked = ranked_nodes(nodes)
    if not ranked:
        return {"models": [], "current": None}
    name, n = ranked[0]   # model list comes from rank 0; assumes mirrored dirs across nodes
    try:
        r = requests.get(f"{node_url(n)}/models", timeout=5)
        return r.json()
    except Exception as e:
        return {"models": [], "error": str(e)}


class ModelSelect(BaseModel):
    model: str
    restart: bool = True


@app.post("/api/models/select")
def select_model(m: ModelSelect):
    nodes = load_nodes()
    updates = []
    for name, n in ranked_nodes(nodes):
        try:
            r = requests.post(f"{node_url(n)}/config", json={"current_model": m.model}, timeout=5)
            updates.append({"node": name, "ok": True, "config": r.json()})
        except Exception as e:
            updates.append({"node": name, "ok": False, "error": str(e)})
    restart_results = None
    if m.restart:
        restart_results = cluster_control(ClusterControl(action="restart"))
    return {"updates": updates, "restart": restart_results}


# ---------- Logs ----------

@app.get("/api/logs/{name}")
def get_logs(name: str, lines: int = 200):
    nodes = load_nodes()
    if name not in nodes:
        return JSONResponse({"error": "not found"}, status_code=404)
    n = nodes[name]
    try:
        r = requests.get(f"{node_url(n)}/logs", params={"lines": lines}, timeout=5)
        return r.json()
    except Exception as e:
        return {"lines": [], "error": str(e)}


# ---------- Bench ----------

@app.post("/api/bench")
def run_bench(max_tokens: int = 50):
    nodes = load_nodes()
    results = []
    for name, n in ranked_nodes(nodes):
        try:
            r = requests.post(f"{node_url(n)}/bench", params={"max_tokens": max_tokens}, timeout=130)
            results.append({"node": name, **r.json()})
        except Exception as e:
            results.append({"node": name, "ok": False, "error": str(e)})
    return {"results": results}


# ---------- Chat ----------

@app.get("/api/chat/config")
def get_chat_config():
    return load_chat_cfg()


class ChatConfig(BaseModel):
    endpoint: str


@app.post("/api/chat/config")
def set_chat_config(c: ChatConfig):
    save_chat_cfg({"endpoint": c.endpoint})
    return {"ok": True}


class ChatMessage(BaseModel):
    message: str


@app.post("/api/chat/send")
def chat_send(m: ChatMessage):
    cfg = load_chat_cfg()
    try:
        r = requests.post(
            cfg["endpoint"], timeout=60,
            json={"model": "local", "messages": [{"role": "user", "content": m.message}]},
        )
        r.raise_for_status()
        data = r.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", json.dumps(data)[:500])
        return {"ok": True, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------- UI ----------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML_PAGE


HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MLX Cluster</title>
<style>
  :root {
    --bg: #ffffff; --panel: #ffffff; --border: #e7e7ea; --border-soft: #f0f0f2;
    --ink: #111114; --muted: #9a9aa3; --label: #b3b3bb; --accent: #111114;
    --good: #1f7a45; --good-bg: #e7f6ee; --bad: #b3261e; --bad-bg: #fbe9e8;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Inter","Helvetica Neue",Arial,sans-serif;
    -webkit-font-smoothing:antialiased; }
  .nav { display:flex; align-items:center; justify-content:space-between; padding:14px 28px; border-bottom:1px solid var(--border-soft); }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand .mark { width:26px; height:26px; border-radius:7px; background:var(--ink); display:flex; align-items:center; justify-content:center; color:#fff; font-size:13px; font-weight:700; }
  .brand .name { font-weight:600; font-size:14px; letter-spacing:-0.01em; }
  .brand .ver { color:var(--muted); font-size:11px; margin-left:2px; }
  .tabs { display:flex; gap:4px; }
  .tab { font-size:13px; color:var(--muted); padding:7px 14px; border-radius:8px; cursor:pointer; user-select:none; }
  .tab.active { background:#f3f3f5; color:var(--ink); font-weight:500; }
  .nav-right { display:flex; align-items:center; gap:14px; color:var(--muted); font-size:13px; }

  main { max-width:1040px; margin:0 auto; padding:32px 28px 80px; }
  .page { display:none; }
  .page.active { display:block; }
  .eyebrow { font-size:11px; letter-spacing:0.08em; color:var(--label); font-weight:600; margin-bottom:6px; }
  .headrow { display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }
  h1 { font-size:24px; margin:0; font-weight:650; letter-spacing:-0.015em; }
  .pill-toggle { display:flex; align-items:center; gap:8px; }
  .pill { font-size:12px; padding:6px 12px; border-radius:999px; border:1px solid var(--border); color:var(--muted); background:#fff; cursor:pointer; }
  .pill.on { background:var(--ink); color:#fff; border-color:var(--ink); }

  .stats-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:16px; }
  .stat { border:1px solid var(--border); border-radius:var(--radius); padding:22px 20px; text-align:center; background:var(--panel); }
  .stat .label { font-size:10.5px; letter-spacing:0.08em; color:var(--label); font-weight:600; margin-bottom:10px; }
  .stat .value { font-size:30px; font-weight:700; letter-spacing:-0.02em; }

  .panel { border:1px solid var(--border); border-radius:var(--radius); margin-bottom:16px; overflow:hidden; }
  .panel-head { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:12px 20px; background:#fafafb; border-bottom:1px solid var(--border-soft); font-size:11.5px; letter-spacing:0.06em; color:#6b6b74; font-weight:650; }
  .panel-body { padding:20px; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; }
  .two-col > div:first-child { border-right:1px solid var(--border-soft); padding-right:28px; }
  .two-col > div:last-child { padding-left:28px; }
  .metric-label { font-size:12.5px; color:var(--muted); margin-bottom:8px; }
  .metric-value { font-size:22px; font-weight:700; }

  .endpoint-row { display:flex; align-items:center; justify-content:space-between; padding:14px 0; border-bottom:1px solid var(--border-soft); }
  .endpoint-row:last-child { border-bottom:none; }
  .endpoint-name { font-size:13px; color:#45454d; }
  .endpoint-url { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; background:#f6f6f8; border:1px solid var(--border-soft); border-radius:7px; padding:6px 10px; color:#33333a; }
  .badge { font-size:11px; padding:3px 9px; border-radius:999px; font-weight:600; }
  .badge.good { background:var(--good-bg); color:var(--good); }
  .badge.bad { background:var(--bad-bg); color:var(--bad); }

  .node-card { border:1px solid var(--border-soft); border-radius:10px; padding:14px 16px; margin-bottom:10px; }
  .node-card:last-child { margin-bottom:0; }
  .node-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
  .node-name { font-size:13.5px; font-weight:600; }
  .node-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
  .node-grid .k { font-size:10.5px; color:var(--label); letter-spacing:0.05em; margin-bottom:4px; }
  .node-grid .v { font-size:13px; font-weight:600; }
  .bar { height:5px; border-radius:3px; background:#eee; margin-top:6px; overflow:hidden; }
  .bar > div { height:100%; background:var(--ink); }
  .updated { color:var(--muted); font-size:11.5px; margin-top:6px; }

  button.btn { font-size:12.5px; padding:8px 14px; border-radius:8px; border:1px solid var(--border); background:#fff; color:var(--ink); cursor:pointer; font-weight:500; }
  button.btn.dark { background:var(--ink); color:#fff; border-color:var(--ink); }
  button.btn.danger { color:var(--bad); border-color:#f0d3d1; }
  button.btn:hover { background:#f6f6f8; }
  button.btn.dark:hover { background:#2a2a2e; }
  input, select, textarea { font-size:13px; padding:8px 10px; border-radius:8px; border:1px solid var(--border); font-family:inherit; width:100%; }
  label.field { display:block; font-size:11px; color:var(--muted); margin-bottom:4px; margin-top:10px; }
  .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:0 16px; }
  .model-row { display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--border-soft); }
  .model-row:last-child { border-bottom:none; }
  select.node-select { max-width:220px; }
  pre.logbox { background:#0e0e10; color:#d8d8dc; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; padding:16px; border-radius:10px; height:420px; overflow:auto; white-space:pre-wrap; }
  .chat-log { border:1px solid var(--border-soft); border-radius:10px; height:360px; overflow:auto; padding:14px; margin-bottom:12px; }
  .chat-msg { margin-bottom:10px; font-size:13px; }
  .chat-msg .who { font-size:10.5px; color:var(--label); letter-spacing:0.05em; margin-bottom:2px; }
  .chat-input-row { display:flex; gap:8px; }
</style>
</head>
<body>
  <div class="nav">
    <div class="brand">
      <div class="mark">M</div>
      <div class="name">MLX Cluster</div>
      <div class="ver">MM7 · MM4</div>
    </div>
    <div class="tabs" id="tabs">
      <div class="tab active" data-page="status">Status</div>
      <div class="tab" data-page="models">Models</div>
      <div class="tab" data-page="settings">Settings</div>
      <div class="tab" data-page="logs">Logs</div>
      <div class="tab" data-page="bench">Bench</div>
      <div class="tab" data-page="chat">Chat</div>
    </div>
    <div class="nav-right"><span>●</span></div>
  </div>

  <main>

  <!-- STATUS -->
  <div class="page active" id="page-status">
    <div class="eyebrow">STATUS</div>
    <div class="headrow">
      <h1>Cluster Stats</h1>
      <div class="pill-toggle">
        <div class="pill dark" onclick="clusterControl('start')">Start</div>
        <div class="pill" onclick="clusterControl('restart')">Restart</div>
        <div class="pill" onclick="clusterControl('stop')">Stop</div>
        <div class="pill" id="polled">connecting…</div>
      </div>
    </div>
    <div class="stats-grid">
      <div class="stat"><div class="label">CLUSTER TOKENS/SEC</div><div class="value" id="tps">—</div></div>
      <div class="stat"><div class="label">NODES ONLINE</div><div class="value" id="online">—</div></div>
      <div class="stat"><div class="label">AVG MEMORY USED</div><div class="value" id="avgmem">—</div></div>
    </div>
    <div class="panel">
      <div class="panel-head">⟳ CLUSTER OVERVIEW</div>
      <div class="panel-body two-col">
        <div><div class="metric-label">Fastest Node</div><div class="metric-value" id="fastest">—</div></div>
        <div><div class="metric-label">Total Memory Headroom</div><div class="metric-value" id="headroom">—</div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">🔗 NODE ENDPOINTS</div>
      <div class="panel-body" id="endpoints"></div>
    </div>
    <div class="panel">
      <div class="panel-head">▸ NODE DETAILS</div>
      <div class="panel-body" id="nodedetails"></div>
    </div>
  </div>

  <!-- MODELS -->
  <div class="page" id="page-models">
    <div class="eyebrow">MODELS</div>
    <h1 style="margin-bottom:24px;">Model Selection</h1>
    <div class="panel">
      <div class="panel-head">AVAILABLE MODELS (from rank 0's models directory)</div>
      <div class="panel-body" id="modelsList">loading…</div>
    </div>
    <div class="panel">
      <div class="panel-head">NOTE</div>
      <div class="panel-body" style="font-size:12.5px;color:var(--muted);">
        Selecting a model updates <code>current_model</code> on every node, then restarts the ring
        (rank 0 first, then rank 1, ...) so it launches with the new model. Model directory names
        must exist under <code>/Users/genefsbng3/Documents/mlx</code> on every node — e.g. drop a
        Qwen or GPT-OSS MLX-format folder there and it'll show up here automatically.
      </div>
    </div>
  </div>

  <!-- SETTINGS -->
  <div class="page" id="page-settings">
    <div class="eyebrow">SETTINGS</div>
    <h1 style="margin-bottom:24px;">Nodes</h1>
    <div class="panel">
      <div class="panel-head">CURRENT NODES</div>
      <div class="panel-body" id="nodesTable"></div>
    </div>
    <div class="panel">
      <div class="panel-head">ADD NODE</div>
      <div class="panel-body">
        <div class="form-grid">
          <div><label class="field">Name</label><input id="add-name" placeholder="MM10"></div>
          <div><label class="field">IP</label><input id="add-ip" placeholder="10.43.110.50"></div>
          <div><label class="field">Rank</label><input id="add-rank" type="number" placeholder="2"></div>
          <div><label class="field">Agent Port</label><input id="add-port" type="number" value="9100"></div>
          <div><label class="field">SSH User</label><input id="add-user" value="genefsbng3"></div>
          <div><label class="field">Agent Path (on that node)</label><input id="add-path" value="~/mlx-dashboard/node_agent.py"></div>
        </div>
        <label class="field"><input type="checkbox" id="add-bootstrap" style="width:auto;display:inline;margin-right:6px;">
          Try to SSH-start node_agent.py on that node now (requires passwordless SSH + file already there)</label>
        <div style="margin-top:14px;"><button class="btn dark" onclick="addNode()">Add Node</button></div>
        <div id="addResult" style="font-size:12px;color:var(--muted);margin-top:8px;"></div>
      </div>
    </div>
  </div>

  <!-- LOGS -->
  <div class="page" id="page-logs">
    <div class="eyebrow">LOGS</div>
    <div class="headrow">
      <h1>Node Logs</h1>
      <select class="node-select" id="logNodeSelect" onchange="loadLogs()"></select>
    </div>
    <pre class="logbox" id="logbox">select a node…</pre>
  </div>

  <!-- BENCH -->
  <div class="page" id="page-bench">
    <div class="eyebrow">BENCH</div>
    <div class="headrow"><h1>Performance Benchmark</h1>
      <button class="btn dark" onclick="runBench()">Run Benchmark</button>
    </div>
    <div class="panel">
      <div class="panel-head">RESULTS (50-token generation per node)</div>
      <div class="panel-body" id="benchResults">Run a benchmark to see results.</div>
    </div>
  </div>

  <!-- CHAT -->
  <div class="page" id="page-chat">
    <div class="eyebrow">CHAT</div>
    <h1 style="margin-bottom:24px;">Chat</h1>
    <div class="panel">
      <div class="panel-head">ENDPOINT</div>
      <div class="panel-body">
        <div style="display:flex;gap:8px;">
          <input id="chatEndpoint" placeholder="http://10.43.110.29:8000/v1/chat/completions">
          <button class="btn" onclick="saveChatEndpoint()">Save</button>
        </div>
        <div style="font-size:11.5px;color:var(--muted);margin-top:6px;">
          Point this at whatever OpenAI-compatible server your ring/launch script exposes.
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">CONVERSATION</div>
      <div class="panel-body">
        <div class="chat-log" id="chatLog"></div>
        <div class="chat-input-row">
          <input id="chatInput" placeholder="Ask something…" onkeydown="if(event.key==='Enter') sendChat()">
          <button class="btn dark" onclick="sendChat()">Send</button>
        </div>
      </div>
    </div>
  </div>

  </main>

<script>
// ---- tab switching ----
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.page').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('page-' + t.dataset.page).classList.add('active');
  if (t.dataset.page === 'models') loadModels();
  if (t.dataset.page === 'settings') loadNodesTable();
  if (t.dataset.page === 'logs') loadNodeSelect();
  if (t.dataset.page === 'chat') loadChatConfig();
}));

function fmtGB(n) { return (n ?? 0).toFixed(1) + ' GB'; }

// ---- status ----
async function tickStatus() {
  try {
    const r = await fetch('/api/cluster');
    const data = await r.json();
    document.getElementById('polled').textContent = 'updated ' + new Date(data.polled_at * 1000).toLocaleTimeString();
    document.getElementById('tps').textContent = data.cluster_tokens_per_sec + ' tok/s';
    const reachable = data.nodes.filter(n => n.reachable);
    document.getElementById('online').textContent = reachable.length + ' / ' + data.nodes.length;
    const avgMemPct = reachable.length ? Math.round(reachable.reduce((s,n)=>s+(n.mem_percent||0),0)/reachable.length) : 0;
    document.getElementById('avgmem').textContent = avgMemPct + '%';

    let fastest = null, headroomGb = 0;
    for (const n of reachable) {
      const tps = n.inference?.tokens_per_sec || 0;
      if (!fastest || tps > fastest.tps) fastest = { host: n.hostname, tps };
      headroomGb += Math.max((n.mem_total_gb||0) - (n.mem_used_gb||0), 0);
    }
    document.getElementById('fastest').textContent = fastest && fastest.tps ? `${fastest.host} (${fastest.tps} tok/s)` : '—';
    document.getElementById('headroom').textContent = fmtGB(headroomGb);

    const ep = document.getElementById('endpoints'); ep.innerHTML = '';
    for (const n of data.nodes) {
      const row = document.createElement('div'); row.className = 'endpoint-row';
      row.innerHTML = `<div class="endpoint-name">${n.name} — ${n.hostname||n.url} (rank ${n.rank})</div>
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="badge ${n.reachable?'good':'bad'}">${n.reachable?'online':'unreachable'}</span>
          <span class="endpoint-url">http://${n.url}</span></div>`;
      ep.appendChild(row);
    }

    const nd = document.getElementById('nodedetails'); nd.innerHTML = '';
    for (const n of data.nodes) {
      const card = document.createElement('div'); card.className = 'node-card';
      if (!n.reachable) {
        card.innerHTML = `<div class="node-top"><div class="node-name">${n.name}</div><span class="badge bad">unreachable</span></div>
          <div style="font-size:12px;color:var(--muted);">${n.error||''}</div>`;
        nd.appendChild(card); continue;
      }
      const inf = n.inference || {};
      card.innerHTML = `<div class="node-top"><div class="node-name">${n.hostname}</div>
          <span class="badge ${n.running?'good':'bad'}">${n.running ? 'running · ' : 'stopped · '}${n.current_model||'no model'}</span></div>
        <div class="node-grid">
          <div><div class="k">MEMORY</div><div class="v">${fmtGB(n.mem_used_gb)} / ${fmtGB(n.mem_total_gb)}</div>
            <div class="bar"><div style="width:${n.mem_percent}%"></div></div></div>
          <div><div class="k">CPU</div><div class="v">${n.cpu_percent}%</div></div>
          <div><div class="k">SHARD</div><div class="v">${inf.shard||'—'}</div></div>
          <div><div class="k">TOK/S</div><div class="v">${inf.tokens_per_sec ?? '—'}</div></div>
        </div>`;
      nd.appendChild(card);
    }
  } catch (e) { document.getElementById('polled').textContent = 'dashboard unreachable'; }
}
tickStatus(); setInterval(tickStatus, 2000);

async function clusterControl(action) {
  document.getElementById('polled').textContent = action + '…';
  await fetch('/api/cluster/control', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action}) });
  tickStatus();
}

// ---- models ----
async function loadModels() {
  const el = document.getElementById('modelsList'); el.textContent = 'loading…';
  const r = await fetch('/api/models'); const data = await r.json();
  if (data.error) { el.textContent = 'Error: ' + data.error; return; }
  el.innerHTML = '';
  for (const m of data.models) {
    const row = document.createElement('div'); row.className = 'model-row';
    const isCurrent = m === data.current;
    row.innerHTML = `<div>${m} ${isCurrent ? '<span class="badge good">active</span>' : ''}</div>
      <button class="btn ${isCurrent?'':'dark'}" ${isCurrent?'disabled':''}>${isCurrent?'Active':'Use this model'}</button>`;
    row.querySelector('button').onclick = () => selectModel(m);
    el.appendChild(row);
  }
  if (!data.models.length) el.innerHTML = '<div style="color:var(--muted);font-size:13px;">No model folders found.</div>';
}
async function selectModel(model) {
  if (!confirm(`Switch to "${model}" and restart the cluster with it?`)) return;
  document.getElementById('modelsList').textContent = 'switching + restarting cluster…';
  await fetch('/api/models/select', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model, restart:true}) });
  loadModels();
}

// ---- settings / nodes ----
async function loadNodesTable() {
  const el = document.getElementById('nodesTable'); el.textContent = 'loading…';
  const r = await fetch('/api/nodes'); const nodes = await r.json();
  el.innerHTML = '';
  for (const [name, n] of Object.entries(nodes)) {
    const row = document.createElement('div'); row.className = 'model-row';
    row.innerHTML = `<div>${name} — rank ${n.rank} — ${n.ip}:${n.port}</div>
      <button class="btn danger">Remove</button>`;
    row.querySelector('button').onclick = () => removeNode(name);
    el.appendChild(row);
  }
}
async function removeNode(name) {
  if (!confirm(`Remove ${name} from the cluster? You'll need to Restart Cluster on the Status tab to apply it.`)) return;
  await fetch('/api/nodes/' + name, { method:'DELETE' });
  loadNodesTable();
}
async function addNode() {
  const body = {
    name: document.getElementById('add-name').value,
    ip: document.getElementById('add-ip').value,
    rank: parseInt(document.getElementById('add-rank').value || '0'),
    port: parseInt(document.getElementById('add-port').value || '9100'),
    ssh_user: document.getElementById('add-user').value,
    agent_path: document.getElementById('add-path').value,
    bootstrap: document.getElementById('add-bootstrap').checked,
  };
  const r = await fetch('/api/nodes', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await r.json();
  document.getElementById('addResult').textContent = data.bootstrap ? JSON.stringify(data.bootstrap) : 'Added. Restart cluster from Status tab to include it.';
  loadNodesTable();
}

// ---- logs ----
async function loadNodeSelect() {
  const r = await fetch('/api/nodes'); const nodes = await r.json();
  const sel = document.getElementById('logNodeSelect'); sel.innerHTML = '';
  for (const name of Object.keys(nodes)) {
    const opt = document.createElement('option'); opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  }
  loadLogs();
}
async function loadLogs() {
  const name = document.getElementById('logNodeSelect').value;
  if (!name) return;
  const r = await fetch(`/api/logs/${name}?lines=300`);
  const data = await r.json();
  document.getElementById('logbox').textContent = (data.lines || []).join('') || (data.error || 'no logs yet');
}

// ---- bench ----
async function runBench() {
  const el = document.getElementById('benchResults'); el.textContent = 'running benchmark on each node (this can take a minute)…';
  const r = await fetch('/api/bench', { method:'POST' });
  const data = await r.json();
  el.innerHTML = '';
  for (const res of data.results) {
    const row = document.createElement('div'); row.className = 'model-row';
    row.innerHTML = res.ok
      ? `<div>${res.node}</div><div><b>${res.tokens_per_sec} tok/s</b> · ${res.elapsed_sec}s</div>`
      : `<div>${res.node}</div><div style="color:var(--bad);">${res.error}</div>`;
    el.appendChild(row);
  }
}

// ---- chat ----
async function loadChatConfig() {
  const r = await fetch('/api/chat/config'); const cfg = await r.json();
  document.getElementById('chatEndpoint').value = cfg.endpoint;
}
async function saveChatEndpoint() {
  const endpoint = document.getElementById('chatEndpoint').value;
  await fetch('/api/chat/config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({endpoint}) });
}
function addChatMsg(who, text) {
  const log = document.getElementById('chatLog');
  const div = document.createElement('div'); div.className = 'chat-msg';
  div.innerHTML = `<div class="who">${who}</div><div>${text}</div>`;
  log.appendChild(div); log.scrollTop = log.scrollHeight;
}
async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim(); if (!msg) return;
  addChatMsg('YOU', msg); input.value = '';
  addChatMsg('CLUSTER', '…thinking…');
  const r = await fetch('/api/chat/send', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg}) });
  const data = await r.json();
  const log = document.getElementById('chatLog'); log.lastChild.remove();
  addChatMsg('CLUSTER', data.ok ? data.reply : ('Error: ' + data.error));
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    load_nodes()
    load_chat_cfg()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
