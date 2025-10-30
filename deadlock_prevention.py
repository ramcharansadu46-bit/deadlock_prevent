#!/usr/bin/env python3
"""
Deadlock Prevention & Toolkit
Single-file Python app that provides:
 - Core simulator (Processes, Resources, Allocation)
 - Deadlock detection (WFG cycle detection)
 - Banker's algorithm safety check
 - Simple Flask web UI with Cytoscape graph visualization
 - Minimal REST API to interactively add processes/resources, request/release resources,
   detect deadlocks and run Banker's safety check

Usage
-----
1) Install dependencies:
    pip install flask

2) Run server:
    python deadlock_toolkit.py runserver

3) Open http://127.0.0.1:5000 in your browser.

CLI demo:
    python deadlock_toolkit.py demo  # runs a small scenario in console

This file is intentionally self-contained and lightweight for learning and extension.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import argparse
import threading
import json
import time

# Flask imports
from flask import Flask, jsonify, request, render_template_string

# -------------------------
# Core models & simulator
# -------------------------
class Resource:
    def __init__(self, rid: str, total: int = 1):
        self.rid = rid
        self.total = int(total)
        self.available = int(total)

    def to_dict(self):
        return {"rid": self.rid, "total": self.total, "available": self.available}

class Process:
    def __init__(self, pid: str):
        self.pid = pid
        self.allocated = defaultdict(int)   # rid -> count
        self.requesting = defaultdict(int)  # rid -> count (current outstanding request)
        self.max_demand = defaultdict(int)  # rid -> max demand (for Banker)

    def to_dict(self):
        return {
            "pid": self.pid,
            "allocated": dict(self.allocated),
            "requesting": dict(self.requesting),
            "max_demand": dict(self.max_demand),
        }

class Simulator:
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.processes: Dict[str, Process] = {}
        self.lock = threading.Lock()  # for safe concurrent API calls

    # Resource / Process management
    def add_resource(self, rid: str, total: int = 1):
        with self.lock:
            if rid in self.resources:
                raise ValueError(f"Resource {rid} already exists")
            self.resources[rid] = Resource(rid, total)

    def add_process(self, pid: str):
        with self.lock:
            if pid in self.processes:
                raise ValueError(f"Process {pid} already exists")
            self.processes[pid] = Process(pid)

    def set_max_demand(self, pid: str, rid: str, count: int):
        with self.lock:
            p = self.processes[pid]
            p.max_demand[rid] = int(count)

    # Allocation / Request / Release
    def request(self, pid: str, rid: str, count: int = 1) -> bool:
        """Attempt to allocate if available; otherwise record request (blocking state).
        Returns True if allocation succeeded immediately, False if blocked (request recorded).
        """
        with self.lock:
            if pid not in self.processes:
                raise ValueError(f"Unknown process {pid}")
            if rid not in self.resources:
                raise ValueError(f"Unknown resource {rid}")
            p = self.processes[pid]
            r = self.resources[rid]
            count = int(count)
            if r.available >= count:
                r.available -= count
                p.allocated[rid] += count
                # if a previous requesting entry exists, reduce it
                if p.requesting.get(rid):
                    p.requesting[rid] = max(0, p.requesting[rid] - count)
                return True
            else:
                # cannot allocate now -> mark as waiting
                p.requesting[rid] += count
                return False

    def release(self, pid: str, rid: str, count: int = 1) -> int:
        """Release up to 'count' instances of 'rid' held by pid. Returns number actually released."""
        with self.lock:
            if pid not in self.processes:
                raise ValueError(f"Unknown process {pid}")
            if rid not in self.resources:
                raise ValueError(f"Unknown resource {rid}")
            p = self.processes[pid]
            r = self.resources[rid]
            held = p.allocated.get(rid, 0)
            freed = min(held, int(count))
            p.allocated[rid] -= freed
            if p.allocated[rid] == 0 and rid in p.allocated:
                del p.allocated[rid]
            r.available += freed
            return freed

    def state_snapshot(self):
        with self.lock:
            return {
                "resources": {rid: r.to_dict() for rid, r in self.resources.items()},
                "processes": {pid: p.to_dict() for pid, p in self.processes.items()},
            }

    # Graph builders
    def build_wait_for_graph(self) -> Dict[str, Set[str]]:
        """Return WFG as adjacency: P -> set(P) it is waiting on."""
        with self.lock:
            wfg = defaultdict(set)
            for pid, p in self.processes.items():
                for rid, req_count in p.requesting.items():
                    if req_count <= 0:
                        continue
                    # find processes holding this resource
                    for other_pid, other_p in self.processes.items():
                        if other_pid == pid:
                            continue
                        if other_p.allocated.get(rid, 0) > 0:
                            wfg[pid].add(other_pid)
            return dict((k, set(v)) for k, v in wfg.items())

    def detect_deadlock(self) -> Tuple[bool, List[List[str]]]:
        """Detect cycles in WFG. Returns (has_deadlock, list_of_cycles)."""
        graph = self.build_wait_for_graph()
        visited = {}
        stack = []
        cycles: List[List[str]] = []

        def dfs(u):
            visited[u] = 1
            stack.append(u)
            for v in graph.get(u, []):
                if visited.get(v) == 1:
                    idx = stack.index(v)
                    cycles.append(stack[idx:].copy())
                elif visited.get(v) is None:
                    dfs(v)
            stack.pop()
            visited[u] = 2

        for node in list(self.processes.keys()):
            if visited.get(node) is None:
                dfs(node)
        return (len(cycles) > 0, cycles)

    # ------------------------------
    # Banker's algorithm
    # ------------------------------
    def bankers_is_safe_after_grant(self, pid: str, rid: str, count: int) -> Tuple[bool, List[str]]:
        """
        Simulate granting the request (pid requests 'count' of 'rid') and check safety.
        Returns (is_safe, safe_sequence_if_any).
        """
        with self.lock:
            processes = list(self.processes.keys())
            resources = list(self.resources.keys())

            available = {r: self.resources[r].available for r in resources}
            alloc = {p: {r: self.processes[p].allocated.get(r, 0) for r in resources} for p in processes}
            need  = {p: {r: max(0, self.processes[p].max_demand.get(r, 0) - alloc[p][r]) for r in resources} for p in processes}

            if rid not in available:
                return (False, [])
            if available[rid] < count:
                return (False, [])

            # Tentatively grant
            available[rid] -= count
            alloc[pid][rid] += count
            need[pid][rid] = max(0, need[pid][rid] - count)

            finish = {p: False for p in processes}
            safe_sequence: List[str] = []

            while True:
                progressed = False
                for p in processes:
                    if not finish[p]:
                        if all(need[p][r] <= available[r] for r in resources):
                            for r in resources:
                                available[r] += alloc[p][r]
                            finish[p] = True
                            safe_sequence.append(p)
                            progressed = True
                if not progressed:
                    break

            is_safe = all(finish.values())
            return (is_safe, safe_sequence if is_safe else [])

# -------------------------
# Flask web UI + API
# -------------------------
app = Flask(__name__)
sim = Simulator()

# Basic HTML template using Cytoscape.js to display process nodes and resource nodes
HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Deadlock Toolkit</title>
    <script src="https://unpkg.com/cytoscape@3.24.0/dist/cytoscape.min.js"></script>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
      #cy { width: 100%; height: 70vh; display: block; border-bottom: 1px solid #ddd; }
      #controls { padding: 12px; }
      input, button, select { margin: 4px; }
      pre { background: #f7f7f7; padding: 8px; }
    </style>
  </head>
  <body>
    <div id="cy"></div>
    <div id="controls">
      <strong>Quick actions:</strong>
      <input id="pid" placeholder="Process id (e.g. P1)" />
      <input id="rid" placeholder="Resource id (e.g. R1)" />
      <input id="count" placeholder="count" style="width:60px" />
      <button onclick="addProcess()">Add Process</button>
      <button onclick="addResource()">Add Resource</button>
      <button onclick="req()">Request</button>
      <button onclick="rel()">Release</button>
      <button onclick="detect()">Detect Deadlocks</button>
      <button onclick="bankerCheck()">Banker Check (propose)</button>
      <button onclick="demoCycle()">Load Deadlock Demo</button>
      <div id="log"></div>
    </div>

    <script>
      let cy = cytoscape({ container: document.getElementById('cy'), elements: [], style: [
        { selector: 'node[type="process"]', style: { 'shape': 'roundrectangle', 'background-color': '#66ccff', 'label': 'data(label)' } },
        { selector: 'node[type="resource"]', style: { 'shape': 'ellipse', 'background-color': '#ffcc66', 'label': 'data(label)' } },
        { selector: 'edge', style: { 'width': 3, 'line-color': '#ccc', 'target-arrow-color': '#ccc', 'target-arrow-shape': 'triangle' } },
        { selector: '.cycle', style: { 'line-color': '#ff4444', 'target-arrow-color': '#ff4444' } },
        { selector: '.danger', style: { 'background-color': '#ff4444' } }
      ], layout: { name: 'cose' } });

      async function fetchState(){
        const r = await fetch('/api/state');
        return r.json();
      }

      function clearGraph(){ cy.elements().remove(); }

      function addNodesEdgesFromState(state){
        clearGraph();
        const elements = [];
        // add resource nodes
        for(const rid in state.resources){
          const r = state.resources[rid];
          elements.push({ data: { id: 'R:'+rid, label: rid+" (A:"+r.available+"/T:"+r.total+')', type: 'resource' } });
        }
        for(const pid in state.processes){
          const p = state.processes[pid];
          elements.push({ data: { id: 'P:'+pid, label: pid, type: 'process' } });
          // allocation edges P -> R (allocated)
          for(const rid in p.allocated){
            elements.push({ data: { id: 'alloc:'+pid+rid, source: 'P:'+pid, target: 'R:'+rid, label: 'alloc:'+p.allocated[rid] } });
          }
          // request edges P -> R (requesting)
          for(const rid in p.requesting){
            if(p.requesting[rid] > 0){
              elements.push({ data: { id: 'req:'+pid+rid, source: 'P:'+pid, target: 'R:'+rid, label: 'req:'+p.requesting[rid] } });
            }
          }
        }
        cy.add(elements);
        cy.layout({ name: 'cose' }).run();
      }

      async function refresh(){
        const state = await fetchState();
        addNodesEdgesFromState(state);
        document.getElementById('log').innerText = '';
      }

      async function addProcess(){
        const pid = document.getElementById('pid').value || 'P'+Date.now();
        await fetch('/api/add_process', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid}) });
        await refresh();
      }
      async function addResource(){
        const rid = document.getElementById('rid').value || 'R'+Date.now();
        const count = parseInt(document.getElementById('count').value || '1');
        await fetch('/api/add_resource', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rid, total: count}) });
        await refresh();
      }

      async function req(){
        const pid = document.getElementById('pid').value;
        const rid = document.getElementById('rid').value;
        const cnt = parseInt(document.getElementById('count').value || '1');
        const r = await fetch('/api/request', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid, rid, count: cnt}) });
        const js = await r.json();
        document.getElementById('log').innerText = JSON.stringify(js);
        await refresh();
      }

      async function rel(){
        const pid = document.getElementById('pid').value;
        const rid = document.getElementById('rid').value;
        const cnt = parseInt(document.getElementById('count').value || '1');
        const r = await fetch('/api/release', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid, rid, count: cnt}) });
        const js = await r.json();
        document.getElementById('log').innerText = JSON.stringify(js);
        await refresh();
      }

      async function detect(){
        const r = await fetch('/api/detect');
        const js = await r.json();
        document.getElementById('log').innerText = JSON.stringify(js);
        // highlight cycles
        const cycles = js.cycles || [];
        // remove previous cycle classes
        cy.elements().removeClass('cycle');
        // for each cycle (list of processes), color edges between those processes via resource
        cycles.forEach((cycle, idx) => {
          // cycle is list of PIDs like ["P1","P2"]
          for(let i=0;i<cycle.length;i++){
            const a = 'P:'+cycle[i];
            const b = 'P:'+cycle[(i+1)%cycle.length];
            // find an edge representing waiting path P->R and R->P' not directly modeled here; we'll mark nodes
            cy.getElementById(a).addClass('danger');
          }
        });
      }

      async function bankerCheck(){
        const pid = document.getElementById('pid').value;
        const rid = document.getElementById('rid').value;
        const cnt = parseInt(document.getElementById('count').value || '1');
        const r = await fetch('/api/banker_check', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid, rid, count: cnt}) });
        const js = await r.json();
        document.getElementById('log').innerText = JSON.stringify(js);
      }

      async function demoCycle(){
        // loads a classic deadlock: P1 holds R1 wants R2; P2 holds R2 wants R1
        await fetch('/api/reset', { method: 'POST' });
        await fetch('/api/add_resource', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rid:'R1', total:1}) });
        await fetch('/api/add_resource', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rid:'R2', total:1}) });
        await fetch('/api/add_process', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P1'}) });
        await fetch('/api/add_process', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P2'}) });
        await fetch('/api/request', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P1', rid:'R1', count:1}) });
        await fetch('/api/request', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P2', rid:'R2', count:1}) });
        // now P1 requests R2 (blocked), P2 requests R1 (blocked)
        await fetch('/api/request', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P1', rid:'R2', count:1}) });
        await fetch('/api/request', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pid:'P2', rid:'R1', count:1}) });
        await refresh();
      }

      // initial refresh on load
      refresh();
    </script>
  </body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# API endpoints
@app.route('/api/state')
def api_state():
    return jsonify(sim.state_snapshot())

@app.route('/api/add_resource', methods=['POST'])
def api_add_resource():
    data = request.get_json() or {}
    rid = data.get('rid')
    total = int(data.get('total', 1))
    try:
        sim.add_resource(rid, total)
        return jsonify({"ok": True, "rid": rid, "total": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/add_process', methods=['POST'])
def api_add_process():
    data = request.get_json() or {}
    pid = data.get('pid')
    try:
        sim.add_process(pid)
        return jsonify({"ok": True, "pid": pid})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/request', methods=['POST'])
def api_request():
    data = request.get_json() or {}
    pid = data.get('pid')
    rid = data.get('rid')
    count = int(data.get('count', 1))
    try:
        ok = sim.request(pid, rid, count)
        return jsonify({"ok": True, "granted": ok})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/release', methods=['POST'])
def api_release():
    data = request.get_json() or {}
    pid = data.get('pid')
    rid = data.get('rid')
    count = int(data.get('count', 1))
    try:
        freed = sim.release(pid, rid, count)
        return jsonify({"ok": True, "freed": freed})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/detect')
def api_detect():
    try:
        has, cycles = sim.detect_deadlock()
        return jsonify({"ok": True, "has_deadlock": has, "cycles": cycles})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/banker_check', methods=['POST'])
def api_banker_check():
    data = request.get_json() or {}
    pid = data.get('pid')
    rid = data.get('rid')
    count = int(data.get('count', 1))
    try:
        safe, seq = sim.bankers_is_safe_after_grant(pid, rid, count)
        return jsonify({"ok": True, "safe": safe, "safe_sequence": seq})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global sim
    sim = Simulator()
    return jsonify({"ok": True})

# -------------------------
# CLI utilities & demo
# -------------------------

def run_demo_console():
    s = Simulator()
    s.add_resource('R1', 1)
    s.add_resource('R2', 1)
    s.add_process('P1')
    s.add_process('P2')
    # classic deadlock
    s.request('P1', 'R1', 1)
    s.request('P2', 'R2', 1)
    s.request('P1', 'R2', 1)  # blocked
    s.request('P2', 'R1', 1)  # blocked
    print('State snapshot:')
    print(json.dumps(s.state_snapshot(), indent=2))
    has, cycles = s.detect_deadlock()
    print('Deadlock detected:', has)
    print('Cycles:', cycles)
    # Banker check
    s2 = Simulator()
    s2.add_resource('R1', 1); s2.add_resource('R2', 1)
    s2.add_process('P1'); s2.add_process('P2')
    s2.set_max_demand('P1','R1',1); s2.set_max_demand('P1','R2',1)
    s2.set_max_demand('P2','R1',1); s2.set_max_demand('P2','R2',1)
    s2.request('P1','R1',1); s2.request('P2','R2',1)
    ok, seq = s2.bankers_is_safe_after_grant('P2','R1',1)
    print('Banker safe to grant P2->R1?', ok, 'seq=', seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deadlock Toolkit')
    parser.add_argument('cmd', nargs='?', default='runserver', help='runserver | demo')
    args = parser.parse_args()
    if args.cmd == 'demo':
        run_demo_console()
    else:
        print('Starting Flask server at http://127.0.0.1:5000')
        app.run(debug=True)

