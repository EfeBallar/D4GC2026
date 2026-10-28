#!/usr/bin/env python3
"""Start the complete local demo: Ollama, an installed model, and the field app.

    ./start.sh
    ./start.sh --port 8767
    ./start.sh --model mistral:7b-instruct

No model downloads or existing-process termination. Ctrl+C stops only the backend
started by this launcher. Ollama stays available for the next run.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434"


class StartupError(Exception):
    pass


def api(url, body=None, timeout=2):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise StartupError(f"{url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, PermissionError):
            raise StartupError("Local network access is blocked in this session. Run ./start.sh in your terminal.") from exc
        raise
    except (ValueError, UnicodeError) as exc:
        raise StartupError(f"{url} did not return valid JSON. Check which service is using that port.") from exc


def probe(url):
    try:
        return api(url)
    except (urllib.error.URLError, TimeoutError):
        return None


def log_tail(path):
    return "\n".join(path.read_text(errors="replace").splitlines()[-12:]) if path.exists() else ""


def wait_ready(url, process, log, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise StartupError(f"Process exited during startup. {log}\n{log_tail(log)}")
        result = probe(url)
        if result is not None:
            return result
        time.sleep(0.3)
    raise StartupError(f"Timed out waiting for {url}. Check {log}")


def ensure_library():
    index = ROOT / "build/index.db"
    if not index.is_file():
        dataset = ROOT.parent / "D4GC data"
        if not (dataset / "Parsed").is_dir():
            raise StartupError(f"Missing {index} and {dataset / 'Parsed'}. Put the D4GC data folder next to this repo.")
        print("Building the search index...", flush=True)
        subprocess.run([sys.executable, "scripts/build_index.py", "--data", str(dataset),
                        "--csv", "data/documentation.csv"], cwd=ROOT, check=True)
    # The extras are served by the same backend. Only build missing assets.
    for asset, builder in [("galaxy.json", "build_galaxy.py"),
                           ("trail.json", "build_trail.py"),
                           ("radar_sim.json", "make_radar_sim.py")]:
        if not (ROOT / "data" / asset).is_file():
            print(f"Preparing {asset}...", flush=True)
            subprocess.run([sys.executable, f"scripts/{builder}"], cwd=ROOT, check=True)


def ensure_ollama():
    tags = probe(OLLAMA_URL + "/api/tags")
    if tags is not None:
        print("Ollama is already running.", flush=True)
        return tags
    executable = shutil.which("ollama")
    if not executable:
        raise StartupError("Ollama is not installed or is not on PATH. Install/start Ollama, then run ./start.sh again.")
    log = ROOT / "logs/ollama-start.log"
    print("Starting Ollama...", flush=True)
    env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434")
    with log.open("a") as output:
        process = subprocess.Popen([executable, "serve"], cwd=ROOT, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    return wait_ready(OLLAMA_URL + "/api/tags", process, log)


def choose_model(tags, requested=None):
    names = {item.get("name", item.get("model", "")) for item in tags.get("models", [])}
    candidates = [requested] if requested else ["mistral-field", "mistral:7b-instruct"]
    for name in candidates:
        if name in names or name + ":latest" in names:
            return name
    installed = ", ".join(sorted(names)) or "none"
    raise StartupError(f"Requested demo model is not installed. Installed models: {installed}. "
                       "Use ./start.sh --model <installed-model-name>.")


def show_urls(port):
    base = f"http://localhost:{port}"
    print(f"\nSearch and local AI chat: {base}\n"
          f"Document galaxy:         {base}/galaxy\n"
          f"Evidence trail:          {base}/trail\n"
          f"SMS demo:                {base}/sms\n"
          f"Radar simulation:        {base}/radar\n"
          f"Search gaps:             {base}/gaps\n", flush=True)


def check_chat(port):
    """Exercise the same retrieval + streaming model path used by the chat UI."""
    print("Checking a complete RAG request...", flush=True)
    body = json.dumps({"question": "What is early postpartum haemorrhage?", "history": []}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    event, meta, done, answer = "", {}, {}, ""
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event == "meta":
                    meta = data
                elif event == "token":
                    answer += data.get("text", "")
                elif event == "error":
                    raise StartupError(data.get("error", "Chat returned an error"))
                elif event == "done":
                    done = data
                    break
    sources = meta.get("sources", [])
    numbers = {str(s["n"]) for s in sources}
    cited = set(re.findall(r"\[(\d+)\]", answer)) & numbers
    if not done.get("llm_called") or not answer.strip() or not cited:
        raise StartupError("Chat did not produce a cited model answer. Check the chat panel before demonstrating it.")
    if done.get("unsupported_terms"):
        raise StartupError("Chat produced a source-check warning. Review the reply in the chat panel before demonstrating it.")
    print(f"RAG check passed: {done.get('model')}, cited sources "
          f"{', '.join(sorted({s['doc_id'] for s in sources if str(s['n']) in cited}))}.", flush=True)


def optional_chat_check(port, enabled):
    if enabled:
        try:
            check_chat(port)
        except (StartupError, OSError, ValueError) as exc:
            print(f"Chat check needs attention: {exc}\nThe backend stays available.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", default=os.environ.get("CHAT_MODEL"), help="installed Ollama model, default mistral-field")
    parser.add_argument("--check-chat", action="store_true", help="test an end-to-end RAG reply with source citations")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    os.chdir(ROOT)
    (ROOT / "logs").mkdir(exist_ok=True)
    ensure_library()
    tags = ensure_ollama()
    model = choose_model(tags, args.model)
    print(f"Loading local model {model} (first load may take a little while)...", flush=True)
    warm = api(OLLAMA_URL + "/api/generate", {"model": model, "prompt": "Reply OK.", "stream": False,
               "keep_alive": "30m", "options": {"num_predict": 1, "num_ctx": 4096}}, timeout=180)
    if warm.get("error") or not warm.get("done"):
        raise StartupError(f"Model did not finish loading: {warm.get('error', 'incomplete response')}")
    print(f"Local model ready: {model}", flush=True)
    url = f"http://127.0.0.1:{args.port}/api/info"
    existing = probe(url)
    if existing is not None:
        if not isinstance(existing, dict) or "library_version" not in existing:
            raise StartupError(f"Port {args.port} belongs to another service. Use ./start.sh --port 8767.")
        if not existing.get("llm"):
            raise StartupError(f"An app without the AI summary flag is already running on {args.port}. "
                               "Leave it running and use ./start.sh --port 8767 for the full demo.")
        print("The field app is already running with AI enabled. Reusing it without changing its model settings.", flush=True)
        show_urls(args.port)
        optional_chat_check(args.port, args.check_chat)
        return 0
    env = dict(os.environ, CHAT_MODEL=model, CHAT_BASE_URL=OLLAMA_URL + "/v1", CHAT_MODEL_LABEL=model)
    env.pop("CHAT_API_KEY", None)
    command = [sys.executable, "-u", "-m", "msf_search.server", "--host", args.host,
               "--port", str(args.port), "--llm-model", model, "--no-chat-warm"]
    originals = ROOT.parent / "D4GC data/Original"
    if originals.is_dir():
        command.extend(["--originals", str(originals)])
    else:
        print("Original files are unavailable; indexed text and chat remain available.", flush=True)
    log = ROOT / "logs/demo-server.log"
    with log.open("a") as output:
        backend = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        info = wait_ready(url, backend, log)
        print(f"Ready: {info['n_docs']} documents, local RAG model {model}.", flush=True)
        show_urls(args.port)
        optional_chat_check(args.port, args.check_chat)
        print(f"Keep this terminal open. Ctrl+C stops this backend. Ollama stays running.\nBackend log: {log}", flush=True)
        return backend.wait()
    finally:
        if backend.poll() is None:
            backend.terminate()
            try:
                backend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Existing services were left running.")
    except (StartupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
