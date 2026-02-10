from pathlib import Path
import subprocess
from threading import Lock

from flask import Flask, Response, jsonify, request, send_from_directory
from ollama import ChatResponse, chat

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "HTML Front-End"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
current_model = ""
status_lock = Lock()
model_status = {
  "state": "idle",
  "message": "",
  "model": "",
}


def update_status(state: str, message: str, model: str = "") -> None:
  with status_lock:
    model_status["state"] = state
    model_status["message"] = message
    model_status["model"] = model


def run_ollama_command(args: list[str]) -> subprocess.CompletedProcess:
  return subprocess.run(args, capture_output=True, text=True)


def list_models() -> list[str]:
  result = run_ollama_command(["ollama", "list"])
  if result.returncode != 0:
    raise RuntimeError(result.stderr.strip() or "Failed to list models.")

  lines = result.stdout.strip().splitlines()
  if len(lines) <= 1:
    return []

  models: list[str] = []
  for line in lines[1:]:
    if not line.strip():
      continue
    models.append(line.split()[0])
  return models


def ensure_model_pulled(model: str) -> None:
  update_status("checking", "Checking model...", model)
  models = list_models()
  if model in models:
    update_status("loading", "Loading model...", model)
    return

  update_status("pulling", "Pulling model...", model)
  result = run_ollama_command(["ollama", "pull", model])
  if result.returncode != 0:
    raise RuntimeError(result.stderr.strip() or "Failed to pull model.")

  update_status("loading", "Loading model...", model)


@app.get("/")
def serve_index() -> Response:
  return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/set-model")
def set_model():
  data = request.get_json(silent=True) or {}
  model = str(data.get("model", "")).strip()
  if not model:
    return jsonify({"ok": False, "error": "Model is required."}), 400

  global current_model
  current_model = model
  update_status("selected", "Model selected.", model)
  return jsonify({"ok": True, "model": model})


@app.get("/models")
def get_models():
  try:
    models = list_models()
    return jsonify({"ok": True, "models": models})
  except Exception as exc:
    return jsonify({"ok": False, "error": str(exc) or "Failed to load models."}), 500


@app.get("/model-status")
def get_model_status():
  with status_lock:
    return jsonify({"ok": True, **model_status})


@app.post("/chat")
def chat_prompt():
  global current_model
  data = request.get_json(silent=True) or {}
  prompt = str(data.get("prompt", "")).strip()
  model = str(data.get("model", "")).strip() or current_model

  if not prompt:
    return jsonify({"ok": False, "error": "Prompt is required."}), 400
  if not model:
    return jsonify({"ok": False, "error": "Model is required."}), 400

  try:
    ensure_model_pulled(model)
    current_model = model
    update_status("ready", "Model ready.", model)
  except Exception as exc:
    update_status("error", str(exc) or "Failed to pull model.", model)
    return jsonify({"ok": False, "error": str(exc) or "Failed to pull model."}), 500

  response: ChatResponse = chat(
    model=model,
    messages=[
      {
        "role": "user",
        "content": prompt,
      },
    ],
  )
  return jsonify(
    {
      "ok": True,
      "model": model,
      "response": response.message.content,
    }
  )


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)