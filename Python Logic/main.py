from pathlib import Path
import json
from datetime import datetime
import subprocess
from threading import Lock

from flask import Flask, Response, jsonify, request, send_from_directory
from ollama import ChatResponse, chat

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "HTML Front-End"
CHATS_DIR = ROOT_DIR / "Chats"

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


def sanitize_chat_id(chat_id: str) -> str:
  return "".join(char for char in chat_id if char.isalnum() or char in "-_")


def save_chat_payload(chat_id: str, payload: dict) -> None:
  CHATS_DIR.mkdir(parents=True, exist_ok=True)
  path = CHATS_DIR / f"{chat_id}.json"
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_chat_payload(chat_id: str) -> dict | None:
  path = CHATS_DIR / f"{chat_id}.json"
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def extract_first_user_message(messages: list) -> str:
  for message in messages:
    if isinstance(message, dict) and message.get("role") == "user":
      return str(message.get("content", "")).strip()
  return ""


def generate_chat_title(first_message: str, model: str) -> str:
  if not first_message or not model:
    return ""

  response: ChatResponse = chat(
    model=model,
    messages=[
      {
        "role": "system",
        "content": "Create a short chat title, 3 to 6 words. Return only the title.",
      },
      {
        "role": "user",
        "content": first_message,
      },
    ],
  )
  if not response.message or response.message.content is None:
    return ""
  title = response.message.content.strip().strip('"')
  return title[:80]


def list_saved_chats() -> list[dict]:
  if not CHATS_DIR.exists():
    return []

  chats: list[dict] = []
  for path in CHATS_DIR.glob("*.json"):
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
      chat_id = sanitize_chat_id(str(payload.get("id", "")).strip())
      if not chat_id:
        continue
      chats.append(
        {
          "id": chat_id,
          "title": payload.get("title", ""),
          "model": payload.get("model", ""),
          "updatedAt": payload.get("updatedAt", ""),
        }
      )
    except Exception:
      continue

  chats.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
  return chats


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


@app.post("/save-chat")
def save_chat():
  data = request.get_json(silent=True) or {}
  chat_id = sanitize_chat_id(str(data.get("id", "")).strip())
  messages = data.get("messages", [])
  model = str(data.get("model", "")).strip()
  title = str(data.get("title", "")).strip()

  if not chat_id:
    return jsonify({"ok": False, "error": "Chat id is required."}), 400
  if not isinstance(messages, list):
    return jsonify({"ok": False, "error": "Messages must be a list."}), 400

  existing = load_chat_payload(chat_id)
  if existing and not title:
    title = str(existing.get("title", "")).strip()

  if not title:
    first_message = extract_first_user_message(messages)
    if first_message and model:
      try:
        title = generate_chat_title(first_message, model)
      except Exception:
        title = ""

  payload = {
    "id": chat_id,
    "title": title,
    "model": model,
    "messages": messages,
    "updatedAt": f"{datetime.utcnow().isoformat()}Z",
  }
  save_chat_payload(chat_id, payload)
  return jsonify({"ok": True, "id": chat_id})


@app.get("/chats/<chat_id>")
def load_chat(chat_id: str):
  safe_id = sanitize_chat_id(chat_id)
  if not safe_id:
    return jsonify({"ok": False, "error": "Chat id is required."}), 400

  path = CHATS_DIR / f"{safe_id}.json"
  if not path.exists():
    return jsonify({"ok": False, "error": "Chat not found."}), 404

  payload = json.loads(path.read_text(encoding="utf-8"))
  payload["ok"] = True
  return jsonify(payload)


@app.get("/chats")
def list_chats():
  try:
    chats = list_saved_chats()
    return jsonify({"ok": True, "chats": chats})
  except Exception as exc:
    return jsonify({"ok": False, "error": str(exc) or "Failed to list chats."}), 500


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