import json
import os
import threading
import time
import uuid
import asyncio
import copy
from pathlib import Path
from flask import Flask, jsonify, render_template, request, Response, stream_with_context

# Use the fixed agent
import sys
sys.path.insert(0, os.path.dirname(__file__))

from agent_orchestrator import get_orchestrator

app = Flask(__name__, template_folder="templates")

# ── Session store ─────────────────────────────────────────────────────────────

_sessions: dict = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600

# Saved conversations store (in-memory)
_saved_conversations: dict = {}
_saved_lock = threading.Lock()


def _cleanup_old_sessions():
    now = time.time()
    with _sessions_lock:
        to_del = [sid for sid, s in _sessions.items()
                  if now - s.get("last_active", 0) > SESSION_TTL]
        for sid in to_del:
            del _sessions[sid]


def _list_saved_conversations():
    with _saved_lock:
        convs = sorted(_saved_conversations.values(), key=lambda c: c.get('created', 0), reverse=True)
        out = []
        for c in convs:
            preview = ''
            for m in c.get('history', []):
                if m.get('role') == 'user' and m.get('content'):
                    preview = m.get('content')[:120]
                    break
            if not preview and c.get('history'):
                preview = c['history'][-1].get('content','')[:120]
            out.append({'id': c['id'], 'title': c.get('title',''), 'created': c.get('created'), 'preview': preview})
        return out


def _get_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "history": [], "created": time.time(), "last_active": time.time(),
            }
        _sessions[session_id]["last_active"] = time.time()
        return _sessions[session_id]


def _push_message(session_id: str, role: str, content: str) -> None:
    session = _get_session(session_id)
    with _sessions_lock:
        session["history"].append({"role": role, "content": content})
        if len(session["history"]) > 24:
            session["history"] = session["history"][-24:]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """SSE streaming chat endpoint."""
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_message:
        return jsonify({"error": "Tin nhắn không được trống"}), 400

    # Use Gemini API Key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "Vui lòng cấu hình GEMINI_API_KEY"}), 401

    _push_message(session_id, "user", user_message)
    
    def generate():
        orchestrator = get_orchestrator(api_key)
        
        # We need a synchronous bridge to async chat
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        full_response = ""
        tool_log = []

        try:
            # Create the generator
            chat_gen = orchestrator.chat(user_message)
            
            while True:
                try:
                    # Run one step of the async generator
                    event = loop.run_until_complete(anext(chat_gen))
                    etype = event.get("type")

                    if etype == "status":
                        yield f"data: {json.dumps({'type': 'status', 'text': event['text']})}\n\n"

                    elif etype == "plan":
                        yield f"data: {json.dumps({'type': 'plan', 'plan': event['plan']})}\n\n"

                    elif etype == "text":
                        chunk = event["text"]
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

                    elif etype == "error":
                        err = event["error"]
                        full_response = err
                        yield f"data: {json.dumps({'type': 'error', 'error': err})}\n\n"
                        break

                    elif etype == "done":
                        break
                except StopAsyncIteration:
                    break

        except Exception as e:
            err = f"❌ Lỗi server: {str(e)[:300]}"
            full_response = err
            yield f"data: {json.dumps({'type': 'error', 'error': err})}\n\n"

        if full_response:
            _push_message(session_id, "assistant", full_response)

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
        loop.close()

    _cleanup_old_sessions()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/conversations")
def list_conversations():
    return jsonify({"conversations": _list_saved_conversations()})


@app.route("/api/conversations/save", methods=["POST"])
def save_conversation():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    title = (data.get("title") or "").strip() or None
    new_session = str(uuid.uuid4())
    conv_meta = None

    if session_id:
        with _sessions_lock:
            session = _sessions.get(session_id)
        if session and session.get("history"):
            hist = session.get("history", [])
            # pick last user message as default title
            last_user = None
            for msg in reversed(hist):
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    last_user = msg.get("content").strip()
                    break
            if not last_user:
                last_user = f"Cuộc hội thoại {time.strftime('%Y-%m-%d %H:%M:%S')}"
            use_title = title or (last_user[:200])
            conv_id = str(uuid.uuid4())
            conv_meta = {"id": conv_id, "title": use_title, "history": hist, "created": time.time()}
            with _saved_lock:
                _saved_conversations[conv_id] = conv_meta
            # remove the old session only after we've saved it
            with _sessions_lock:
                _sessions.pop(session_id, None)

    # create new empty session to continue
    with _sessions_lock:
        _sessions[new_session] = {"history": [], "created": time.time(), "last_active": time.time()}

    return jsonify({"success": True, "new_session": new_session, "conversation": conv_meta})


@app.route("/api/conversations/load", methods=["POST"])
def load_conversation():
    data = request.get_json() or {}
    conv_id = data.get("conv_id")
    if not conv_id:
        return jsonify({"success": False, "error": "conv_id missing"}), 400
    with _saved_lock:
        conv = _saved_conversations.get(conv_id)
        if not conv:
            return jsonify({"success": False, "error": "Not found"}), 404
        hist = copy.deepcopy(conv["history"])

    new_session = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[new_session] = {"history": hist, "created": time.time(), "last_active": time.time()}

    return jsonify({"success": True, "session_id": new_session, "history": hist})


@app.route("/api/conversations/rename", methods=["POST"])
def rename_conversation():
    data = request.get_json() or {}
    conv_id = data.get("conv_id")
    title = (data.get("title") or "").strip()
    if not conv_id or not title:
        return jsonify({"success": False, "error": "Missing fields"}), 400
    with _saved_lock:
        conv = _saved_conversations.get(conv_id)
        if not conv:
            return jsonify({"success": False, "error": "Not found"}), 404
        conv["title"] = title
    return jsonify({"success": True, "conv_id": conv_id, "title": title})


@app.route("/api/session/clear", methods=["POST"])
def clear_session():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if session_id:
        with _sessions_lock:
            _sessions.pop(session_id, None)
    return jsonify({"success": True, "new_session": str(uuid.uuid4())})


@app.route("/api/key", methods=["POST"])
def set_api_key():
    data = request.get_json() or {}
    api_key = (data.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"error": "API key trống"}), 400
    
    os.environ["GEMINI_API_KEY"] = api_key
    get_orchestrator(api_key=api_key)
    return jsonify({"success": True})


@app.route("/api/status")
def status():
    api_key_gemini = os.environ.get("GEMINI_API_KEY", "")
    
    return jsonify({
        "api_key_set": bool(api_key_gemini),
        "provider": "Gemini" if api_key_gemini else "None",
        "agent_ready": bool(api_key_gemini),
    })


@app.route("/api/suggestions")
def suggestions():
    return jsonify({
        "categories": [
            {
                "name": "Thông tin doanh nghiệp",
                "questions": [
                    "Cho tôi thông tin về VIC",
                    "FPT hoạt động trong lĩnh vực gì?",
                ]
            },
            {
                "name": "Thống kê & Dữ liệu lịch sử",
                "questions": [
                    "Giá cổ phiếu VCB 1 tháng qua?",
                    "Giá cổ phiếu FPT hôm nay thế nào?",
                ]
            },
            {
                "name": "Phân tích kỹ thuật",
                "questions": [
                    "Chỉ báo RSI và SMA của VIC thế nào?",
                    "HPG đang có tín hiệu các chỉ báo kỹ thuật như nào?",
                ]
            },
            {
                "name": "Tin tức & Sentiment",
                "questions": [
                    "Tin tức mới nhất về FPT",
                    "Có tin gì xấu về STB không?",
                ]
            },
            {
                "name": "Báo cáo tài chính",
                "questions": [
                    "Cho tôi biết báo cáo tài chính mới nhất của FPT",
                    "Báo cáo tài chính của Hòa Phát (HPG)",
                ]
            },
            {
                "name": "So sánh cổ phiếu",
                "questions": [
                    "So sánh VIC và VHM",
                    "So sánh FPT và CMG",
                ]
            },
            {
                "name": "Tổng quan thị trường",
                "questions": [
                    "Tình hình VNIndex hiện tại ra sao?",
                    "VN30 hôm nay thế nào?",
                ]
            },
        ]
    })


if __name__ == "__main__":
    Path("templates").mkdir(exist_ok=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    print("=" * 62)
    print("  GOLINE VN — Multi-Agent AI System (Gemini Only)")
    print(f"  Status: {'✓ Sẵn sàng' if api_key else '⚠ Chờ Gemini API key'}")
    print("  URL     : http://127.0.0.1:5001")
    print("=" * 62)
    app.run(debug=False, host="127.0.0.1", port=5001, threaded=True)