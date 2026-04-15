import json
import re
import time
import urllib.error
import urllib.request
import requests

import streamlit as st


DEFAULT_BACKEND_URL = "http://localhost:5000"


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your assistant. Ask me anything.",
            }
        ]
    if "token" not in st.session_state:
        st.session_state.token = ""
    if "current_user" not in st.session_state:
        st.session_state.current_user = {}


def api_request(
    method: str,
    url: str,
    payload: dict | None = None,
    token: str | None = None,
) -> tuple[dict | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(body)
            return None, parsed.get("error") or f"HTTP {e.code}"
        except json.JSONDecodeError:
            return None, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return None, str(e)


def login(base_url: str, username: str, password: str) -> tuple[str | None, str | None]:
    data, err = api_request(
        "POST",
        f"{base_url.rstrip('/')}/api/auth/login",
        {"username": username, "password": password},
    )
    if err:
        return None, err

    token = (data or {}).get("token")
    if not token:
        return None, "Login succeeded but no token was returned."

    st.session_state.current_user = (data or {}).get("user") or {}
    return token, None


def ask_backend(base_url: str, token: str, prompt: str) -> tuple[dict | None, str | None]:
    data, err = api_request(
        "POST",
        f"{base_url.rstrip('/')}/api/query",
        {"query": prompt, "language": "auto", "include_voice": False},
        token=token,
    )
    if err:
        return None, err

    if not data or not (data.get("summary") or "").strip():
        return None, "No summary text found in backend response."
    return data, None


def transcribe_audio(base_url: str, token: str, audio_bytes: bytes) -> str | None:
    url = f"{base_url.rstrip('/')}/api/voice/transcribe"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("transcript")
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def ocr_image(base_url: str, token: str, image_bytes: bytes, filename: str) -> str | None:
    url = f"{base_url.rstrip('/')}/api/ocr"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files = {"file": (filename, image_bytes, "image/png")}
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=120)  # OCR can be slow
        resp.raise_for_status()
        data = resp.json()
        return data.get("text")
    except Exception as e:
        print(f"OCR error: {e}")
        return None


def get_health(base_url: str) -> tuple[dict | None, str | None]:
    return api_request("GET", f"{base_url.rstrip('/')}/api/health")


def build_response_markdown(data: dict) -> str:
    text = (data.get("summary") or "").strip()
    # Remove any hidden reasoning tags if backend/model includes them.
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or "No answer generated."


def stream_text(text: str):
    for i in range(0, len(text), 5):
        yield text[i : i + 5]
        time.sleep(0.01)


st.set_page_config(page_title="Policy Sarthi", page_icon="🛡️", layout="centered")
init_state()

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 950px; }
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.3);
}
[data-testid="stSidebar"] * { color: #1a2a6c !important; }
.stChatMessage {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 20px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.stChatMessage:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
.stButton>button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white !important;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4); }
[data-testid="stMetric"] {
    background: white !important;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
h1, h2, h3 { color: #1a2a6c !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Policy Sarthi")
st.caption("Strategic Policy Assistant")

# ── Sidebar Content ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Connection")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", value="admin123", type="password")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Login", use_container_width=True):
            token, err = login(backend_url, username, password)
            if not err:
                st.session_state.token = token
                st.success("Authenticated")
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.token = ""
            st.rerun()
    with col3:
        if st.button("Health", use_container_width=True):
            health, _ = get_health(backend_url)
            if health: st.success("Online")

    st.divider()
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
        st.rerun()
        
    st.divider()
    st.subheader("🔮 Intelligence Status")
    stats, _ = api_request("GET", f"{backend_url.rstrip('/')}/api/analytics")
    health, _ = get_health(backend_url)
    if health: st.caption(f"Knowledge: {health.get('vectorsIndexed', 0)} vectors")
    if stats: st.caption(f"Learned Lessons: {stats.get('total', 0)}")

# ── Main Chat Context ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about hospital policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 Searching across brains...")

        if not st.session_state.token:
            token, _ = login(backend_url, username, password)
            st.session_state.token = token
        
        response_data, err = ask_backend(backend_url, st.session_state.token, prompt)
        placeholder.empty()

        if err:
            final_text = f"Backend error: {err}"
        else:
            final_text = build_response_markdown(response_data or {})
            st.write_stream(stream_text(final_text))
            
            applied = (response_data or {}).get("appliedLessons", [])
            if applied:
                st.info(f"✨ **Experience Applied:** I improved this answer using a verified lesson: *\"{applied[0]}\"*")

    st.session_state.messages.append({"role": "assistant", "content": final_text})

# ── Multimodal Features (Bottom) ───────────────────────────────────
st.divider()
colA, colB = st.columns(2)
with colA:
    st.subheader("Visual Analysis")
    img_file = st.file_uploader("Upload doc for OCR", type=["pdf", "png", "jpg"])
with colB:
    st.subheader("Voice Query")
    audio_val = st.audio_input("Speak your query")

if img_file:
    key = f"ocr_{img_file.name}"
    if key not in st.session_state:
        st.session_state[key] = False
    if not st.session_state[key]:
        with st.chat_message("assistant"):
            st.markdown("🔍 Analyzing document...")
            text = ocr_image(backend_url, st.session_state.token, img_file.read(), img_file.name)
            if text:
                st.markdown(f"**Extracted Snippet:** {text[:200]}...")
                st.session_state.messages.append({"role": "user", "content": f"[Document Analysis]: {text}"})
                res, _ = ask_backend(backend_url, st.session_state.token, text)
                ans = build_response_markdown(res or {})
                st.write_stream(stream_text(ans))
                st.session_state.messages.append({"role": "assistant", "content": ans})
        st.session_state[key] = True

if audio_val:
    if "last_v" not in st.session_state or audio_val != st.session_state.last_v:
        st.session_state.last_v = audio_val
        with st.chat_message("assistant"):
            st.markdown("🎤 Transcribing voice...")
            txt = transcribe_audio(backend_url, st.session_state.token, audio_val.read())
            if txt:
                st.markdown(f"**Transcript:** {txt}")
                st.session_state.messages.append({"role": "user", "content": txt})
                res, _ = ask_backend(backend_url, st.session_state.token, txt)
                ans = build_response_markdown(res or {})
                st.write_stream(stream_text(ans))
                st.session_state.messages.append({"role": "assistant", "content": ans})


