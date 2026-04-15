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


st.set_page_config(page_title="Policy Sarthi", page_icon=":speech_balloon:", layout="centered")
init_state()

# ── Premium Design System (Glassmorphism Restoration) ────────────────
css_style = """
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
"""
st.markdown(css_style, unsafe_allow_html=True)

st.title("Policy Sarthi")
st.caption("Policy assistant UI integrated with backend APIs")

# ── Sidebar Content ─────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Connection")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    username = st.text_input("Username", value="admin")
    password = st.text_input("Password", value="admin123", type="password")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Login", use_container_width=True):
            token, err = login(backend_url, username, password)
            if err: st.error(err)
            else:
                st.session_state.token = token
                st.write("Logged in as:", st.session_state.current_user.get("displayName", username))
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.token = ""
            st.session_state.current_user = {}
            st.rerun()
    with col3:
        if st.button("Health", use_container_width=True):
            health, err = get_health(backend_url)
            if err: st.error(f"Backend unreachable: {err}")
            else: st.success(f"Status: {health.get('status', 'ok')}")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "New chat started. How can I help you?"}]
        st.rerun()

    mode = "Connected" if st.session_state.token else "Not authenticated"
    st.caption(f"Current: {mode}")
    user = st.session_state.current_user
    if user:
        display_name = user.get("displayName", user.get("username", "-"))
        role = user.get("role", "-")
        st.caption(f"User: {display_name} ({role})")
        
    st.divider()
    st.subheader("🔮 Intelligence Status")
    stats, err = api_request("GET", f"{backend_url.rstrip('/')}/api/analytics")
    if not err and stats:
        pos = stats.get("positive", 0)
        neg = stats.get("negative", 0)
        total = stats.get("total", 0)
        health, _ = get_health(backend_url)
        policy_vecs = (health or {}).get("vectorsIndexed", 0)
        
        st.markdown(f"""
        <div style="background: rgba(79, 172, 254, 0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(79, 172, 254, 0.3);">
            <div style="font-size: 0.9rem; opacity: 0.8;">Policy Knowledge Base</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #4facfe;">{policy_vecs} Vectors</div>
            <hr style="margin: 10px 0; opacity: 0.2;">
            <div style="font-size: 0.9rem; opacity: 0.8;">Experience Memories</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #00f2fe;">{total} Lessons</div>
        </div>
        """, unsafe_allow_html=True)
        
        if total > 0:
            rate = int((pos / total) * 100) if total > 0 else 100
            st.write("")
            st.caption(f"Success Rate: {rate}%")
            st.progress(rate / 100)
    
    st.caption("✨ Semantic Memory Active")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking.......")

        if not st.session_state.token:
            token, err = login(backend_url, username, password)
            if err:
                final_text = f"Login failed: {err}. Use sidebar credentials and click Login."
            else:
                st.session_state.token = token

        if st.session_state.token:
            response_data, err = ask_backend(backend_url, st.session_state.token, prompt)
            if err:
                final_text = f"Backend error: {err}"
            else:
                final_text = build_response_markdown(response_data or {})

        thinking_placeholder.empty()
        st.write_stream(stream_text(final_text))
        
        # Highlight if a semantic lesson was applied
        if st.session_state.token and not err:
            applied = response_data.get("appliedLessons", [])
            if applied:
                st.info(f"✨ **Experience Applied:** I've improved this answer using a verified lesson from my memory: *\"{applied[0]}\"*")

    st.session_state.messages.append({"role": "assistant", "content": final_text})


st.divider()
st.subheader("Visual Analysis (OCR)")
image_file = st.file_uploader("Upload an image or document (PDF/PNG/JPG) for analysis", type=["png", "jpg", "jpeg", "pdf"])

if image_file is not None:
    # Use a unique key based on file content/name to avoid re-triggering on every rerun
    file_key = f"last_ocr_{image_file.name}"
    if file_key not in st.session_state:
        st.session_state[file_key] = False
        
    if not st.session_state[file_key]:
        image_bytes = image_file.read()
        
        with st.chat_message("user"):
            st.markdown(f"📷 *Uploaded file: {image_file.name}*")
            
        with st.chat_message("assistant"):
            ocr_placeholder = st.empty()
            ocr_placeholder.markdown("🔍 **Analyzing document via Sarvam AI Vision...** (this may take 10-20 seconds)")
            
            if not st.session_state.token:
                token, err = login(backend_url, username, password)
                if not err:
                    st.session_state.token = token
            
            extracted_text = ocr_image(backend_url, st.session_state.token, image_bytes, image_file.name)
            
            if not extracted_text:
                ocr_placeholder.error("Failed to extract text from document.")
            else:
                ocr_placeholder.markdown(f"**Extracted Text Snippet:**\n\n{extracted_text[:300]}...")
                st.session_state.messages.append({"role": "user", "content": f"[Document Analysis of {image_file.name}]:\n{extracted_text}"})
                
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("Thinking...")
                
                # Automatically query RAG with extracted text
                response_data, err = ask_backend(backend_url, st.session_state.token, extracted_text)
                if err:
                    final_text = f"Backend error: {err}"
                else:
                    final_text = build_response_markdown(response_data or {})
                    
                thinking_placeholder.empty()
                st.write_stream(stream_text(final_text))
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                st.session_state[file_key] = True


audio_value = st.audio_input("Record a voice message")
if audio_value is not None:
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None
        
    if audio_value != st.session_state.last_audio:
        st.session_state.last_audio = audio_value
        audio_bytes = audio_value.read()
        
        with st.chat_message("user"):
            st.markdown("🎤 *Voice message*")
            
        with st.chat_message("assistant"):
            transcript_placeholder = st.empty()
            transcript_placeholder.markdown("Transcribing audio...")
            
            if not st.session_state.token:
                token, err = login(backend_url, username, password)
                if not err:
                    st.session_state.token = token
            
            transcript = transcribe_audio(backend_url, st.session_state.token, audio_bytes)
            
            if not transcript:
                transcript_placeholder.error("Failed to transcribe audio.")
            else:
                transcript_placeholder.markdown(f"**Transcript:** {transcript}")
                st.session_state.messages.append({"role": "user", "content": transcript})
                
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("Thinking...")
                
                response_data, err = ask_backend(backend_url, st.session_state.token, transcript)
                if err:
                    final_text = f"Backend error: {err}"
                else:
                    final_text = build_response_markdown(response_data or {})
                    
                thinking_placeholder.empty()
                st.write_stream(stream_text(final_text))
                st.session_state.messages.append({"role": "assistant", "content": final_text})  # Mark as done for this file


