import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st
from groq import Groq


# Optional voice input deps
try:
    from audio_recorder_streamlit import audio_recorder  # type: ignore
    HAS_AUDIO_RECORDER = True
except Exception:
    HAS_AUDIO_RECORDER = False

try:
    import speech_recognition as sr  # type: ignore
    HAS_SPEECH_RECOGNITION = True
except Exception:
    HAS_SPEECH_RECOGNITION = False


# === PAGE CONFIG ===
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")


# === MODELS & PERSONAS ===
AVAILABLE_MODELS: List[str] = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

PERSONALITIES: Dict[str, str] = {
    "Friendly": "Be friendly, warm, and encouraging.",
    "Formal": "Be professional, concise, and neutral in tone.",
    "Enthusiastic": "Be upbeat and energetic.",
    "Playful": "Be witty and playful.",
    "Technical": "Be precise, detailed, and technical.",
    "Concise": "Be brief and to the point.",
}


# === THEME & CSS ===
def inject_theme_css(theme: str) -> None:
    """Inject CSS with variables for the selected theme (no JS needed)."""
    if theme == "dark":
        vars_css = {
            "bg": "#0b1220",
            "text": "#e5e7eb",
            "muted": "#9aa4b2",
            "primary": "#7c7eff",
            "primary_hover": "#9aa0ff",
            "bubble_user_bg": "#1e293b",
            "bubble_bot_bg": "#0f172a",
            "bubble_border": "#243046",
            "input_bg": "#0f172a",
            "input_border": "#243046",
            "shadow": "0 10px 30px rgba(0,0,0,0.35)",
        }
    else:
        vars_css = {
            "bg": "#f7f8fb",
            "text": "#101828",
            "muted": "#475467",
            "primary": "#4f46e5",
            "primary_hover": "#4338ca",
            "bubble_user_bg": "#e8ecff",
            "bubble_bot_bg": "#ffffff",
            "bubble_border": "#e5e7eb",
            "input_bg": "#ffffff",
            "input_border": "#e5e7eb",
            "shadow": "0 6px 20px rgba(16,24,40,0.08)",
        }

    st.markdown(
        f"""
        <style>
        :root {{
          --bg: {vars_css['bg']};
          --text: {vars_css['text']};
          --muted: {vars_css['muted']};
          --primary: {vars_css['primary']};
          --primary-hover: {vars_css['primary_hover']};
          --bubble-user-bg: {vars_css['bubble_user_bg']};
          --bubble-bot-bg: {vars_css['bubble_bot_bg']};
          --bubble-border: {vars_css['bubble_border']};
          --input-bg: {vars_css['input_bg']};
          --input-border: {vars_css['input_border']};
          --shadow: {vars_css['shadow']};
          --radius: 14px;
          --chat-max-width: 860px;
          --inputbar-height: 96px;
        }}

        html, body, [data-testid="stAppViewContainer"] {{
          background: var(--bg) !important;
          color: var(--text) !important;
        }}
        .block-container {{
          padding-top: 8px;
          padding-bottom: calc(var(--inputbar-height) + 26px);
          max-width: var(--chat-max-width);
        }}

        /* Header bar */
        .app-header {{
          position: sticky;
          top: 0;
          z-index: 50;
          background: transparent;
          padding: 16px 0 8px;
        }}
        .app-header .title {{
          display: flex;
          align-items: center;
          gap: 10px;
          color: var(--text);
          font-weight: 700;
          font-size: 22px;
          letter-spacing: 0.2px;
        }}
        .app-header .title .badge {{
          padding: 2px 8px;
          border-radius: 999px;
          background: rgba(79,70,229,0.08);
          color: var(--primary);
          font-size: 12px;
          border: 1px solid rgba(79,70,229,0.18);
        }}

        /* Bubbles */
        .bubble {{
          border-radius: var(--radius);
          padding: 14px 16px;
          margin: 4px 0 8px;
          border: 1px solid var(--bubble-border);
          box-shadow: var(--shadow);
          animation: fadeInUp 240ms ease-out both;
          line-height: 1.55;
          font-size: 16px;
          color: var(--text);
          background: var(--bubble-bot-bg);
        }}
        .bubble-user {{
          background: var(--bubble-user-bg);
          border-top-right-radius: 6px;
        }}
        .bubble-bot {{
          background: var(--bubble-bot-bg);
          border-top-left-radius: 6px;
        }}

        /* Typing */
        .typing {{ display: inline-flex; gap: 5px; align-items: center; }}
        .typing .dot {{
          width: 6px; height: 6px; border-radius: 999px; background: var(--muted);
          animation: blink 1.4s infinite ease-in-out;
        }}
        .typing .dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing .dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes blink {{
          0%, 80%, 100% {{ opacity: 0.3; transform: translateY(0); }}
          40% {{ opacity: 1; transform: translateY(-2px); }}
        }}
        @keyframes fadeInUp {{
          from {{ opacity: 0; transform: translateY(6px); }}
          to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Bottom input bar */
        .inputbar {{
          position: fixed;
          left: 50%;
          transform: translateX(-50%);
          bottom: 12px;
          width: min(100% - 24px, var(--chat-max-width));
          background: var(--input-bg);
          border: 1px solid var(--input-border);
          border-radius: 16px;
          padding: 12px 12px 14px;
          box-shadow: var(--shadow);
          z-index: 60;
          color: var(--text);
        }}
        .inputbar input[type="text"], .inputbar textarea {{
          border-radius: 12px;
          border: 1px solid var(--input-border) !important;
          background: var(--input-bg) !important;
          color: var(--text) !important;
        }}
        .inputbar input::placeholder, .inputbar textarea::placeholder {{
          color: {vars_css['muted']};
        }}
        .inputbar .send-btn button {{
          width: 100%;
          background: var(--primary);
          color: white;
          border-radius: 12px;
          border: 0;
          transition: transform 120ms ease, background 160ms ease;
        }}
        .inputbar .send-btn button:hover {{ background: var(--primary-hover); transform: translateY(-1px); }}
        .inputbar .send-btn button:active {{ transform: translateY(0); }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, rgba(79,70,229,0.06), rgba(79,70,229,0.00) 30%),
                      linear-gradient(0deg, var(--bg), var(--bg));
          border-right: 1px solid var(--bubble-border);
          color: var(--text);
        }}
        .sidebar-card {{
          border: 1px solid var(--bubble-border);
          border-radius: 14px;
          padding: 12px;
          background: var(--input-bg);
          box-shadow: var(--shadow);
          color: var(--text);
        }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {{
          color: var(--text);
        }}

        /* Responsive */
        @media (max-width: 640px) {{
          .block-container {{ padding-left: 10px; padding-right: 10px; }}
          .inputbar {{ bottom: 8px; border-radius: 14px; }}
          .bubble {{ font-size: 15px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# === HELPERS ===
def get_client() -> Optional[Groq]:
    """Return a Groq client if an API key is set via st.secrets or environment."""
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def build_system_message(persona: str) -> Dict[str, str]:
    persona_instruction = PERSONALITIES.get(persona, PERSONALITIES["Friendly"])
    return {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            f"{persona_instruction} Use clear, readable formatting."
        ),
    }


def init_state() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "persona" not in st.session_state:
        st.session_state.persona = "Friendly"
    if "model" not in st.session_state:
        st.session_state.model = AVAILABLE_MODELS[0]
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "voice_in" not in st.session_state:
        st.session_state.voice_in = False
    if "voice_out" not in st.session_state:
        st.session_state.voice_out = False
    if "prefill_text" not in st.session_state:
        st.session_state.prefill_text = ""


def render_sidebar(client_ready: bool) -> None:
    with st.sidebar:
        st.markdown("### 🤖 Bot")
        st.image(
            "https://raw.githubusercontent.com/github/explore/main/topics/chatbot/chatbot.png",
            width=72,
        )
        st.markdown(
            '<div class="sidebar-card">\n'
            "<b>Welcome!</b> Choose a model and personality. Toggle theme anytime."
            "</div>",
            unsafe_allow_html=True,
        )

        st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=(
                AVAILABLE_MODELS.index(st.session_state.model)
                if st.session_state.model in AVAILABLE_MODELS
                else 0
            ),
            key="model",
        )
        st.selectbox(
            "Personality",
            list(PERSONALITIES.keys()),
            index=list(PERSONALITIES.keys()).index(st.session_state.persona),
            key="persona",
        )

        theme_choice = st.radio(
            "Theme",
            ["Light", "Dark"],
            horizontal=True,
            index=0 if st.session_state.theme == "light" else 1,
            key="theme_radio",
        )
        st.session_state.theme = "light" if theme_choice == "Light" else "dark"

        st.toggle("Voice input", value=st.session_state.voice_in, key="voice_in")
        st.toggle("Voice output (placeholder)", value=st.session_state.voice_out, key="voice_out")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
        with col2:
            st.write(f"Msgs: {len(st.session_state.messages)}")

        st.markdown("---")
        if not client_ready:
            st.info("Set `GROQ_API_KEY` in env or `st.secrets` to chat.")


def render_header() -> None:
    st.markdown(
        """
        <div class=\"app-header\">
          <div class=\"title\">🤖 Groq Chatbot <span class=\"badge\">beta</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content", "")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(
                f"<div class='bubble {'bubble-user' if role=='user' else 'bubble-bot'}'>{content}</div>",
                unsafe_allow_html=True,
            )


def build_api_messages(state_msgs: List[Dict[str, Any]], persona: str) -> List[Dict[str, str]]:
    api_msgs: List[Dict[str, str]] = [build_system_message(persona)]
    for m in state_msgs:
        if m.get("role") == "user":
            api_msgs.append({"role": "user", "content": m.get("content", "")})
        elif m.get("role") == "assistant":
            api_msgs.append({"role": "assistant", "content": m.get("content", "")})
    return api_msgs


def stream_completion(
    client: Groq,
    model: str,
    messages_for_api: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> str:
    full_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='bubble bubble-bot'><span class='typing'><span class='dot'></span><span class='dot'></span><span class='dot'></span></span></div>",
            unsafe_allow_html=True,
        )
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta: Optional[str] = None
                try:
                    delta = chunk.choices[0].delta.content  # type: ignore[attr-defined]
                except Exception:
                    delta = None
                if delta:
                    full_text += delta
                    placeholder.markdown(
                        f"<div class='bubble bubble-bot'>{full_text}</div>",
                        unsafe_allow_html=True,
                    )
            placeholder.markdown(
                f"<div class='bubble bubble-bot'>{full_text}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            placeholder.empty()
            st.error(f"❌ Error calling Groq API: {e}")
    return full_text


def transcribe_audio_bytes_to_text(audio_bytes: bytes) -> Optional[str]:
    if not HAS_SPEECH_RECOGNITION:
        return None
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        # Uses Google Web Speech API (free, requires internet)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return None


def main() -> None:
    init_state()

    client = get_client()
    render_sidebar(client_ready=client is not None)
    inject_theme_css(st.session_state.theme)
    render_header()

    # Chat history
    render_chat_history()

    # Bottom input bar
    with st.container():
        st.markdown('<div class="inputbar">', unsafe_allow_html=True)
        with st.form("chat-input-form", clear_on_submit=True):
            if st.session_state.voice_in and HAS_AUDIO_RECORDER:
                cols = st.columns([6, 2, 2])
            else:
                cols = st.columns([7, 3])

            if len(cols) == 3:
                c1, cV, c2 = cols
            else:
                c1, c2 = cols
                cV = None

            with c1:
                user_text: str = st.text_input(
                    "Type your question here…",
                    key="user_text",
                    value=st.session_state.get("prefill_text", ""),
                    label_visibility="collapsed",
                )
                st.session_state.prefill_text = user_text  # keep current value

            if cV is not None and st.session_state.voice_in:
                with cV:
                    if HAS_AUDIO_RECORDER:
                        st.caption("Voice input")
                        audio_bytes = audio_recorder(text="Record", icon_name="microphone")  # type: ignore
                        if audio_bytes:
                            text = transcribe_audio_bytes_to_text(audio_bytes)
                            if text:
                                st.success(f"Heard: {text}")
                                st.session_state.prefill_text = text
                                st.rerun()
                            else:
                                st.warning("Could not transcribe audio.")
                    else:
                        st.info("Install audio-recorder-streamlit for voice input.")

            with c2:
                send_clicked = st.form_submit_button("✈️ Send", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if client is None:
        st.stop()

    if send_clicked and (st.session_state.prefill_text and st.session_state.prefill_text.strip()):
        text = st.session_state.prefill_text.strip()

        with st.chat_message("user"):
            st.markdown(
                f"<div class='bubble bubble-user'>{text}</div>",
                unsafe_allow_html=True,
            )

        st.session_state.messages.append({"role": "user", "content": text})

        api_messages = build_api_messages(st.session_state.messages, st.session_state.persona)
        bot_reply = stream_completion(
            client=client,
            model=st.session_state.model,
            messages_for_api=api_messages,
        )
        if bot_reply:
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.session_state.prefill_text = ""


if __name__ == "__main__":
    main()

