"""Chat-first home page: the AI assistant is the main feature of the web app.

Use in app.py:
    import chat_page
    chat_page.render(selected_location)   # selected_location = province code from the sidebar, e.g. "tp_hcm"
"""
import time

import streamlit as st

import assistant
import ui_theme as ui

SUGGESTIONS = [
    "Thời tiết Sài Gòn bây giờ thế nào?",
    "Miền Trung có bão hay mưa lớn không?",
    "Which province has the worst air right now?",
    "Hà Nội 6 giờ tới bao nhiêu độ?",
    "Bão Yagi năm 2024 gây thiệt hại gì?",
    "Tháng 7/2024 Hà Nội mưa bao nhiêu?",
]

_CSS = f"""
<style>
.wq-chat-hero {{ text-align: center; padding: 34px 12px 10px; }}
.wq-chat-hero .logo {{ width: 56px; height: 56px; border-radius: 18px; background: {ui.ACCENT}; display: inline-grid;
  place-items: center; box-shadow: 0 10px 30px -12px {ui.ACCENT}; margin-bottom: 18px; }}
.wq-chat-hero h1 {{ font-size: 2.1rem !important; font-weight: 500 !important; margin: 0 0 10px !important; padding: 0 !important; }}
.wq-chat-hero p {{ color: {ui.INK_SOFT}; max-width: 640px; margin: 0 auto; font-size: 1rem; line-height: 1.6; }}
.wq-status {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 18px 0 26px; }}
.wq-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
  background: {ui.PANEL}; border: 1px solid {ui.RULE}; color: {ui.INK_SOFT}; font-size: .82rem; }}
.wq-chip b {{ color: {ui.INK}; font-weight: 600; }}
.wq-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.wq-sugg-title {{ color: {ui.INK_SOFT}; font-size: .85rem; margin: 4px 0 8px; }}
.wq-meta {{ color: {ui.INK_SOFT}; font-size: .78rem; margin-top: 2px; }}
[data-testid="stChatMessage"] hr {{ margin: 14px 0 10px; border-color: {ui.RULE}; }}
[data-testid="stChatMessage"] ul {{ margin-bottom: 0; }}
[data-testid="stChatMessage"] a {{ color: {ui.ACCENT}; text-decoration: none; }}
[data-testid="stChatMessage"] a:hover {{ text-decoration: underline; }}
[data-testid="stChatInput"] {{ border-radius: 16px; }}
</style>
"""


def _css():
    # Markdown ends an HTML block at a blank line, so remove blank lines
    st.markdown("\n".join(l.strip() for l in _CSS.splitlines() if l.strip()), unsafe_allow_html=True)


def _status_chips():
    ok, msg = assistant.check_ollama()
    color = "#22a35a" if ok else "#d93b3b"
    chips = [f'<span class="wq-chip"><span class="wq-dot" style="background:{color}"></span>{msg}</span>']
    try:
        w = assistant._read("weather_current.csv")
        if w is not None and not w.empty:
            latest = str(w["time"].max()).replace("T", " ")
            chips.append(f'<span class="wq-chip">Data updated <b>{latest}</b></span>')
    except Exception:
        pass
    try:
        n = assistant.news_count()
        if n:
            chips.append(f'<span class="wq-chip"><b>{n:,}</b> news articles scraped</span>')
    except Exception:
        pass
    chips.append('<span class="wq-chip"><b>34</b> provinces</span>')
    st.markdown('<div class="wq-status">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def _hero():
    icon = ('<svg width="30" height="30" viewBox="0 0 44 48"><circle cx="27" cy="18" r="7" fill="#f7dfb0"/>'
            '<path d="M12 33 a7 7 0 0 1 3 -13 a9 9 0 0 1 17 3 a5.5 5.5 0 0 1 0 10 z" fill="#fff"/></svg>')
    st.markdown(
        f'<div class="wq-chat-hero"><div class="logo">{icon}</div>'
        "<h1>Ask about weather and air quality in Vietnam</h1>"
        "<p>Answers use live data for 34 provinces, our own forecast model, news from Vietnamese "
        "newspapers, 5 years of daily history and past storms and floods from Wikipedia. Every answer shows its sources. If we do not have the data, the assistant says so. "
        "You can ask in English or Vietnamese.</p></div>",
        unsafe_allow_html=True,
    )


def _suggestions():
    st.markdown('<div class="wq-sugg-title">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state.wq_pending = q
            st.rerun()


def _show_message(m):
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🌤️"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("seconds") is not None:
                st.markdown(f'<div class="wq-meta">Answered in {m["seconds"]:.1f} s</div>', unsafe_allow_html=True)
            if m.get("context"):
                with st.expander("Data given to the AI"):
                    st.code(m["context"], language=None)


def render(selected_location=None):
    _css()
    st.session_state.setdefault("wq_chat", [])
    history = st.session_state.wq_chat

    with st.sidebar:
        if history and st.button("New chat", icon=":material/add_comment:", use_container_width=True):
            st.session_state.wq_chat = []
            st.rerun()

    if not history:
        _hero()
        _status_chips()
        _suggestions()
    else:
        for m in history:
            _show_message(m)

    typed = st.chat_input("Ask about weather, air quality or storms in Vietnam...")
    question = typed or st.session_state.pop("wq_pending", None)
    if not question:
        return

    _show_message({"role": "user", "content": question})
    with st.chat_message("assistant", avatar="🌤️"):
        start = time.time()
        try:
            answer = st.write_stream(assistant.ask(question, history, selected_location))
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
            st.error(answer)
        seconds = time.time() - start
        st.markdown(f'<div class="wq-meta">Answered in {seconds:.1f} s</div>', unsafe_allow_html=True)
        try:
            context = assistant.build_context(question, selected_location)
        except Exception:
            context = ""
        if context:
            with st.expander("Data given to the AI"):
                st.code(context, language=None)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer if isinstance(answer, str) else str(answer),
                    "seconds": seconds, "context": context})
    if not typed or len(history) == 2:  # first answer or a suggestion: redraw without the welcome screen
        st.rerun()
