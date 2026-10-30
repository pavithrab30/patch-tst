"""
Streamlit voice component — returns recognized speech text to Python.
Usage:
    from dashboard.voice_component import voice_input
    text = voice_input(lang="en-IN")
    if text:
        st.write("You said:", text)
"""
import streamlit.components.v1 as components


def voice_input(lang: str = "en-IN", key: str = "voice") -> str | None:
    """
    Renders a mic button + TTS button inside a Streamlit component.
    Returns the recognized speech text (str) or None.
    Uses Streamlit.setComponentValue to pass the result back to Python.
    """
    result = components.html(
        f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: Inter, sans-serif; }}
  body {{ background: transparent; }}
  .bar {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px;
  }}
  .mic {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 6px 14px; border-radius: 6px; border: none;
    font-size: 12px; font-weight: 600; cursor: pointer;
    background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0;
    transition: all .15s;
  }}
  .mic.on {{
    background: #dc2626; color: #fff; border-color: #dc2626;
    animation: pulse .8s infinite;
  }}
  .tts {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 6px 14px; border-radius: 6px;
    border: 1px solid #e2e8f0; background: #fff;
    color: #475569; font-size: 12px; font-weight: 600; cursor: pointer;
  }}
  .tts:hover {{ background: #f1f5f9; }}
  .status {{ font-size: 11px; color: #64748b; flex: 1; }}
  .heard-box {{
    display: none; margin-top: 6px;
    padding: 6px 10px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 6px; font-size: 12px; color: #15803d;
    font-weight: 600;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
</style>
</head>
<body>
<div class="bar">
  <button class="mic" id="micBtn" onclick="toggleMic()">🎤 <span id="micLbl">Speak</span></button>
  <button class="tts" onclick="speakLast()">🔊 Read Reply</button>
  <span class="status" id="statusTxt">Click 🎤 — speak your question — it sends automatically</span>
</div>
<div class="heard-box" id="heardBox"></div>

<script>
  // ── Streamlit component bridge ──────────────────────────────────
  function sendResult(val) {{
    window.parent.postMessage({{
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: val
    }}, "*");
  }}

  window.addEventListener("message", (e) => {{
    if (e.data && e.data.type === "streamlit:render") {{
      const args = e.data.args || {{}};
      window._lastReply = args.last_reply || "";
      window._lang      = args.lang || "{lang}";
    }}
  }});

  // Tell Streamlit component is ready
  window.parent.postMessage({{
    isStreamlitMessage: true,
    type: "streamlit:componentReady",
    apiVersion: 1
  }}, "*");

  // ── TTS ──────────────────────────────────────────────────────────
  function speakLast() {{
    const txt = window._lastReply || "";
    if (!txt) {{ setStatus("No reply to read yet."); return; }}
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(txt);
    utt.lang = window._lang || "{lang}";
    utt.rate = 0.92;
    setStatus("Reading reply…");
    utt.onend = () => setStatus("Done.");
    window.speechSynthesis.speak(utt);
  }}

  // ── STT ──────────────────────────────────────────────────────────
  let listening = false, recognition = null;

  function toggleMic() {{
    if (listening) {{ stopListening(); return; }}
    startListening();
  }}

  function startListening() {{
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {{
      setStatus("Use Chrome browser for voice input.");
      return;
    }}
    recognition = new SR();
    recognition.lang = window._lang || "{lang}";
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart = () => {{
      listening = true;
      document.getElementById("micBtn").className = "mic on";
      document.getElementById("micLbl").textContent = "Listening…";
      setStatus("Listening… speak now.");
    }};

    recognition.onresult = (e) => {{
      let interim = "";
      let final   = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {{
        if (e.results[i].isFinal) final   += e.results[i][0].transcript;
        else                      interim += e.results[i][0].transcript;
      }}
      if (interim) setStatus('Hearing: "' + interim + '"…');
      if (final) {{
        showHeard(final);
        setStatus('Sending: "' + final + '"');
        sendResult(final);
      }}
    }};

    recognition.onerror = (e) => {{
      setStatus("Mic error: " + e.error + ". Try again.");
      stopListening();
    }};

    recognition.onend = () => stopListening();
    recognition.start();
  }}

  function stopListening() {{
    listening = false;
    if (recognition) recognition.stop();
    document.getElementById("micBtn").className = "mic";
    document.getElementById("micLbl").textContent = "Speak";
  }}

  function setStatus(msg) {{
    document.getElementById("statusTxt").textContent = msg;
  }}

  function showHeard(text) {{
    const b = document.getElementById("heardBox");
    b.style.display = "block";
    b.textContent = "You said: \\"" + text + "\\"";
  }}
</script>
</body>
</html>
        """,
        height=85,
    )
    return result if isinstance(result, str) and result.strip() else None


def speak_text(text: str, lang: str = "en-IN") -> None:
    """Inject a one-shot TTS component that reads text aloud immediately."""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("\"", "\\\"")
    components.html(
        f"""<script>
        const u = new SpeechSynthesisUtterance("{safe}");
        u.lang = "{lang}"; u.rate = 0.92;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""",
        height=0,
    )
