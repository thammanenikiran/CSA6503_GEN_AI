const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

function addMessage(role, text, sources, status) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sources && sources.length) {
    const src = document.createElement("span");
    src.className = "sources";
    src.textContent = "Sources: " + sources.join(" · ");
    bubble.appendChild(src);
  }

  if (status && status !== "ok") {
    const badge = document.createElement("span");
    badge.className = "status-badge";
    badge.textContent = status.replace("_", " ");
    bubble.appendChild(document.createElement("br"));
    bubble.appendChild(badge);
  }

  wrap.appendChild(bubble);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = "typing-indicator";
  wrap.innerHTML = '<div class="bubble typing">Thinking…</div>';
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

async function sendQuery(query) {
  if (!query || !query.trim()) {
    addMessage("bot", "Please type a question before sending.");
    return;
  }
  addMessage("user", query);
  chatInput.value = "";
  addTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    removeTyping();
    addMessage("bot", data.answer, data.sources, data.status);
  } catch (err) {
    removeTyping();
    addMessage("bot", "Network error -- could not reach the chatbot server.");
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuery(chatInput.value);
});

function sendSample(btn) {
  sendQuery(btn.textContent);
}
