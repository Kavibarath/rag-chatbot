import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";
const STORAGE_KEY = "rag-conversations";

// ---------- localStorage helpers ------------------------------------------

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convs) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
  } catch {
    /* quota exceeded — silently drop */
  }
}

function newId() {
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function titleFrom(question) {
  const trimmed = question.trim().replace(/\s+/g, " ");
  return trimmed.length > 40 ? trimmed.slice(0, 40) + "…" : trimmed;
}

// Strip any source references from answer text.
// The source chips below already show this info, so the answer prose should be clean.
function stripCitations(text) {
  if (!text) return text;
  return text
    // "Sources: X, Y, Z" / "Source: X" — anywhere in the text
    .replace(/\s*Sources?\s*:\s*[^\n]*/gi, "")
    // Bracketed refs like [filename.pdf:16]
    .replace(/\s*\[[^\]]*?\.(?:pdf|docx|pptx|txt|md):\s*\d+\]/gi, "")
    // Bare refs like "filename.pptx:16" (no brackets) — used after "Sources:" strip
    .replace(/\s*\(?[^\s()]+\.(?:pdf|docx|pptx|txt|md):\s*\d+\)?/gi, "")
    // "According to Lecture X" / "In slide N" leftovers
    .replace(/\s*According to [^\n.]*\./gi, "")
    .replace(/\s*In (?:slide|page|Lecture)\s+\d+[^\n.]*\./gi, "")
    // Trailing dangling commas, semicolons, whitespace
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/[,;]\s*(?=[.\n]|$)/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

// ---------- Icons (inline SVG, no dependency) -----------------------------

const IconPlus = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const IconChat = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const IconTrash = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6" />
  </svg>
);

const IconSend = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const IconSparkle = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3l1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2z" />
  </svg>
);

// ---------- Main component ------------------------------------------------

export default function App() {
  const [conversations, setConversations] = useState(loadConversations);
  const [activeId, setActiveId] = useState(null); // null = new blank chat
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const scrollRef = useRef(null);

  // Persist conversations whenever they change
  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // ----- conversation management -----

  function startNewChat() {
    setActiveId(null);
    setMessages([]);
    setInput("");
  }

  function loadConversation(id) {
    const conv = conversations.find((c) => c.id === id);
    if (!conv) return;
    setActiveId(id);
    setMessages(conv.messages);
    setInput("");
  }

  function deleteConversation(id, e) {
    e.stopPropagation();
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) startNewChat();
  }

  function persistCurrent(newMessages) {
    // If active is a saved conversation, update it. Else create a new one from the first user question.
    if (activeId) {
      setConversations((prev) =>
        prev.map((c) => (c.id === activeId ? { ...c, messages: newMessages } : c))
      );
    } else {
      const firstUser = newMessages.find((m) => m.role === "user");
      if (!firstUser) return;
      const id = newId();
      const conv = {
        id,
        title: titleFrom(firstUser.text),
        createdAt: Date.now(),
        messages: newMessages,
      };
      setConversations((prev) => [conv, ...prev]);
      setActiveId(id);
    }
  }

  // ----- streaming send -----

  async function sendQuestion(question) {
    setIsStreaming(true);
    // Capture the pre-send state so we can rebuild finalMessages later
    const priorMessages = messages;
    setMessages([
      ...priorMessages,
      { role: "user", text: question },
      { role: "assistant", text: "", sources: [], pending: true },
    ]);

    // Track streamed content locally so we can persist it once at the end
    let assistantText = "";
    let assistantSources = [];

    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try {
            event = JSON.parse(line.slice(6));
          } catch {
            continue;
          }

          if (event.type === "sources") {
            assistantSources = event.sources;
            updateAssistant({ sources: event.sources });
          } else if (event.type === "token") {
            assistantText += event.text;
            updateAssistant({ text: assistantText });
          } else if (event.type === "error") {
            assistantText = `Error: ${event.error}`;
            updateAssistant({ text: assistantText, sources: [] });
          }
        }
      }
    } catch (err) {
      assistantText = `Error: ${err.message}`;
      updateAssistant({ text: assistantText, sources: [] });
    } finally {
      // Build finalMessages once, outside any state updater — so persistCurrent
      // runs exactly once (React StrictMode double-fires setState updaters in dev).
      const finalMessages = [
        ...priorMessages,
        { role: "user", text: question },
        {
          role: "assistant",
          text: assistantText || "I don't have that in the provided notes.",
          sources: assistantSources,
          pending: false,
        },
      ];
      setMessages(finalMessages);
      persistCurrent(finalMessages);
      setIsStreaming(false);
    }
  }

  function updateAssistant(patch) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") next[next.length - 1] = { ...last, ...patch };
      return next;
    });
  }

  function handleSubmit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || isStreaming) return;
    setInput("");
    sendQuestion(q);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  // ----- render -----

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={startNewChat}>
          <IconPlus />
          <span>New chat</span>
        </button>

        <div className="sidebar-label">Recent</div>

        <nav className="conv-list">
          {conversations.length === 0 && (
            <p className="sidebar-empty">No conversations yet.</p>
          )}
          {conversations.map((c) => (
            <button
              key={c.id}
              className={`conv-item ${c.id === activeId ? "active" : ""}`}
              onClick={() => loadConversation(c.id)}
              title={c.title}
            >
              <IconChat />
              <span className="conv-title">{c.title}</span>
              <span
                className="conv-delete"
                onClick={(e) => deleteConversation(c.id, e)}
                aria-label="Delete conversation"
              >
                <IconTrash />
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="footer-label">Powered by</span>
          <span className="footer-tech">Llama 3.3 · Chroma · MiniLM</span>
        </div>
      </aside>

      <section className="main">
        <header className="header">
          <div>
            <h1>IT Notes Chat</h1>
            <p>Grounded Q&A over your lecture notes — every claim cited.</p>
          </div>
        </header>

        <main className="chat" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty">
              <div className="empty-icon"><IconSparkle /></div>
              <h2>Ask anything from your notes</h2>
              <p>The model only uses what's in your slides — no invented facts, and it says so when it doesn't know.</p>
              <div className="examples">
                <button onClick={() => sendQuestion("what is a system call")}>
                  What is a system call?
                </button>
                <button onClick={() => sendQuestion("compare asymmetric and symmetric multiprocessing")}>
                  Compare AMP and SMP
                </button>
                <button onClick={() => sendQuestion("what is NoSQL")}>
                  What is NoSQL?
                </button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="bubble-wrap">
                <div className="role-badge">{msg.role === "user" ? "You" : "AI"}</div>
                <div className="bubble">
                  {(msg.role === "assistant" ? stripCitations(msg.text) : msg.text) ||
                    (msg.pending && <span className="cursor">▍</span>)}
                  {msg.pending && msg.text && <span className="cursor">▍</span>}
                </div>
              </div>
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <div className="sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-chip" title={`relevance: ${s.score}`}>
                      {s.cite}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </main>

        <form className="input-bar" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something from your notes…"
            rows={1}
            disabled={isStreaming}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={isStreaming || !input.trim()}
            aria-label="Send"
          >
            <IconSend />
          </button>
        </form>
      </section>
    </div>
  );
}
