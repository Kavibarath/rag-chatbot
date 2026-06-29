import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const scrollRef = useRef(null);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function sendQuestion(question) {
    setIsStreaming(true);
    // Push the user message + a placeholder assistant message we'll stream into
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "", sources: [], pending: true },
    ]);

    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        throw new Error(`Server returned ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE messages end with \n\n
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
          handleEvent(event);
        }
      }
    } catch (err) {
      handleEvent({ type: "error", error: err.message });
    } finally {
      // Mark the assistant message as done streaming
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") next[next.length - 1] = { ...last, pending: false };
        return next;
      });
      setIsStreaming(false);
    }
  }

  function handleEvent(event) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role !== "assistant") return prev;

      if (event.type === "sources") {
        next[next.length - 1] = { ...last, sources: event.sources };
      } else if (event.type === "token") {
        next[next.length - 1] = { ...last, text: last.text + event.text };
      } else if (event.type === "error") {
        next[next.length - 1] = {
          ...last,
          text: `Error: ${event.error}`,
          sources: [],
        };
      }
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

  return (
    <div className="app">
      <header className="header">
        <h1>IT Notes Chat</h1>
        <p>Grounded Q&A over your university lecture notes. Cites every claim.</p>
      </header>

      <main className="chat" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask anything from your lecture notes.</p>
            <div className="examples">
              <button onClick={() => sendQuestion("what is a system call")}>
                what is a system call
              </button>
              <button onClick={() => sendQuestion("compare AMP and SMP")}>
                compare AMP and SMP
              </button>
              <button onClick={() => sendQuestion("what is NoSQL")}>
                what is NoSQL
              </button>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="bubble">
              {msg.text || (msg.pending && <span className="cursor">▍</span>)}
              {msg.pending && msg.text && <span className="cursor">▍</span>}
            </div>
            {msg.role === "assistant" && msg.sources?.length > 0 && (
              <div className="sources">
                {msg.sources.map((s, j) => (
                  <span
                    key={j}
                    className="source-chip"
                    title={`relevance score: ${s.score}`}
                  >
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
          placeholder="Ask something from your notes... (Enter to send, Shift+Enter for newline)"
          rows={2}
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
