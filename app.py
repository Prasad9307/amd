import os
import io
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader


# =========================
# Embeddings + Vector "DB"
# =========================

def embed_text(text: str, client: genai.Client) -> np.ndarray:
    """
    Return embedding for a single text using Gemini embeddings.
    Uses `text-embedding-004` via google-genai SDK.
    """
    if not text.strip():
        # 768 dims for text-embedding-004
        return np.zeros(768, dtype=np.float32)

    # We send as a list to get consistent response shape
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=[text],
    )
    # First (and only) embedding
    emb = result.embeddings[0].values
    return np.array(emb, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def add_document_to_vectordb(filename: str, text: str, client: genai.Client):
    """
    Split text into chunks, embed and store in session 'vectordb'.
    In-memory "vector DB".
    """
    CHUNK_SIZE = 800  # characters
    OVERLAP = 200

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP

    for chunk in chunks:
        if not chunk.strip():
            continue
        emb = embed_text(chunk, client)
        st.session_state["vectordb"].append(
            {
                "filename": filename,
                "text": chunk,
                "embedding": emb,
            }
        )


def retrieve_relevant_chunks(
    query: str, client: genai.Client, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Return top_k chunks from vectordb most similar to query."""
    if not st.session_state["vectordb"]:
        return []

    q_emb = embed_text(query, client)
    scored = []
    for item in st.session_state["vectordb"]:
        score = cosine_similarity(q_emb, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k] if s[0] > 0]


# =========================
# Tools
# =========================

def tool_calculator(user_input: str) -> str:
    """
    Simple calculator tool.

    Examples:
    - "calc 2+2*5"
    - "calculate 100/4"
    """
    expr = (
        user_input.lower()
        .replace("calculate", "")
        .replace("calc", "")
        .strip()
    )
    if not expr:
        return "Please provide an expression, e.g. `calc 2+3*5`."

    # Safe evaluation: only digits and + - * / . ( )
    allowed_chars = set("0123456789+-*/.() ")
    if not set(expr).issubset(allowed_chars):
        return "Only basic arithmetic (+, -, *, /) is allowed."

    try:
        result = eval(expr, {"__builtins__": {}})
        return f"🧮 The result of `{expr}` is **{result}**."
    except Exception as e:
        return f"Sorry, I could not evaluate that expression. Error: {e}"


def tool_todo(user_input: str) -> str:
    """
    Simple in-memory todo list.

    Commands:
    - "todo add Buy milk"
    - "todo list"
    - "todo clear"
    """
    if "todo_list" not in st.session_state:
        st.session_state["todo_list"] = []

    lower = user_input.lower()

    if "todo add" in lower:
        task = user_input.split("todo add", 1)[1].strip()
        if not task:
            return "Please provide a task after `todo add`."
        st.session_state["todo_list"].append(task)
        return f"✅ Added to todo: **{task}**"

    if "todo clear" in lower:
        st.session_state["todo_list"] = []
        return "🧹 Cleared all todo items."

    # default: list
    if not st.session_state["todo_list"]:
        return "Your todo list is empty. Add something with `todo add <task>`."

    lines = [f"{i+1}. {t}" for i, t in enumerate(st.session_state["todo_list"])]
    return "📝 **Your todo list:**\n\n" + "\n".join(lines)


def tool_translate(user_input: str, client: genai.Client) -> str:
    """
    Translation tool using Gemini:

    Examples:
    - "translate to hindi: How are you?"
    - "translate to marathi: Good morning"
    """
    lower = user_input.lower()
    if "translate to" not in lower:
        return "Use format: `translate to <language>: <text>`."

    try:
        after = lower.split("translate to", 1)[1].strip()
        if ":" not in after:
            return "Use format: `translate to <language>: <text>`."

        lang_part, _ = after.split(":", 1)
        target_lang = lang_part.strip()

        # Preserve original case for actual text
        text_to_translate = user_input.split(":", 1)[1].strip()

        prompt = (
            f"Translate the following text to {target_lang}. "
            "Just give the translation, nothing else.\n\n"
            f"Text: {text_to_translate}"
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return f"🌐 **Translation ({target_lang}):**\n\n{resp.text}"
    except Exception as e:
        return f"Sorry, I could not translate. Error: {e}"


# =========================
# Agent Routing Logic
# =========================

def choose_route(user_input: str) -> str:
    """
    Decide which route to use:
    - 'calculator'
    - 'todo'
    - 'translator'
    - 'rag' (documents)
    - 'chat' (plain Gemini chat)
    """
    msg = user_input.lower()

    if msg.startswith("calc") or "calculate" in msg:
        return "calculator"

    if msg.startswith("todo"):
        return "todo"

    if msg.startswith("translate to"):
        return "translator"

    # If docs exist, default to RAG for general questions
    if st.session_state["vectordb"]:
        return "rag"

    return "chat"


def run_agent(user_input: str, client: genai.Client) -> str:
    """
    Main agent: chooses tool / RAG / chat based on user input.
    """
    route = choose_route(user_input)

    if route == "calculator":
        return tool_calculator(user_input)

    if route == "todo":
        return tool_todo(user_input)

    if route == "translator":
        return tool_translate(user_input, client)

    if route == "rag":
        context_chunks = retrieve_relevant_chunks(user_input, client, top_k=5)
        if not context_chunks:
            route = "chat"  # fallback to normal chat
        else:
            context_text = "\n\n---\n\n".join(
                f"[{c['filename']}]\n{c['text']}" for c in context_chunks
            )
            prompt = (
                "You are a helpful AI assistant. Use ONLY the context below to "
                "answer the user's question. If the answer is not in the context, "
                "say you don't know.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"USER QUESTION:\n{user_input}"
            )
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return resp.text

    # default chat route
    prompt = (
        "You are a friendly AI assistant. Answer the user clearly and concisely.\n\n"
        f"User: {user_input}"
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return resp.text


# =========================
# File Reading
# =========================

def read_file(file) -> str:
    """Return plain text from uploaded file (PDF or TXT)."""
    filename = file.name.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

    return ""


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="Smart AI Agent – RAG + Tools",
        page_icon="🤖",
        layout="wide",
    )

    # --- Custom top banner (simple but attractive) ---
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            padding: 0.5rem 0;
        }
        .subtitle {
            font-size: 16px;
            color: #5f6c80;
        }
        .feature-box {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            border: 1px solid #e4e6eb;
            background: linear-gradient(135deg, #f9fbff, #fdfbff);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "vectordb" not in st.session_state:
        st.session_state["vectordb"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar: settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        default_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=default_key,
            help="You can also set GEMINI_API_KEY env variable.",
        )

        if api_key:
            st.success("API key entered.")
        else:
            st.warning("Please provide your Gemini API key.")

        st.markdown("---")
        st.markdown("### 📄 Uploaded Docs")
        if st.session_state["vectordb"]:
            filenames = sorted({d["filename"] for d in st.session_state["vectordb"]})
            for fn in filenames:
                st.write(f"• {fn}")
        else:
            st.write("No documents uploaded yet.")

        st.markdown("---")
        st.markdown("### 🛠 Tools Guide")
        st.caption("• `calc 2+3*5` or `calculate 100/4`\n"
                   "• `todo add Finish report`\n"
                   "• `todo list`, `todo clear`\n"
                   "• `translate to hindi: How are you?`")

    # Top header
    st.markdown('<div class="main-title">🤖 Smart AI Agent – RAG + Tools</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions from your documents, use tools like calculator, todo & translator – all in one clean interface.</div>',
        unsafe_allow_html=True,
    )

    # Feature row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-box">📚 <b>RAG</b><br><span style="font-size:13px;">Upload PDFs/TXT and ask questions based on content.</span></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="feature-box">🧰 <b>Tools</b><br><span style="font-size:13px;">Calculator, Todo manager, and Translator built-in.</span></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="feature-box">⚡ <b>Gemini API</b><br><span style="font-size:13px;">Powered by Google Gemini for generation & embeddings.</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to start.")
        st.stop()

    # Create Gemini client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to create Gemini client: {e}")
        st.stop()

    # --- Document upload area ---
    st.markdown("### 📁 Upload Documents (PDF or TXT)")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            for f in uploaded_files:
                text = read_file(f)
                if not text.strip():
                    st.warning(f"Could not read text from {f.name}.")
                    continue
                try:
                    add_document_to_vectordb(f.name, text, client)
                except Exception as e:
                    st.error(f"Error embedding {f.name}: {e}")
            st.success("✅ Documents added to in-memory vector store.")

    st.markdown("### 💬 Chat with Your AI Agent")

    # Show chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question or use tools (calc, todo, translate)...")
    if user_input:
        # User message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent(user_input, client)
                except Exception as e:
                    answer = f"❌ Error while generating response: {e}"
                st.markdown(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()
