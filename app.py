"""
app.py — Sprint 3
Streamlit chat interface for the Retail RAG Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
from rag_pipeline import load_rag_chain, ask

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailBot AI",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        margin: 4px 0 0;
        opacity: 0.75;
        font-size: 0.9rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 4px;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a73e8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 3px;
        border: 1px solid #c2dff7;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .sidebar-section h4 {
        margin: 0 0 8px 0;
        font-size: 0.85rem;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Example query chips */
    .example-query {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #333;
        margin: 4px 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    llm_type = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai", "groq"],
        help="Choose your LLM backend",
    )

    model_options = {
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
        "openai": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
    }

    model_name = st.selectbox("Model", options=model_options[llm_type])

    if llm_type in ["openai", "groq"]:
        api_key_label = "OpenAI API Key" if llm_type == "openai" else "Groq API Key"
        api_key = st.text_input(api_key_label, type="password")
        if api_key:
            import os
            env_key = "OPENAI_API_KEY" if llm_type == "openai" else "GROQ_API_KEY"
            os.environ[env_key] = api_key

    st.divider()

    st.markdown("### 💡 Example Queries")
    example_queries = [
        "Does SmartPhone X1 support 5G?",
        "Compare SmartPhone X1 and X2",
        "Which laptops are available?",
        "What is the return policy for laptops?",
        "Show me products under ₹30,000",
        "What warranty does LaptopPro Z7 have?",
        "Tell me about noise cancelling earbuds",
        "Is the RoboVac in stock?",
    ]
    for q in example_queries:
        st.markdown(f"<div class='example-query'>💬 {q}</div>", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 📊 Session Stats")
    msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.metric("Questions Asked", msg_count)


# ── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛍️ RetailBot AI</h1>
    <p>Powered by RAG — Ask anything about our products</p>
</div>
""", unsafe_allow_html=True)

# ── Load RAG Chain (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_chain(llm_type, model_name):
    return load_rag_chain(llm_type=llm_type, model_name=model_name)


# Load chain with spinner
chain_key = f"{llm_type}_{model_name}"
if "chain_key" not in st.session_state or st.session_state.chain_key != chain_key:
    with st.spinner("🔄 Loading AI model and knowledge base..."):
        try:
            st.session_state.chain = get_chain(llm_type, model_name)
            st.session_state.chain_key = chain_key
            st.success("✅ Knowledge base loaded! Ready to answer your questions.")
        except FileNotFoundError:
            st.error("❌ Vector store not found. Please run `python ingest.py` first!")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.info("💡 If using Ollama, make sure it's running: `ollama serve` then `ollama pull llama3`")
            st.stop()

# ── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hi! I'm your RetailBot assistant. I can help you with product information, pricing, comparisons, warranties, and availability. What would you like to know?",
        "sources": [],
    })

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🛍️" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(
                "**📌 Based on:** " +
                "".join([f"<span class='source-badge'>{s}</span>" for s in msg["sources"]]),
                unsafe_allow_html=True,
            )

# ── User Input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about any product... (e.g. 'Compare X1 and X2')"):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("🔍 Searching product knowledge base..."):
            result = ask(st.session_state.chain, user_input)

        st.write(result["answer"])

        if result["sources"]:
            st.markdown(
                "**📌 Based on:** " +
                "".join([f"<span class='source-badge'>{s}</span>" for s in result["sources"]]),
                unsafe_allow_html=True,
            )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
