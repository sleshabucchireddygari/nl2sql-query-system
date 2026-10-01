"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import os

import streamlit as st

from nl2sql import NL2SQLEngine, Settings
from nl2sql.providers import PROVIDERS, required_key

st.set_page_config(page_title="Natural Language to SQL", page_icon="🗄️", layout="wide")

EXAMPLES = [
    "What is the average length of stay for each department?",
    "What are the top 5 most prescribed drugs?",
    "Which department had the highest mortality rate?",
    "What is the 30-day readmission rate by admission type?",
    "How many distinct patients were prescribed Carboplatin?",
]
PROVIDER_LABELS = {
    "gemini": "Google Gemini (free)",
    "ollama": "Ollama – local model (free)",
    "openai": "OpenAI (paid)",
}
KEY_HELP = {
    "GOOGLE_API_KEY": "Free key, no credit card: https://aistudio.google.com/apikey",
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys (needs paid credit)",
}


@st.cache_resource(show_spinner="Indexing database schema…")
def load_engine(db_url: str, provider: str, model: str, max_retries: int,
                use_retrieval: bool, _key: str) -> NL2SQLEngine:
    s = Settings()
    s.database_url, s.llm_provider, s.max_retries = db_url, provider, max_retries
    setattr(s, f"{provider}_model", model)
    return NL2SQLEngine.from_settings(s, use_table_retrieval=use_retrieval)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "quota" in low or "rate" in low and "limit" in low:
        return ("The AI provider is rate-limiting or out of quota. On the free Gemini tier, wait "
                "a minute and try again (limit is about 10 requests/minute).")
    if "api key" in low or "api_key" in low or "401" in msg or "403" in msg or "permission" in low:
        return "The API key was rejected. Check that the key is correct and saved in .env (or the sidebar)."
    if "connect" in low and "11434" in msg:
        return "Can't reach Ollama. Install it from ollama.com and make sure it is running."
    if "404" in msg or "not found" in low:
        return "Model not found. Check the model name in the sidebar."
    return msg.splitlines()[0][:400]


# ------------------------------------------------------------------ sidebar
settings = Settings()
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "AI provider", PROVIDERS,
        index=PROVIDERS.index(settings.llm_provider) if settings.llm_provider in PROVIDERS else 0,
        format_func=PROVIDER_LABELS.get)
    key_name = required_key(provider)
    if key_name and not os.getenv(key_name):
        key = st.text_input(key_name, type="password", help=KEY_HELP[key_name])
        if key:
            os.environ[key_name] = key.strip()
    model = st.text_input("Model", getattr(settings, f"{provider}_model"))
    db_url = st.text_input("Database URL", settings.database_url,
                           help="SQLAlchemy URL, e.g. mysql+pymysql://user:pw@localhost:3306/hospital")
    max_retries = st.slider("Self-correction retries", 0, 4, settings.max_retries)
    use_retrieval = st.toggle("Schema retrieval (LlamaIndex)", value=True)
    summarize = st.toggle("Plain-English answer", value=True,
                          help="Uses one extra AI request per question.")

st.title("🗄️ Natural Language to SQL")
st.caption("Ask a question about the hospital database. The system grounds the LLM in the real "
           "schema with LlamaIndex, validates the generated SQL, and self-corrects before running it.")

if key_name and not os.getenv(key_name):
    st.info(f"Add your {key_name} in the sidebar or in a `.env` file to start. "
            f"{KEY_HELP[key_name]}")
    st.stop()

try:
    engine = load_engine(db_url, provider, model, max_retries, use_retrieval,
                         os.getenv(key_name or "", ""))
except Exception as e:  # noqa: BLE001
    st.error(friendly_error(e))
    st.stop()

if engine.retrieval_warning:
    st.warning(engine.retrieval_warning)

with st.expander("Database schema"):
    st.code(engine.catalog.describe(), language="sql")

cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, width="stretch"):
        st.session_state["question"] = ex

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Which doctors have the longest average length of stay?")

if question:
    try:
        with st.spinner("Generating SQL…"):
            result = engine.query(question, summarize=summarize)
    except Exception as e:  # noqa: BLE001
        st.error(friendly_error(e))
        st.stop()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Generated SQL")
        st.code(result.sql or "-- none", language="sql")
    with right:
        st.subheader("Run details")
        st.metric("Status", "✅ Valid" if result.success else "❌ Failed")
        st.write(f"**Tables used:** {', '.join(result.tables_used)}")
        st.write(f"**Self-corrections:** {result.retries}  ·  **Time:** {result.seconds}s")

    if result.retries:
        with st.expander(f"Validation history ({len(result.attempts)} attempts)"):
            for i, a in enumerate(result.attempts, 1):
                st.markdown(f"**Attempt {i}**")
                st.code(a.sql, language="sql")
                for err in a.errors:
                    st.error(err)

    if result.success:
        if result.answer:
            st.success(result.answer)
        st.dataframe(result.data, width="stretch")
        if result.truncated:
            st.caption(f"Showing the first {engine.row_limit} rows.")
        num = result.data.select_dtypes("number").columns
        if len(result.data) > 1 and len(result.data.columns) == 2 and len(num) == 1:
            st.bar_chart(result.data.set_index(result.data.columns[0]))
        st.download_button("Download CSV", result.data.to_csv(index=False), "result.csv", "text/csv")
    else:
        st.error(result.error)
