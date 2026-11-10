"""Streamlit front end for the Multimodal AI Intelligence Platform."""
import streamlit as st
from src.ingestion import extract_file
from src.retrieval import LocalRetriever
from src import config

st.set_page_config(page_title="Multimodal AI Intelligence", page_icon="🧠", layout="wide")
st.title("🧠 Multimodal AI Intelligence Platform")
st.caption("Upload documents, images, audio, and video. Search across their extracted content and ask grounded questions.")

if "documents" not in st.session_state:
    st.session_state.documents = []
if "retriever" not in st.session_state:
    st.session_state.retriever = LocalRetriever()

with st.sidebar:
    st.header("Workspace")
    st.write(f"OpenAI integrations: {'Enabled' if config.OPENAI_API_KEY else 'Not configured'}")
    st.caption("Add OPENAI_API_KEY to .env for audio transcription, image descriptions, and LLM answers.")
    if st.button("Clear workspace", use_container_width=True):
        st.session_state.documents = []
        st.session_state.retriever = LocalRetriever()
        st.rerun()

uploaded = st.file_uploader(
    "Add files to your workspace",
    type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg", "webp", "bmp",
          "wav", "mp3", "m4a", "mp4", "mov", "mkv", "avi"],
    accept_multiple_files=True,
)
if uploaded and st.button("Process uploaded files", type="primary"):
    added = []
    progress = st.progress(0)
    for index, file in enumerate(uploaded):
        try:
            text, method = extract_file(file.name, file.getvalue())
            if text.strip():
                added.append((file.name, text))
                st.success(f"{file.name}: processed with {method}")
            else:
                st.warning(f"{file.name}: no text extracted.")
        except Exception as exc:
            st.error(f"{file.name}: {exc}")
        progress.progress((index + 1) / len(uploaded))
    if added:
        st.session_state.documents.extend(added)
        retriever = LocalRetriever()
        retriever.add_documents(st.session_state.documents)
        st.session_state.retriever = retriever
        st.success(f"Added {len(added)} file(s) to the workspace.")

st.subheader("Workspace files")
if st.session_state.documents:
    for name, text in st.session_state.documents:
        with st.expander(f"📄 {name} · {len(text):,} characters"):
            st.text(text[:5000] + ("…" if len(text) > 5000 else ""))
else:
    st.info("No files yet. Upload one or more supported files above.")

st.divider()
st.subheader("Ask your workspace")
question = st.text_input("Question", placeholder="What are the main findings across these files?")
top_k = st.slider("Number of context passages", min_value=1, max_value=10, value=5)
if st.button("Search and answer", disabled=not bool(st.session_state.documents) or not question.strip()):
    results = st.session_state.retriever.search(question, top_k=top_k)
    if not results:
        st.warning("No matching text found. Try different keywords or check whether extraction succeeded.")
    else:
        if config.OPENAI_API_KEY:
            try:
                from src.ai_services import answer_question
                answer = answer_question(question, results)
            except Exception as exc:
                st.warning(f"AI answer unavailable ({exc}). Showing retrieved evidence instead.")
                answer = "\n\n".join(f"**{r['source']}**\n{r['text']}" for r in results)
        else:
            answer = "No OpenAI key configured. Here are the most relevant passages:"
            answer += "\n\n" + "\n\n".join(f"**{r['source']}**\n{r['text']}" for r in results)
        st.markdown(answer)
        with st.expander("Retrieved evidence and relevance scores"):
            for item in results:
                st.markdown(f"**{item['source']}** · relevance {item['score']:.3f}")
                st.write(item["text"])
