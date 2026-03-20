import os
import streamlit as st
import time
import re

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import NewsURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("News Research Tool 📈")
st.sidebar.header("Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        if url.startswith("http"):
            urls.append(url.strip())
        else:
            st.sidebar.warning(f"Invalid URL {i+1}")

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# ---------------------------
# Embedding Model
# ---------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------------------
# Local LLM
# ---------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ---------------------------
# Process URLs
# ---------------------------
if process_url_clicked:
    if len(urls) == 0:
        st.error("Please enter at least one valid URL")
        st.stop()

    main_placeholder.info("Loading URLs...")

    try:
        loader = NewsURLLoader(
            urls=urls,
            requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}}
        )
        data = loader.load()

        cleaned_data = []
        for i, doc in enumerate(data):
            if len(doc.page_content.strip()) > 500:
                doc.metadata["source"] = urls[i] if i < len(urls) else "Unknown"
                cleaned_data.append(doc)

        data = cleaned_data

    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        st.stop()

    if not data:
        st.error("No content found in URLs")
        st.stop()

    st.success(f"Loaded {len(data)} valid documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")

    main_placeholder.success("Processing completed ✅")

# ---------------------------
# Prompts
# ---------------------------
question_prompt = PromptTemplate(
    template="""
Extract only stock target prices from the context.

Context:
{context}

Question:
{question}

Answer (only target prices):
""",
    input_variables=["context", "question"]
)

recommendation_prompt = PromptTemplate(
    template="""
Extract only analyst recommendations from the context.

Context:
{context}

Answer (only Buy, Accumulate, Sell):
""",
    input_variables=["context"]
)

# ---------------------------
# Ask Questions
# ---------------------------
query = st.text_input("Ask a Question about the articles:")

if query:
    if not os.path.exists("faiss_index"):
        st.error("Please process URLs first.")
        st.stop()

    # Load FAISS once
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = st.session_state.vectorstore

    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)
    try:
        # ---------------------------
        # Get raw documents
        # ---------------------------
        docs = retriever.get_relevant_documents(query)
        full_text = " ".join([doc.page_content for doc in docs])

        answer = ""

        # ---------------------------
        # STOCK TARGET EXTRACTION
        # ---------------------------
        # ---------------------------
# STOCK TARGET EXTRACTION
# ---------------------------
        if any(word in query.lower() for word in ["target", "price", "stock", "suggest"]):

            numbers = re.findall(r'(?:Rs\.?|₹)\s*([\d,]{3,6})', full_text)

            cleaned_numbers = []

            for num in numbers:
                num = num.replace(",", "")   # remove commas
                
                if num.isdigit():
                    value = int(num)
                    
                    if 100 <= value <= 5000:
                        cleaned_numbers.append(str(value))

            # ✅ ONLY this (no overwrite!)
            filtered = sorted(set(cleaned_numbers))

            if filtered:
                answer = ", ".join(filtered)
            else:
                answer = "Not available in articles"

        # ---------------------------
        # RECOMMENDATION EXTRACTION
        # ---------------------------
        elif any(word in query.lower() for word in ["recommend", "should", "buy", "sell"]):

            recs = re.findall(r'\b(Buy|Accumulate|Sell)\b', full_text, re.IGNORECASE)

            if recs:
                answer = ", ".join(sorted(set([r.capitalize() for r in recs])))
            else:
                answer = "Not available in articles"

        # ---------------------------
        # DEFAULT LLM ANSWER
        # ---------------------------
        else:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=retriever
            )

            result = chain.invoke({"question": query})
            answer = result.get("answer", "")

        # ---------------------------
        # DISPLAY
        # ---------------------------
        st.header("Answer")

        if answer and "not available" not in answer.lower():
            st.write(answer)
        else:
            st.warning("Answer not found in the selected articles.")

        with st.expander("Debug: Retrieved Chunks"):
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content[:500])

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------
# Summarization
# ---------------------------
if st.button("Summarize Articles"):
    if not os.path.exists("faiss_index"):
        st.error("Please process URLs first.")
        st.stop()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search("summarize", k=5)
    combined_text = " ".join([doc.page_content for doc in docs])

    prompt = f"""
Summarize the following news:

{combined_text}
"""

    with st.spinner("Generating summary..."):
        summary = llm.invoke(prompt)

    st.subheader("Summary")
    st.write(summary)

# ---------------------------
# Chat History
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if query and 'answer' in locals():
    st.session_state.history.append((query, answer))

if st.session_state.history:
    st.subheader("Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")