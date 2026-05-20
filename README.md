# Enterprise AI Copilot

Enterprise AI Copilot is a Microsoft Copilot-style Streamlit application for asking questions over uploaded PDF documents using Retrieval-Augmented Generation (RAG). It extracts text from PDFs, chunks the content, embeds each chunk with Sentence Transformers, stores vectors in a persistent ChromaDB collection, retrieves relevant context for a question, and calls an OpenAI-compatible chat model to generate grounded answers with citations.

## Architecture

```text
User
 |
 | uploads PDFs / asks questions
 v
Streamlit UI (app.py)
 |
 +--> PDF Loader (pypdf)
 |      |
 |      v
 |   Text Splitter (LangChain)
 |      |
 |      v
 |   Embeddings (Sentence Transformers)
 |      |
 |      v
 |   ChromaDB Persistent Vector Store
 |
 +--> RAG Pipeline
        |
        +--> Retrieve top-k chunks from ChromaDB
        |
        +--> Prompt OpenAI-compatible LLM
        |
        v
      Answer + source citations
```

## Project Structure

```text
enterprise-ai-copilot/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
├── chroma_db/
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── prompts.py
└── tests/
    └── test_pdf_loader.py
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your local environment file:

```bash
copy .env.example .env
```

4. Update `.env` with your OpenAI-compatible API settings:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

## Running the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, upload one or more PDF files, click **Process Documents**, then ask questions about the uploaded content.

## Running Tests

```bash
pytest
```

## Configuration

The app is configured through environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | API key for the OpenAI-compatible LLM provider | empty |
| `OPENAI_BASE_URL` | Base URL for the chat completions API | `https://api.openai.com/v1` |
| `LLM_MODEL` | Chat model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL_NAME` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Character chunk size | `1000` |
| `CHUNK_OVERLAP` | Character overlap between chunks | `200` |
| `RETRIEVAL_TOP_K` | Number of source chunks to retrieve | `4` |
| `CHROMA_PERSIST_DIR` | Local ChromaDB persistence directory | `chroma_db` |

## Example Screenshots

```text
[Screenshot placeholder: PDF upload and processing sidebar]
[Screenshot placeholder: question input and grounded answer]
[Screenshot placeholder: source citation expanders]
```

## Future Enhancements

- FastAPI backend for API-first deployments
- Authentication and role-based access control
- Multi-agent workflows for document review and summarization
- Azure OpenAI integration with managed identity support
- Deployment with Docker and Kubernetes
- CI/CD with GitHub Actions
- Document-level access controls and audit logging
- Incremental indexing and duplicate document detection


## ========================== Overview ===============================

Absolutely. Think of this app as having **two separate flows**.

1. **Upload and process documents**
2. **Ask a question and get an answer**

The user cannot get a useful answer until the documents are processed first.

**Flow 1: User Uploads PDFs**
The user starts in [app.py](C:/codeBase/AI_Project_1/enterprise-ai-copilot/app.py).

They upload PDFs here:

```python
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)
```

Then they click:

```python
process_clicked = st.button("Process Documents")
```

When that button is clicked, this runs:

```python
inserted_chunks = ingest_documents(list(uploaded_files))
```

That function lives in:

[rag_pipeline.py](C:/codeBase/AI_Project_1/enterprise-ai-copilot/src/rag_pipeline.py)

**What happens inside `ingest_documents()`**
The app does this:

```text
Uploaded PDF files
        |
        v
load_pdfs(files)
        |
        v
Extract text page by page
        |
        v
split_documents(documents)
        |
        v
Break text into smaller chunks
        |
        v
add_document_chunks(chunks)
        |
        v
Create embeddings and save in ChromaDB
```

So the full upload flow is:

```text
User uploads PDF
→ app.py receives file
→ rag_pipeline.ingest_documents()
→ pdf_loader reads text
→ text_splitter creates chunks
→ embeddings model converts chunks into vectors
→ vector_store saves them in ChromaDB
→ app shows "Processed X document chunks"
```

At this point, the documents are now searchable.

**Flow 2: User Asks A Question**
Again, the user starts in [app.py](C:/codeBase/AI_Project_1/enterprise-ai-copilot/app.py).

They type a question here:

```python
question = st.text_input("Question")
```

Then they click:

```python
ask_clicked = st.button("Ask Copilot")
```

When clicked, this runs:

```python
result = answer_question(question)
```

That function lives in:

[rag_pipeline.py](C:/codeBase/AI_Project_1/enterprise-ai-copilot/src/rag_pipeline.py)

**What happens inside `answer_question()`**
The app does this:

```text
User question
     |
     v
Search ChromaDB for similar chunks
     |
     v
Get top relevant chunks
     |
     v
Put chunks into prompt
     |
     v
Send prompt to LLM
     |
     v
Receive answer
     |
     v
Return answer + sources
```

So the full question flow is:

```text
User asks question
→ app.py sends question to rag_pipeline.answer_question()
→ vector_store searches ChromaDB
→ relevant PDF chunks are retrieved
→ prompt is built using prompts.py
→ OpenAI-compatible LLM generates answer
→ app.py displays answer
→ app.py displays source chunks/citations
```

**Simple Example**
Imagine the user uploads a PDF contract.

The PDF contains:

```text
Payment must be completed within 30 days of invoice receipt.
```

During processing:

```text
That sentence becomes a chunk
→ chunk becomes an embedding
→ embedding is stored in ChromaDB
```

Then the user asks:

```text
What is the payment deadline?
```

The app does not search exact words only. It searches meaning.

So it finds:

```text
Payment must be completed within 30 days of invoice receipt.
```

Then the LLM receives:

```text
Context:
Payment must be completed within 30 days of invoice receipt.

Question:
What is the payment deadline?
```

The LLM answers:

```text
The payment deadline is within 30 days of invoice receipt.
```

And the app also shows the source:

```text
Source: contract.pdf, page 2, chunk 1
```

**Who Sends What?**
The user sends two kinds of requests:

```text
1. Upload PDF request
2. Ask question request
```

For upload:

```text
User sends: PDF files
User gets: "Processed X chunks"
```

For question:

```text
User sends: text question
User gets: answer + source citations
```

**Most Important Thing To Remember**
The LLM does not magically know your PDFs.

The app first creates a searchable memory from PDFs:

```text
PDF → ChromaDB
```

Then questions use that memory:

```text
Question → ChromaDB search → relevant context → LLM answer
```

That is the whole RAG flow.
