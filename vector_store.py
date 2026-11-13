import re

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==================================================
# CONFIGURATION
# ==================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DB_PATH = "./chroma_db"
COLLECTION_NAME = "docuagent_current"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150


# ==================================================
# MODEL LOADING
# ==================================================


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# ==================================================
# CHROMA DATABASE
# ==================================================


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


# ==================================================
# DOCUMENT FILTER
# ==================================================


def make_document_filter(document_ids):

    if not document_ids:
        return None

    document_ids = list(dict.fromkeys(document_ids))

    if len(document_ids) == 1:
        return {"document_id": document_ids[0]}

    return {"document_id": {"$in": document_ids}}


# ==================================================
# TEXT CHUNKING
# ==================================================


def split_long_text(text, chunk_size, overlap):
    """
    Split a long paragraph into overlapping chunks.
    Prefer splitting at whitespace instead of cutting words.
    """

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):

            # Try to find a whitespace boundary.
            boundary = text.rfind(" ", start, end)

            if boundary > start + int(chunk_size * 0.7):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def split_sections(
    sections,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP,
):
    """
    Split extracted document sections into chunks.

    Keeps the original page number where available.
    Attempts to preserve paragraph boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size - 1.")

    chunks = []

    for section in sections:

        text = section.get("text", "")

        if not text or not text.strip():
            continue

        page = section.get("page")
        paragraphs = re.split(r"\n\s*\n", text.strip())

        current_text = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) > chunk_size:

                if current_text:
                    chunks.append(
                        {
                            "text": current_text.strip(),
                            "page": page,
                        }
                    )
                    current_text = ""

                for part in split_long_text(
                    paragraph,
                    chunk_size,
                    overlap,
                ):
                    chunks.append(
                        {
                            "text": part,
                            "page": page,
                        }
                    )

                continue

            if not current_text:
                current_text = paragraph

            elif len(current_text) + len(paragraph) + 2 <= chunk_size:
                current_text += "\n\n" + paragraph

            else:

                chunks.append(
                    {
                        "text": current_text.strip(),
                        "page": page,
                    }
                )

                # Retain a little context from the previous chunk.
                previous_words = current_text.split()
                overlap_words = []

                remaining = overlap

                for word in reversed(previous_words):

                    if remaining <= 0:
                        break

                    overlap_words.insert(0, word)
                    remaining -= len(word) + 1

                overlap_text = " ".join(overlap_words)

                current_text = (
                    overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                )

        if current_text.strip():
            chunks.append(
                {
                    "text": current_text.strip(),
                    "page": page,
                }
            )

    return chunks


# ==================================================
# BUILD DOCUMENT INDEX
# ==================================================


def build_index(sections, document_id, filename):
    """
    Generate embeddings and store document chunks.

    Existing chunks for the same document ID are replaced.
    """

    if not sections:
        raise ValueError("No document sections were provided.")

    if not document_id:
        raise ValueError("Document ID is required.")

    chunks = split_sections(sections)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    collection = get_collection()
    embedding_model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        ids.append(f"{document_id}_chunk_{index}")

        metadatas.append(
            {
                "document_id": document_id,
                "filename": filename,
                "page": int(chunk.get("page") or 0),
                "chunk_index": index,
            }
        )

    # Replace previous chunks for this document.
    collection.delete(where={"document_id": document_id})

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


# ==================================================
# CROSS-ENCODER RERANKING
# ==================================================


def rerank_chunks(question, chunks, top_k=4):
    """
    Rank retrieved passages by relevance to the question.

    If the reranker cannot load, retain the original
    semantic retrieval order.
    """

    if not chunks:
        return []

    try:

        reranker = get_reranker()

        pairs = [[question, chunk["text"]] for chunk in chunks]

        scores = reranker.predict(pairs)

        ranked_chunks = []

        for chunk, score in zip(chunks, scores):

            item = dict(chunk)
            item["rerank_score"] = float(score)

            ranked_chunks.append(item)

        ranked_chunks.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_chunks[:top_k]

    except Exception as error:

        print(f"Reranking unavailable: {error}")

        return chunks[:top_k]


# ==================================================
# SEMANTIC DOCUMENT RETRIEVAL
# ==================================================


def retrieve_chunks(
    question,
    top_k=4,
    document_ids=None,
    candidate_k=None,
):
    """
    Retrieve relevant passages using:

    1. Question embeddings.
    2. ChromaDB semantic similarity search.
    3. Cross-encoder reranking.

    Optionally restricts retrieval to selected documents.
    """

    if not question or not question.strip():
        return []

    if top_k <= 0:
        return []

    collection = get_collection()

    if collection.count() == 0:
        return []

    where_filter = make_document_filter(document_ids)

    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    if candidate_k is None:
        candidate_k = max(20, top_k * 5)

    candidate_count = min(
        candidate_k,
        collection.count(),
    )

    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": candidate_count,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter:
        query_args["where"] = where_filter

    results = collection.query(**query_args)

    result_ids = (results.get("ids") or [[]])[0]
    result_documents = (results.get("documents") or [[]])[0]
    result_metadatas = (results.get("metadatas") or [[]])[0]
    result_distances = (results.get("distances") or [[]])[0]

    candidates = []

    for chunk_id, document, metadata, distance in zip(
        result_ids,
        result_documents,
        result_metadatas,
        result_distances,
    ):

        metadata = metadata or {}

        candidates.append(
            {
                "id": chunk_id,
                "text": document,
                "page": metadata.get("page", 0),
                "filename": metadata.get("filename", "Document"),
                "document_id": metadata.get("document_id", ""),
                "distance": float(distance),
            }
        )

    if not candidates:
        return []

    return rerank_chunks(
        question=question,
        chunks=candidates,
        top_k=top_k,
    )


# ==================================================
# GET ALL DOCUMENT CHUNKS
# ==================================================


def get_all_chunks(document_ids=None):
    """
    Return all stored chunks.

    Can optionally filter by document IDs.
    """

    collection = get_collection()

    if collection.count() == 0:
        return []

    where_filter = make_document_filter(document_ids)

    get_args = {
        "include": [
            "documents",
            "metadatas",
        ],
    }

    if where_filter:
        get_args["where"] = where_filter

    results = collection.get(**get_args)

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    chunks = []

    for chunk_id, document, metadata in zip(
        ids,
        documents,
        metadatas,
    ):

        metadata = metadata or {}

        chunks.append(
            {
                "id": chunk_id,
                "text": document,
                "page": metadata.get("page", 0),
                "filename": metadata.get("filename", "Document"),
                "document_id": metadata.get("document_id", ""),
            }
        )

    return chunks
