import hashlib
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
from docx import Document

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)

from document_processor import extract_document
from vector_store import build_index, get_all_chunks
from agent import ask_document

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="DocuAgent | AI Workspace",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "chat_history": [],
    "documents": {},
    "chat_document_ids": None,
    "selected_documents": [],
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# DOCUMENT LIBRARY RESTORATION
# ==================================================


def restore_document_library():
    """
    Reconstruct the document library from persistent
    ChromaDB records.

    Original extracted sections are not stored in ChromaDB.
    Restored previews therefore use indexed chunks.
    """

    chunks = get_all_chunks()

    documents = {}

    for chunk in chunks:

        document_id = chunk.get("document_id")

        if not document_id:
            continue

        filename = chunk.get("filename", "Document")

        if document_id not in documents:

            documents[document_id] = {
                "filename": filename,
                "sections": [],
                "chunk_count": 0,
                "restored": True,
            }

        documents[document_id]["sections"].append(
            {
                "page": chunk.get("page"),
                "text": chunk.get("text", ""),
            }
        )

        documents[document_id]["chunk_count"] += 1

    # Preserve full extracted sections for documents
    # already uploaded during the current session.
    for document_id, current_document in st.session_state["documents"].items():

        if document_id in documents:

            if not current_document.get("restored", False):

                documents[document_id] = current_document

    st.session_state["documents"] = documents


try:

    restore_document_library()

except Exception as error:

    st.error(f"Could not restore the document library: {error}")


# ==================================================
# REPORT EXPORT FUNCTIONS
# ==================================================


def get_export_conversations(chat_history):

    conversations = []
    current_question = None

    for message in chat_history:

        if message.get("role") == "user":

            current_question = message.get("content", "")

        elif message.get("role") == "assistant":

            conversations.append(
                {
                    "question": current_question or "(Question not recorded)",
                    "answer": message.get("content", ""),
                    "sources": message.get("sources", []),
                }
            )

            current_question = None

    return conversations


def build_pdf_report(chat_history, selected_filenames):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="DocuAgent Conversation Report",
        author="DocuAgent",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "QuestionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=8,
    )

    source_style = ParagraphStyle(
        "SourceText",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        leftIndent=10,
        spaceAfter=8,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    story.append(Paragraph("DocuAgent", title_style))

    story.append(
        Paragraph(
            "AI Document Analysis Report",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            body_style,
        )
    )

    story.append(Paragraph("Selected Documents", styles["Heading2"]))

    if selected_filenames:

        for filename in selected_filenames:

            story.append(
                Paragraph(
                    f"• {escape(filename)}",
                    body_style,
                )
            )

    else:

        story.append(
            Paragraph(
                "No document names recorded.",
                body_style,
            )
        )

    conversations = get_export_conversations(chat_history)

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Questions and Answers",
            styles["Heading1"],
        )
    )

    if not conversations:

        story.append(
            Paragraph(
                "No completed conversations to export.",
                body_style,
            )
        )

    for index, conversation in enumerate(conversations, start=1):

        story.append(
            Paragraph(
                f"Question {index}: " f"{escape(conversation['question'])}",
                heading_style,
            )
        )

        answer_text = conversation["answer"].replace(
            "₹",
            "INR ",
        )

        story.append(
            Paragraph(
                escape(answer_text).replace("\n", "<br/>"),
                body_style,
            )
        )

        sources = conversation["sources"]

        if sources:

            story.append(
                Paragraph(
                    "Supporting Sources",
                    styles["Heading3"],
                )
            )

            for source_index, source in enumerate(sources, start=1):

                source_id = source.get(
                    "id",
                    f"S{source_index}",
                )

                label = source.get(
                    "filename",
                    source.get("label", "Document"),
                )

                page = source.get("page", 0)

                source_text = source.get(
                    "text",
                    "",
                ).replace("₹", "INR ")

                location = f" | Page {page}" if page else ""

                story.append(
                    Paragraph(
                        f"<b>[{escape(str(source_id))}] "
                        f"{escape(str(label))}{location}</b>",
                        source_style,
                    )
                )

                if source_text:

                    story.append(
                        Paragraph(
                            escape(source_text).replace(
                                "\n",
                                "<br/>",
                            ),
                            source_style,
                        )
                    )

        else:

            story.append(
                Paragraph(
                    "No source passages were recorded.",
                    source_style,
                )
            )

        story.append(Spacer(1, 10))

    def add_page_number(canvas, doc):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)

        canvas.drawString(
            20 * mm,
            10 * mm,
            "DocuAgent | AI Document Analysis",
        )

        canvas.drawRightString(
            A4[0] - 20 * mm,
            10 * mm,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    pdf.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)

    return buffer.getvalue()


def build_docx_report(chat_history, selected_filenames):

    document = Document()

    document.add_heading("DocuAgent", level=0)

    document.add_paragraph("AI Document Analysis Report")

    document.add_paragraph(
        "Generated: " + datetime.now().strftime("%d %b %Y, %I:%M %p")
    )

    document.add_heading(
        "Selected Documents",
        level=1,
    )

    if selected_filenames:

        for filename in selected_filenames:

            document.add_paragraph(
                filename,
                style="List Bullet",
            )

    else:

        document.add_paragraph("No document names recorded.")

    document.add_heading(
        "Questions and Answers",
        level=1,
    )

    conversations = get_export_conversations(chat_history)

    if not conversations:

        document.add_paragraph("No completed conversations to export.")

    for index, conversation in enumerate(conversations, start=1):

        document.add_heading(
            f"Question {index}: {conversation['question']}",
            level=2,
        )

        document.add_paragraph(conversation["answer"])

        sources = conversation["sources"]

        if sources:

            document.add_heading(
                "Supporting Sources",
                level=3,
            )

            for source_index, source in enumerate(sources, start=1):

                source_id = source.get(
                    "id",
                    f"S{source_index}",
                )

                filename = source.get(
                    "filename",
                    source.get("label", "Document"),
                )

                page = source.get("page", 0)

                location = f" | Page {page}" if page else ""

                document.add_paragraph(
                    f"[{source_id}] {filename}{location}",
                    style="List Bullet",
                )

                source_text = source.get("text", "")

                if source_text:
                    document.add_paragraph(source_text)

        else:

            document.add_paragraph("No source passages were recorded.")

    buffer = BytesIO()

    document.save(buffer)

    buffer.seek(0)

    return buffer.getvalue()


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>

    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stHeader"] {
        background: var(--background-color);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--secondary-background-color);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero-card {
        padding: 26px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: #ffffff !important;
        margin-bottom: 24px;
    }

    .hero-card h1 {
        color: #ffffff !important;
        font-size: 30px;
        margin-bottom: 8px;
    }

    .hero-card p {
        color: #dbeafe !important;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 650;
        color: var(--text-color);
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        padding: 17px;
        border-radius: 14px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 550;
        min-height: 42px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR — BRANDING AND UPLOAD
# ==================================================

with st.sidebar:

    st.markdown("# 📄 DocuAgent")

    st.caption("Your AI document workspace")

    st.divider()

    st.markdown("### Add documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can select multiple files at once.",
    )

    if uploaded_files:

        st.caption(f"{len(uploaded_files)} file(s) selected")

    process_clicked = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files),
    )

    if process_clicked and uploaded_files:

        successful = 0
        failures = []

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                status.info(f"Processing {uploaded_file.name}...")

                file_bytes = uploaded_file.getvalue()

                document_id = hashlib.sha256(
                    uploaded_file.name.encode("utf-8") + file_bytes
                ).hexdigest()[:16]

                sections = extract_document(uploaded_file)

                chunk_count = build_index(
                    sections,
                    document_id=document_id,
                    filename=uploaded_file.name,
                )

                if chunk_count == 0:
                    raise ValueError("No searchable text chunks were created.")

                st.session_state["documents"][document_id] = {
                    "filename": uploaded_file.name,
                    "sections": sections,
                    "chunk_count": chunk_count,
                    "restored": False,
                }

                successful += 1

            except Exception as error:

                failures.append(f"{uploaded_file.name}: {error}")

            progress.progress((index + 1) / len(uploaded_files))

        if successful:

            st.session_state["chat_history"] = []
            st.session_state["chat_document_ids"] = None

            st.session_state["selected_documents"] = []

            status.success(f"Processed {successful} document(s) successfully.")

        if failures:

            for error in failures:
                st.error(error)

        st.rerun()

    st.divider()

    # DOCUMENT FILTERS

    documents = st.session_state["documents"]

    st.markdown("### Document filters")

    document_ids = list(documents.keys())

    if documents:

        selected_document_ids = st.multiselect(
            "Search across",
            options=document_ids,
            default=document_ids,
            format_func=lambda doc_id: documents[doc_id]["filename"],
            key="selected_documents",
            help="Select one or more documents for AI questions.",
        )

        previous_ids = st.session_state["chat_document_ids"]

        if previous_ids is not None and set(previous_ids) != set(selected_document_ids):

            st.session_state["chat_history"] = []

        st.session_state["chat_document_ids"] = list(selected_document_ids)

        st.caption(
            f"{len(selected_document_ids)} of " f"{len(documents)} documents selected"
        )

    else:

        selected_document_ids = []

        st.info("Upload a document to get started.")

    st.divider()

    st.markdown("### AI engine")

    st.markdown("🟢 **Ollama**")

    st.caption("Model: Llama 3.2")

    st.caption("Retrieval: ChromaDB + Hugging Face")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state["chat_history"] = []

        st.rerun()


# ==================================================
# MAIN DASHBOARD HEADER
# ==================================================

st.markdown(
    """
    <div class="hero-card">
        <h1>Document Intelligence Workspace</h1>
        <p>
            Search, analyze, and ask questions about your documents
            using your local AI assistant.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# DASHBOARD METRICS
# ==================================================

documents = st.session_state["documents"]

total_documents = len(documents)

total_chunks = sum(document["chunk_count"] for document in documents.values())

total_messages = sum(
    1 for message in st.session_state["chat_history"] if message["role"] == "user"
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Documents",
        total_documents,
        help="Documents stored in your indexed library.",
    )

with metric2:

    st.metric(
        "Indexed chunks",
        total_chunks,
        help="Text chunks stored for retrieval.",
    )

with metric3:

    st.metric(
        "Questions asked",
        total_messages,
    )

with metric4:

    st.metric(
        "Selected files",
        len(selected_document_ids),
    )


# ==================================================
# MAIN WORKSPACE TABS
# ==================================================

chat_tab, library_tab, export_tab, about_tab = st.tabs(
    [
        "💬 AI Workspace",
        "📚 Document Library",
        "📤 Export Reports",
        "ℹ️ About",
    ]
)


# ==================================================
# TAB 1 — CHAT WORKSPACE
# ==================================================

with chat_tab:

    st.markdown(
        '<div class="section-heading">' "Chat with your documents</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="small-muted">'
        "Ask questions about the selected files. "
        "Use follow-up questions to explore the information."
        "</p>",
        unsafe_allow_html=True,
    )

    if not documents:

        st.info(
            "Your workspace is empty. Upload documents from "
            "the sidebar and click Process Documents."
        )

    elif not selected_document_ids:

        st.warning("Choose at least one document from the sidebar.")

    else:

        if not st.session_state["chat_history"]:

            st.markdown("#### Suggested questions")

            suggestion_columns = st.columns(3)

            suggestions = [
                "Summarize the document",
                "What are the key facts?",
                "What financial figures are mentioned?",
            ]

            for column, suggestion in zip(
                suggestion_columns,
                suggestions,
            ):

                with column:
                    st.caption(f"💡 {suggestion}")

        # DISPLAY CHAT HISTORY

        for message in st.session_state["chat_history"]:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):

                    with st.expander(f"📖 View {len(message['sources'])} source(s)"):

                        for index, source in enumerate(
                            message["sources"],
                            start=1,
                        ):

                            source_id = source.get(
                                "id",
                                f"S{index}",
                            )

                            source_label = source.get(
                                "filename",
                                source.get("label", "Document"),
                            )

                            page = source.get("page", 0)

                            page_label = f" | Page {page}" if page else ""

                            st.markdown(
                                f"**[{source_id}] — " f"{source_label}{page_label}**"
                            )

                            st.write(source.get("text", ""))

        # QUESTION INPUT

        question = st.chat_input("Ask a question about your documents...")

        if question and question.strip():

            st.session_state["chat_history"].append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            history = st.session_state["chat_history"][:-1]

            try:

                with st.spinner("Searching documents and generating your answer..."):

                    result = ask_document(
                        question,
                        chat_history=history[-6:],
                        document_ids=selected_document_ids,
                    )

                answer = result.get(
                    "answer",
                    "I could not generate an answer.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:

                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": (f"Question answering failed: {error}"),
                    }
                )

            st.rerun()


# ==================================================
# TAB 2 — DOCUMENT LIBRARY
# ==================================================

with library_tab:

    st.markdown(
        '<div class="section-heading">' "Your document library</div>",
        unsafe_allow_html=True,
    )

    if not documents:

        st.info("No documents have been indexed yet.")

    else:

        st.caption(
            "Your indexed documents are stored in ChromaDB "
            "and restored when the application restarts."
        )

        for document_id, document in documents.items():

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(f"### 📄 {document['filename']}")

                    st.caption(f"Document ID: {document_id}")

                    if document.get("restored", False):

                        st.caption("Restored from persistent indexed data")

                with col2:

                    st.metric(
                        "Chunks",
                        document["chunk_count"],
                    )

                with st.expander("Preview document text"):

                    for index, section in enumerate(
                        document["sections"],
                        start=1,
                    ):

                        page = section.get("page")

                        if page:
                            st.markdown(f"**Page {page}**")

                        st.caption(f"Text section {index}")

                        st.write(section["text"])


# ==================================================
# TAB 3 — EXPORT REPORTS
# ==================================================

with export_tab:

    st.markdown(
        '<div class="section-heading">' "Export your conversation</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "Download your AI questions, answers, and supporting "
        "document passages as a PDF or editable Word report."
    )

    conversations = get_export_conversations(st.session_state["chat_history"])

    if not conversations:

        st.info(
            "Ask at least one question and receive an answer "
            "before exporting a report."
        )

    else:

        st.metric(
            "Completed question-and-answer pairs",
            len(conversations),
        )

        selected_filenames = [
            st.session_state["documents"][doc_id]["filename"]
            for doc_id in selected_document_ids
            if doc_id in st.session_state["documents"]
        ]

        st.markdown("#### Report contents")

        st.markdown("""
        - Your questions and the AI's answers
        - Source IDs and source filenames
        - Page numbers, where available
        - Supporting text passages
        - Report generation date and time
        """)

        st.markdown("#### Download report")

        try:

            pdf_data = build_pdf_report(
                st.session_state["chat_history"],
                selected_filenames,
            )

            docx_data = build_docx_report(
                st.session_state["chat_history"],
                selected_filenames,
            )

            col_pdf, col_docx = st.columns(2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with col_pdf:

                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"DocuAgent_Report_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_report",
                )

            with col_docx:

                st.download_button(
                    label="📝 Download Word",
                    data=docx_data,
                    file_name=f"DocuAgent_Report_{timestamp}.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True,
                    key="download_docx_report",
                )

            st.caption("PDF is suitable for sharing. Word is editable.")

        except Exception as error:

            st.error(f"Could not generate the report: {error}")


# ==================================================
# TAB 4 — ABOUT
# ==================================================

with about_tab:

    st.markdown(
        '<div class="section-heading">' "About DocuAgent</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "DocuAgent is a document question-answering application "
        "that uses retrieval-augmented generation (RAG) to ground "
        "AI answers in your uploaded documents."
    )

    st.markdown("#### Technology stack")

    st.markdown("""
    - **Frontend:** Streamlit
    - **Agent workflow:** LangGraph
    - **Language model:** Llama 3.2 via Ollama
    - **Embeddings:** Hugging Face Sentence Transformers
    - **Reranking:** Cross-Encoder
    - **Vector database:** ChromaDB
    - **Document processing:** PyMuPDF, python-docx, and text extraction
    """)

    st.info(
        "Answers should be verified against their cited source passages. "
        "Local model execution does not guarantee factual accuracy."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:

    st.caption("📄 DocuAgent · AI Document Intelligence")

with footer_col2:

    st.caption("Powered by local AI · Ollama + ChromaDB")
