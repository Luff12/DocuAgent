import io
import pymupdf
from docx import Document

SUPPORTED_FORMATS = (".pdf", ".docx", ".txt")


def extract_document(uploaded_file):
    """
    Extract text from PDF, DOCX, and TXT files.

    Returns:
        List of sections:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]

    Page numbers are None for DOCX and TXT files.
    """

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if not filename.endswith(SUPPORTED_FORMATS):
        raise ValueError("Unsupported file format. Upload a PDF, DOCX, or TXT file.")

    sections = []

    # ---------------------------------------------
    # PDF EXTRACTION
    # ---------------------------------------------

    if filename.endswith(".pdf"):

        try:
            with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text("text").strip()

                    if text:
                        sections.append({"page": page_number, "text": text})

        except Exception as error:
            raise ValueError(f"Could not read PDF file: {error}") from error

        if not sections:
            raise ValueError(
                "No readable text was found in this PDF. "
                "It may be scanned or image-based. "
                "OCR support is required to process it."
            )

    # ---------------------------------------------
    # DOCX EXTRACTION
    # ---------------------------------------------

    elif filename.endswith(".docx"):

        try:
            doc = Document(io.BytesIO(file_bytes))

            content = []

            # Extract paragraphs
            for paragraph in doc.paragraphs:

                text = paragraph.text.strip()

                if text:
                    content.append(text)

            # Extract tables
            for table_index, table in enumerate(doc.tables, start=1):

                table_content = [f"Table {table_index}:"]

                for row in table.rows:

                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]

                    if any(cells):
                        table_content.append(" | ".join(cells))

                content.append("\n".join(table_content))

            full_text = "\n\n".join(content).strip()

            if full_text:
                sections.append({"page": None, "text": full_text})

        except Exception as error:
            raise ValueError(f"Could not read DOCX file: {error}") from error

        if not sections:
            raise ValueError("No readable text was found in this DOCX file.")

    # ---------------------------------------------
    # TXT EXTRACTION
    # ---------------------------------------------

    elif filename.endswith(".txt"):

        try:
            text = file_bytes.decode("utf-8-sig", errors="replace").strip()

            if text:
                sections.append({"page": None, "text": text})

        except Exception as error:
            raise ValueError(f"Could not read TXT file: {error}") from error

        if not sections:
            raise ValueError("The TXT file is empty.")

    return sections
