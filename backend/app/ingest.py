"""Filings → chunks → Chroma. Introduced in Session 03; parent-child layout in Session 05.

Run from backend/:  python -m app.ingest ../data/filings

This is the offline half of RAG. The online half (search_filings in
tools.py + the retrieve node in graph.py) can only ever be as good as what
happens here — retrieval quality is decided at ingest time.

Session 05 layout: we index small CHILD chunks (precise to match against)
that each point at their PARENT page (rich enough to answer from). Matching
and reading have opposite size preferences — parent-child serves both.
"""

import hashlib
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Children are deliberately small: ~300 chars is one tight claim, which keeps
# embeddings on-topic and BM25 terms dense. The 60-char overlap lets a
# sentence that straddles a boundary survive in at least one child. Parents
# are whole pages — the model reads those. (Session 03 used one 900-char size
# for both jobs; Session 05 splits the jobs.)
CHILD_SIZE = 300
CHILD_OVERLAP = 60

CHROMA_PATH = "./chroma_db"
CHILDREN = "filings"          # queried (vector + BM25)
PARENTS = "filings_parents"   # fetched by id, handed to the model


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Yield (page_number, text) — citations point at pages, so we keep them."""
    if path.suffix == ".pdf":
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages)]
    # .txt files pass straight through; a form-feed (\f) marks a page break,
    # mirroring the PDF path so the sample document cites like a real filing.
    return [(i + 1, text) for i, text in enumerate(path.read_text().split("\f"))]


def main(folder: str) -> None:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )
    # Chroma's default embedding function runs locally (an ONNX MiniLM model,
    # downloaded on first use) — no API key, no per-chunk cost. Alternatives
    # and their trade-offs are a Session 05 discussion.
    children = client.get_or_create_collection(CHILDREN)
    parents = client.get_or_create_collection(PARENTS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP)

    for path in sorted(Path(folder).iterdir()):
        if path.suffix not in {".pdf", ".txt"}:
            continue
        # Idempotency is per *document*: wipe this source's old chunks first,
        # so re-ingesting an edited file (or a changed chunk layout) can never
        # leave stale chunks behind, then insert fresh content-hashed ones.
        children.delete(where={"source": path.name})
        parents.delete(where={"source": path.name})

        c_ids, c_docs, c_metas = [], [], []
        p_ids, p_docs, p_metas = [], [], []
        for page, text in load_pages(path):
            if not text.strip():
                continue
            parent_id = hashlib.sha1(f"{path.name}:{page}".encode()).hexdigest()[:16]
            p_ids.append(parent_id)
            p_docs.append(text)
            p_metas.append({"source": path.name, "page": page})
            for chunk in splitter.split_text(text):
                c_ids.append(hashlib.sha1(f"{parent_id}:{chunk}".encode()).hexdigest()[:16])
                c_docs.append(chunk)
                c_metas.append({"source": path.name, "page": page, "parent_id": parent_id})

        parents.upsert(ids=p_ids, documents=p_docs, metadatas=p_metas)
        children.upsert(ids=c_ids, documents=c_docs, metadatas=c_metas)
        print(f"ingested {path.name}: {len(p_ids)} pages → {len(c_ids)} child chunks")

    print(f"index: {children.count()} children / {parents.count()} parents → {CHROMA_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/filings")
