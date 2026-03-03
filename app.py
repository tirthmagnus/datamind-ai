"""
DataMind AI — Intelligent Text Chunker
=======================================
Splits documents into optimally-sized chunks for embedding.
Strategies:
  - Recursive character splitter (default)
  - Sentence-aware splitter
  - Semantic splitter (embedding-based)
"""

import re
import logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata inherited from parent document."""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class RecursiveChunker:
    """
    Recursively splits text using a hierarchy of separators.
    Mimics LangChain's RecursiveCharacterTextSplitter logic.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        min_chunk_size: int = 50,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or self.DEFAULT_SEPARATORS
        self.min_chunk_size = min_chunk_size

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining  = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (separator if current else "") + split

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) >= self.min_chunk_size:
                        chunks.append(current)
                    # Handle overlap
                    if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                        current = current[-self.chunk_overlap:] + (separator if separator else "") + split
                    else:
                        current = split
                else:
                    # Split is larger than chunk_size — recurse
                    if len(split) > self.chunk_size and remaining:
                        sub_chunks = self._split_text(split, remaining)
                        chunks.extend(sub_chunks[:-1])
                        current = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current = split

        if current and len(current) >= self.min_chunk_size:
            chunks.append(current)

        return chunks

    def chunk_document(self, doc) -> List[Chunk]:
        """Chunk a single Document object."""
        if not doc.content or not doc.content.strip():
            return []

        raw_chunks = self._split_text(doc.content.strip(), self.separators)
        chunks = []

        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue

            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_size": len(text),
                    "source": doc.source,
                    "source_type": doc.source_type,
                },
                doc_id=doc.doc_id,
                chunk_index=i,
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, docs: list) -> List[Chunk]:
        """Chunk a list of Document objects."""
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(docs)} documents → {len(all_chunks)} chunks "
            f"(avg size: {sum(len(c.content) for c in all_chunks) // max(len(all_chunks), 1)} chars)"
        )
        return all_chunks


class SentenceChunker:
    """
    Sentence-aware chunker — never splits mid-sentence.
    Better for Q&A tasks where sentence completeness matters.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 1):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap  # in sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc) -> List[Chunk]:
        sentences = self._split_sentences(doc.content)
        chunks = []
        current_sentences = []
        current_len = 0
        chunk_idx = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={**doc.metadata, "chunk_index": chunk_idx,
                               "source": doc.source},
                    doc_id=doc.doc_id,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
                # Keep overlap sentences
                current_sentences = current_sentences[-self.chunk_overlap:]
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += len(sent)

        if current_sentences:
            chunks.append(Chunk(
                content=" ".join(current_sentences),
                metadata={**doc.metadata, "chunk_index": chunk_idx,
                           "source": doc.source},
                doc_id=doc.doc_id,
                chunk_index=chunk_idx,
            ))

        return chunks

    def chunk_documents(self, docs: list) -> List[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Sentence-chunked {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks


def get_chunker(strategy: str = "recursive", **kwargs):
    """Factory function to get the right chunker."""
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    elif strategy == "sentence":
        return SentenceChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
