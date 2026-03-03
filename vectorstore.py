"""
DataMind AI — Multi-Source Document Loader
==========================================
Supports: PDF, CSV, TXT, JSON, Web URLs, SQL databases
Each loader extracts text + rich metadata for downstream filtering.
"""

import os
import json
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Unified document representation across all source types."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_type: str = ""
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            import hashlib
            self.doc_id = hashlib.md5(
                (self.content[:100] + self.source).encode()
            ).hexdigest()[:12]


class DocumentLoader:
    """
    Universal document loader supporting multiple source types.
    Auto-detects source type and applies appropriate loader.
    """

    SUPPORTED_TYPES = {
        ".pdf":  "pdf",
        ".csv":  "csv",
        ".txt":  "text",
        ".md":   "text",
        ".json": "json",
        ".db":   "sqlite",
    }

    def __init__(self, config: dict = None):
        self.config = config or {}

    def load(self, source: str) -> List[Document]:
        """
        Auto-detect source type and load documents.

        Args:
            source: File path, directory path, or URL

        Returns:
            List of Document objects
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self.load_url(source)

        path = Path(source)

        if path.is_dir():
            return self.load_directory(source)

        suffix = path.suffix.lower()
        loader_type = self.SUPPORTED_TYPES.get(suffix)

        if loader_type == "pdf":
            return self.load_pdf(source)
        elif loader_type == "csv":
            return self.load_csv(source)
        elif loader_type == "text":
            return self.load_text(source)
        elif loader_type == "json":
            return self.load_json(source)
        elif loader_type == "sqlite":
            return self.load_sqlite(source)
        else:
            logger.warning(f"Unsupported file type: {suffix}. Trying as text.")
            return self.load_text(source)

    def load_directory(self, dir_path: str) -> List[Document]:
        """Recursively load all supported files in a directory."""
        docs = []
        for path in Path(dir_path).rglob("*"):
            if path.suffix.lower() in self.SUPPORTED_TYPES:
                try:
                    loaded = self.load(str(path))
                    docs.extend(loaded)
                    logger.info(f"  Loaded {len(loaded)} docs from {path.name}")
                except Exception as e:
                    logger.error(f"  Failed to load {path.name}: {e}")
        logger.info(f"Total loaded from directory: {len(docs)} documents")
        return docs

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF and extract text per page with metadata."""
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
            return []

        docs = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        content=text,
                        metadata={
                            "page": page_num + 1,
                            "total_pages": len(reader.pages),
                            "file_name": Path(file_path).name,
                            "loaded_at": datetime.utcnow().isoformat(),
                        },
                        source=file_path,
                        source_type="pdf",
                    ))
        logger.info(f"  PDF: {Path(file_path).name} → {len(docs)} pages")
        return docs

    def load_csv(self, file_path: str, text_col: str = None) -> List[Document]:
        """Load CSV — each row becomes a document with all columns as metadata."""
        df = pd.read_csv(file_path)
        docs = []

        # Auto-detect longest text column if not specified
        if text_col is None:
            str_cols = df.select_dtypes(include="object").columns
            if len(str_cols) > 0:
                text_col = df[str_cols].apply(lambda c: c.str.len().mean()).idxmax()

        for idx, row in df.iterrows():
            content = str(row.get(text_col, "")) if text_col else row.to_json()
            if not content.strip():
                continue
            docs.append(Document(
                content=content,
                metadata={
                    "row_index": idx,
                    "file_name": Path(file_path).name,
                    **{k: str(v) for k, v in row.items() if k != text_col},
                },
                source=file_path,
                source_type="csv",
            ))
        logger.info(f"  CSV: {Path(file_path).name} → {len(docs)} rows")
        return docs

    def load_text(self, file_path: str) -> List[Document]:
        """Load plain text or markdown file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        doc = Document(
            content=content,
            metadata={
                "file_name": Path(file_path).name,
                "file_size_kb": round(Path(file_path).stat().st_size / 1024, 2),
                "loaded_at": datetime.utcnow().isoformat(),
            },
            source=file_path,
            source_type="text",
        )
        return [doc]

    def load_json(self, file_path: str, text_key: str = None) -> List[Document]:
        """Load JSON file — handles both objects and arrays."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        items = data if isinstance(data, list) else [data]

        for i, item in enumerate(items):
            if isinstance(item, dict):
                # Use specified key or concatenate all string values
                if text_key and text_key in item:
                    content = str(item[text_key])
                else:
                    content = " | ".join(
                        f"{k}: {v}" for k, v in item.items()
                        if isinstance(v, (str, int, float))
                    )
                metadata = {k: str(v) for k, v in item.items()
                            if isinstance(v, (str, int, float))}
            else:
                content = str(item)
                metadata = {}

            if content.strip():
                docs.append(Document(
                    content=content,
                    metadata={"item_index": i, "file_name": Path(file_path).name, **metadata},
                    source=file_path,
                    source_type="json",
                ))

        logger.info(f"  JSON: {Path(file_path).name} → {len(docs)} items")
        return docs

    def load_sqlite(self, db_path: str, tables: List[str] = None) -> List[Document]:
        """Load all rows from SQLite database tables as documents."""
        docs = []
        with sqlite3.connect(db_path) as conn:
            if tables is None:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                try:
                    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1000", conn)
                    for idx, row in df.iterrows():
                        content = " | ".join(f"{k}: {v}" for k, v in row.items())
                        docs.append(Document(
                            content=content,
                            metadata={"table": table, "row_index": idx,
                                      "db_path": db_path},
                            source=db_path,
                            source_type="sqlite",
                        ))
                    logger.info(f"  SQLite table '{table}': {len(df)} rows")
                except Exception as e:
                    logger.error(f"  Failed to load table {table}: {e}")
        return docs

    def load_url(self, url: str) -> List[Document]:
        """Load web page content, stripping HTML tags."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("requests/bs4 not installed. Run: pip install requests beautifulsoup4")
            return []

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else url

        doc = Document(
            content=text,
            metadata={
                "url": url,
                "title": title,
                "loaded_at": datetime.utcnow().isoformat(),
            },
            source=url,
            source_type="url",
        )
        logger.info(f"  URL: {url} → {len(text)} chars")
        return [doc]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DocumentLoader()
    # Example: load a directory of docs
    # docs = loader.load("data/raw/")
    # print(f"Loaded {len(docs)} documents")
