"""Document loader, semantic chunker, and knowledge base parser."""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from agentone.core.state import DocumentChunk


class DocumentChunker:
    """Splits structured text and markdown into semantically coherent knowledge chunks."""

    def __init__(self, target_chunk_size: int = 400, chunk_overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_markdown(self, text: str, source_name: str, category: str = "general") -> List[DocumentChunk]:
        """Split markdown along header sections (`#`, `##`, `###`) and paragraphs."""
        sections = re.split(r"(?=(?:^|\n)#{1,3}\s+)", text)
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # If section is small enough, keep as single chunk
            if len(sec_clean.split()) <= self.target_chunk_size:
                chunk_id = f"{source_name}_chunk_{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        doc_id=chunk_id,
                        content=sec_clean,
                        source=source_name,
                        category=category,
                        metadata={"chunk_index": chunk_idx, "word_count": len(sec_clean.split())},
                    )
                )
                chunk_idx += 1
            else:
                # Subdivide by paragraphs
                paragraphs = sec_clean.split("\n\n")
                buffer = ""
                for p in paragraphs:
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    if len((buffer + "\n\n" + p_clean).split()) > self.target_chunk_size and buffer:
                        chunk_id = f"{source_name}_chunk_{chunk_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=chunk_id,
                                content=buffer.strip(),
                                source=source_name,
                                category=category,
                                metadata={"chunk_index": chunk_idx, "word_count": len(buffer.split())},
                            )
                        )
                        chunk_idx += 1
                        buffer = p_clean
                    else:
                        buffer = f"{buffer}\n\n{p_clean}".strip()

                if buffer:
                    chunk_id = f"{source_name}_chunk_{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            doc_id=chunk_id,
                            content=buffer.strip(),
                            source=source_name,
                            category=category,
                            metadata={"chunk_index": chunk_idx, "word_count": len(buffer.split())},
                        )
                    )
                    chunk_idx += 1

        return chunks


def load_knowledge_directory(dir_path: str) -> List[DocumentChunk]:
    """Scan and ingest all markdown and text documents in a directory."""
    path = Path(dir_path)
    if not path.exists():
        return []

    chunker = DocumentChunker()
    all_chunks: List[DocumentChunk] = []

    for file_path in path.glob("**/*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            category = file_path.parent.name if file_path.parent != path else "general"
            try:
                content = file_path.read_text(encoding="utf-8")
                chunks = chunker.chunk_markdown(content, source_name=file_path.name, category=category)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return all_chunks
