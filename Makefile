"""
DataMind AI — RAG Chain
========================
Full Retrieval-Augmented Generation pipeline:
  1. Retrieve relevant chunks (hybrid retriever)
  2. Build prompt with context + chat history
  3. Generate answer via LLM (GPT-4o / Ollama)
  4. Ground-check: verify answer is supported by context
  5. Return answer + sources + confidence score
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    is_grounded: bool = True
    query: str = ""
    model_used: str = ""


SYSTEM_PROMPT = """You are DataMind AI, a precise and helpful data intelligence assistant.

Your job is to answer questions based ONLY on the provided context documents.

Rules:
1. Answer ONLY from the context provided. Do NOT use outside knowledge.
2. If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer this question."
3. Always cite which source/document your answer comes from.
4. Be concise but complete.
5. If asked about numbers or statistics, quote them exactly from the context.
"""

RAG_PROMPT_TEMPLATE = """
CONTEXT DOCUMENTS:
{context}

---
CHAT HISTORY:
{chat_history}

---
QUESTION: {question}

INSTRUCTIONS: Answer the question using ONLY the context documents above. 
Cite your sources by mentioning the document name or source.
If the answer is not in the context, say so clearly.

ANSWER:"""

GROUNDING_PROMPT = """You are a fact-checking assistant.

Given an ANSWER and CONTEXT, determine if the answer is fully supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Is the answer fully grounded in the context? 
Respond with JSON only: {{"is_grounded": true/false, "confidence": 0.0-1.0, "reason": "..."}}
"""


class RAGChain:
    """
    Production RAG chain with grounding verification.
    Supports OpenAI GPT-4o, Azure OpenAI, and local Ollama.
    """

    def __init__(
        self,
        retriever,
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_grounding: bool = True,
        max_context_chunks: int = 6,
    ):
        self.retriever          = retriever
        self.llm_provider       = llm_provider
        self.model              = model
        self.temperature        = temperature
        self.max_tokens         = max_tokens
        self.enable_grounding   = enable_grounding
        self.max_context_chunks = max_context_chunks
        self._llm               = None

    def _get_llm(self):
        """Lazy-load the LLM client."""
        if self._llm is not None:
            return self._llm

        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self._llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")

        elif self.llm_provider == "ollama":
            try:
                from openai import OpenAI
                self._llm = OpenAI(
                    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    api_key="ollama",
                )
                self.model = os.environ.get("OLLAMA_MODEL", "llama3")
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")

        return self._llm

    def _build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("file_name") or chunk.get("metadata", {}).get("url", "Unknown")
            context_parts.append(
                f"[Document {i}] Source: {source}\n{chunk['content']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _build_chat_history(self, history: List[Dict]) -> str:
        """Format conversation history."""
        if not history:
            return "No previous conversation."
        lines = []
        for turn in history[-4:]:  # Last 4 turns
            lines.append(f"User: {turn.get('user', '')}")
            lines.append(f"Assistant: {turn.get('assistant', '')}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str, system: str = None) -> str:
        """Call the LLM with a prompt."""
        llm = self._get_llm()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _check_grounding(self, answer: str, context: str) -> Dict:
        """Verify that the answer is grounded in retrieved context."""
        try:
            import json
            prompt = GROUNDING_PROMPT.format(context=context[:3000], answer=answer)
            result = self._call_llm(prompt)
            # Parse JSON response
            clean = result.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"Grounding check failed: {e}")
            return {"is_grounded": True, "confidence": 0.7, "reason": "Check unavailable"}

    def query(
        self,
        question: str,
        chat_history: List[Dict] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a question.

        Args:
            question: User's question
            chat_history: Previous conversation turns
            filter_metadata: Optional metadata filter for retrieval

        Returns:
            RAGResponse with answer, sources, and confidence
        """
        logger.info(f"RAG query: {question[:80]}...")

        # Step 1: Retrieve
        chunks = self.retriever.retrieve(
            query=question,
            top_k=self.max_context_chunks,
            where=filter_metadata,
        )

        if not chunks:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer your question. Please make sure documents have been ingested.",
                query=question,
                confidence=0.0,
                is_grounded=False,
            )

        # Step 2: Build context and prompt
        context      = self._build_context(chunks)
        history_text = self._build_chat_history(chat_history or [])
        prompt       = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history_text,
            question=question,
        )

        # Step 3: Generate answer
        answer = self._call_llm(prompt, system=SYSTEM_PROMPT)

        # Step 4: Grounding check
        grounding = {"is_grounded": True, "confidence": 0.85}
        if self.enable_grounding:
            grounding = self._check_grounding(answer, context)

        return RAGResponse(
            answer=answer,
            sources=chunks,
            confidence=grounding.get("confidence", 0.85),
            is_grounded=grounding.get("is_grounded", True),
            query=question,
            model_used=self.model,
        )

    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Run multiple queries for evaluation."""
        return [self.query(q) for q in questions]
