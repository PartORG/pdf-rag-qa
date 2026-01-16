"""QA service - orchestrates retrieval and answer generation."""

from pdf_qa.core.logging import get_logger
from pdf_qa.core.validators import validate_query, is_describe_query
from pdf_qa.models.qa import Answer, Source
from pdf_qa.rag.retriever import Retriever
from pdf_qa.services.llm import OllamaLLM
from pdf_qa.services.prompts import QA_SYSTEM_PROMPT, build_qa_prompt

logger = get_logger(__name__)


class QAService:
    """Orchestrates Q&A over indexed documents using RAG."""

    def __init__(self, retriever: Retriever, llm: OllamaLLM):
        self.retriever = retriever
        self.llm = llm
        self.model_name = llm.model_name

    def ask(self, question: str, use_diversity: bool = True) -> Answer:
        """Answer a question using retrieval-augmented generation."""
        # Validate and clean query
        question = validate_query(question)

        logger.info(f"Processing question: {question[:50]}...\n")

        # Check if this is a "describe documents" query
        if is_describe_query(question):
            return self._handle_describe_query(question)

        # Retrieve relevant chunks
        if use_diversity:
            results = self.retriever.retrieve_with_diversity(question)
        else:
            results = self.retriever.retrieve(question)

        if not results:
            return Answer(
                question=question,
                answer_text="I could not find any relevant information in the indexed documents.",
                sources=[],
                model_used=self.model_name,
                confidence="low",
            )

        # Build context from results
        context = self.retriever.build_context(results)

        # Generate answer
        prompt = build_qa_prompt(question, context)
        answer_text = self.llm.generate(
            prompt=prompt,
            system_prompt=QA_SYSTEM_PROMPT,
            temperature=0.7,
        )

        # Get sources
        sources = self.retriever.get_sources_for_results(results)

        # Determine confidence based on result scores
        avg_score = sum(r.score for r in results) / len(results)
        confidence = self._score_to_confidence(avg_score)

        return Answer(
            question=question,
            answer_text=answer_text.strip(),
            sources=sources,
            model_used=self.model_name,
            confidence=confidence,
        )

    def _handle_describe_query(self, question: str) -> Answer:
        """Special handling for 'describe documents' type queries."""
        logger.info("Handling document description query\n")

        # Get source previews (first chunk of each file, max 100 chars)
        previews = self.retriever.vector_store.get_source_previews(max_chars=100)

        if not previews:
            return Answer(
                question=question,
                answer_text="No documents have been indexed yet.",
                sources=[],
                model_used=self.model_name,
            )

        # Build simple list
        lines = ["**Indexed Documents:**\n"]
        for filename, preview in sorted(previews.items()):
            lines.append(f"- **{filename}**: {preview}")

        return Answer(
            question=question,
            answer_text="\n".join(lines),
            sources=[
                Source(filename=f, page_number=1, relevance_score=1.0)
                for f in previews.keys()
            ],
            model_used=self.model_name,
        )

    def _score_to_confidence(self, score: float) -> str:
        # Lower distance = more similar = higher confidence
        # For cosine distance, lower is more similar
        if score < 0.3:
            return "high"
        elif score < 0.6:
            return "medium"
        else:
            return "low"


def create_qa_service_from_settings() -> QAService:
    """Factory function to create QAService from config."""
    from pdf_qa.rag.retriever import create_retriever_from_settings
    from pdf_qa.services.llm import create_llm_from_settings

    retriever = create_retriever_from_settings()
    llm = create_llm_from_settings()

    return QAService(
        retriever=retriever,
        llm=llm,
    )
