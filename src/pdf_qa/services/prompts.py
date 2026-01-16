"""Prompt templates for LLM interactions.

Contains system prompts and user prompt templates
for Q&A and summarization tasks.
"""

# System prompt for Q&A
QA_SYSTEM_PROMPT = """You are a helpful research assistant answering questions from academic papers and research documents.

Guidelines:
- Answer based on the provided context - synthesize information from multiple excerpts if needed
- Cite sources using [Source: filename, Page X] format when referencing specific information
- Look for related concepts, synonyms, and technical terms (e.g., acronyms may appear as full names)
- Be concise but comprehensive in your answers
- If you find partial or related information, share what's available rather than saying nothing was found
- Only say "I cannot find this information in the provided documents" if the context is truly unrelated to the question
- Do not invent facts - stick to what's in the context"""


# Template for Q&A prompt
QA_PROMPT_TEMPLATE = """Answer the question using the document excerpts below. Each excerpt includes its source file and page number.

## Document Excerpts:
{context}

## Question:
{question}

## Answer:"""


# System prompt for document summarization
SUMMARY_SYSTEM_PROMPT = """You are a research assistant that creates concise summaries of academic documents.

Guidelines:
- Create a brief 2-3 sentence summary
- Focus on the main topic, methodology, and key findings
- Use clear, accessible language
- Do not include unnecessary details"""


# Template for summarization
SUMMARY_PROMPT_TEMPLATE = """Please summarize the following document excerpt in 2-3 sentences.
Focus on the main topic and key points.

## Document: {filename}

## Content:
{content}

## Summary:"""


def build_qa_prompt(question: str, context: str) -> str:
    """Build a Q&A prompt from question and context.

    Args:
        question: User's question.
        context: Retrieved context from documents.

    Returns:
        Formatted prompt string.
    """
    return QA_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )


def build_summary_prompt(filename: str, content: str) -> str:
    """Build a summarization prompt.

    Args:
        filename: Name of the document file.
        content: Document content to summarize.

    Returns:
        Formatted prompt string.
    """
    # Truncate content if too long
    max_content = 3000
    if len(content) > max_content:
        content = content[:max_content] + "..."

    return SUMMARY_PROMPT_TEMPLATE.format(
        filename=filename,
        content=content,
    )
