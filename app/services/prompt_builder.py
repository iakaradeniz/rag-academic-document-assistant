from typing import List, Dict


class PromptBuilder:
    """
    Builds a structured prompt for RAG-based question answering.
    """

    SYSTEM_PROMPT = """
You are an academic document assistant.
Answer the question ONLY using the provided context.
If the answer is not contained in the context, say:
"I could not find the answer in the provided documents."

Rules:
- Do NOT use external knowledge
- Be concise and factual
- Always cite the source and page number
"""

    def build(self, question: str, contexts: List[Dict]) -> Dict[str, str]:
        context_text = self._format_context(contexts)

        user_prompt = f"""
Context:
{context_text}

Question:
{question}

Answer:
"""

        return {
            "system": self.SYSTEM_PROMPT.strip(),
            "user": user_prompt.strip()
        }

    def _format_context(self, contexts: List[Dict]) -> str:
        formatted_chunks = []

        for i, ctx in enumerate(contexts, start=1):
            chunk = f"""
[Source {i}]
Text: {ctx['text']}
Source: {ctx.get('source')}
Page: {ctx.get('page')}
"""
            formatted_chunks.append(chunk.strip())

        return "\n\n".join(formatted_chunks)
