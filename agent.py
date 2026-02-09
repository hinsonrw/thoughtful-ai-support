"""
Thoughtful AI Customer Support Agent

Core agent logic with semantic matching for predefined Q&A
and LLM fallback for unmatched queries.
"""

import os
from typing import Optional
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PREDEFINED_QA = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": "EVA automates the process of verifying a patient's eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
    }
]

SYSTEM_PROMPT = """You are a helpful customer support agent for Thoughtful AI, a company that provides AI-powered automation agents for healthcare processes.

Thoughtful AI's main products include:
- EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.
- CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.
- PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.

Be helpful, professional, and concise. If you don't know something specific about Thoughtful AI, be honest about it while remaining helpful."""

SIMILARITY_THRESHOLD = 0.55


class ThoughtfulAgent:
    """Customer support agent with semantic matching and LLM fallback."""

    def __init__(self):
        self.client = OpenAI()
        self.qa_embeddings: Optional[np.ndarray] = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Pre-compute embeddings for predefined questions."""
        questions = [qa["question"] for qa in PREDEFINED_QA]
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=questions
        )
        self.qa_embeddings = np.array([e.embedding for e in response.data])

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _find_best_match(self, query: str) -> tuple[Optional[str], float]:
        """Find best matching predefined answer using semantic similarity."""
        query_embedding = self._get_embedding(query)

        similarities = [
            self._cosine_similarity(query_embedding, qa_emb)
            for qa_emb in self.qa_embeddings
        ]

        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score >= SIMILARITY_THRESHOLD:
            return PREDEFINED_QA[best_idx]["answer"], best_score

        return None, best_score

    def _get_llm_response(self, query: str) -> str:
        """Get response from LLM for unmatched queries."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content

    def respond(self, query: str) -> tuple[str, str]:
        """
        Generate response to user query.

        Returns:
            tuple: (response_text, source) where source is 'predefined' or 'llm'
        """
        predefined_answer, score = self._find_best_match(query)

        if predefined_answer:
            return predefined_answer, "predefined"

        llm_response = self._get_llm_response(query)
        return llm_response, "llm"
