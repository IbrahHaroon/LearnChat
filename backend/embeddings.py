"""
embeddings.py — HuggingFace embeddings via InferenceClient

HuggingFace's old REST endpoint (api-inference.huggingface.co) is deprecated.
huggingface_hub.InferenceClient handles routing to the correct endpoint automatically.
"""

import os
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient


class HFEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model
        self.client = InferenceClient(api_key=api_key)

    def embed_documents(self, texts: list) -> list:
        results = self.client.text_embeddings(texts, model=self.model)
        return [list(e) for e in results]

    def embed_query(self, text: str) -> list:
        results = self.client.text_embeddings([text], model=self.model)
        return list(results[0])
