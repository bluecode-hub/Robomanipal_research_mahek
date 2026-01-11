from indexer import build_index
from retriever import SemanticRetriever
from rag_pipeline import RAGPipeline

from llm.client import GeminiClient
from llm.controller import LLMController

def main():
    embedder, store = build_index()
    retriever = SemanticRetriever(embedder, store)

    client = GeminiClient()
    llm = LLMController(client)
    rag = RAGPipeline(retriever, llm)

    query = input("Enter your question: ")
    print(rag.answer(query))

if __name__ == "__main__":
    main()
