from retriever import SemanticRetriever
from llm.controller import LLMController
class RAGPipeline:
    def __init__(self,retriever:SemanticRetriever,llm:LLMController):
        self.retriever=retriever
        self.llm=llm
    def answer(self,query:str,top_k=3)->dict:
        retrieved=self.retriever.retrieve(query,top_k=top_k)
        if not retrieved:
            return {
                "reply":"i cannot answer because no relevant context was retrievd.",
                "word_count":0

            }
        context_blocks=[]
        for score,meta in retrieved:
            context_blocks.append(
                f"[Source:{meta['source']}]\n{meta['text']}"

            )
        context="\n\n".join(context_blocks)

        rag_input=f"""
Context:
{context}

Question:
{query}

INSTRUCTION:
Use ONLY the context above.
If the context does not contain the answer, reply that you cannot answer.
"""

        return self.llm.run(rag_input.strip())

       