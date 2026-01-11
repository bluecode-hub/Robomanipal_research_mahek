from indexer import build_index
from retriever import SemanticRetriever
from agent import SingleLevelAgent
from llm.client import GeminiClient
from llm.controller import LLMController
from llm.logger import logger


def main():
    """Main entry point for the single-level RAG agent"""
    
    logger.info("Initializing Single-Level RAG Agent System...")
    
    # Initialize components
    logger.info("Building index...")
    embedder, store = build_index()
    
    retriever = SemanticRetriever(embedder, store)
    client = GeminiClient()
    llm = LLMController(client)
    
    # Create agent
    agent = SingleLevelAgent(retriever, llm)
    logger.info("Agent initialized successfully")
    
    # Interactive loop
    print("\n" + "="*60)
    print("Single-Level RAG Agent System")
    print("="*60)
    print("Enter your questions. Type 'quit' to exit.")
    print("Type 'history' to see conversation history.")
    print("Type 'clear' to clear history.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "history":
                history = agent.get_conversation_history()
                if history:
                    print("\n--- Conversation History ---")
                    for i, msg in enumerate(history, 1):
                        print(f"{i}. [{msg['role'].upper()}]: {msg['content'][:100]}...")
                    print("--- End History ---\n")
                else:
                    print("No conversation history yet.\n")
                continue
            
            if user_input.lower() == "clear":
                agent.clear_history()
                print("History cleared.\n")
                continue
            
            # Process query through agent
            print("\n[Agent Processing...]")
            result = agent.process(user_input)
            
            # Display results
            print(f"\n[Tool Used]: {result['tool_chosen']}")
            print(f"[Reasoning]: {result['tool_reasoning']}")
            print(f"\n[Agent]: {result['reply']}")
            print(f"[Word Count]: {result['word_count']}")
            
            if result['retrieved_context']:
                print(f"\n[Context Retrieved]: {result['retrieved_context'][:200]}...")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
