import json
import re
from typing import Optional, Dict, List, Any
from retriever import SemanticRetriever
from llm.controller import LLMController
from llm.logger import logger


class Tool:
    """Base class for agent tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class RAGTool(Tool):
    """Tool for retrieving information using RAG"""
    def __init__(self, retriever: SemanticRetriever):
        super().__init__(
            name="retrieve_context",
            description="Retrieve relevant information from the knowledge base. Use this when you need context to answer a question."
        )
        self.retriever = retriever
    
    def execute(self, query: str, top_k: int = 3) -> str:
        """Execute retrieval and return formatted context"""
        logger.info(f"RAGTool retrieving context for: {query}")
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        
        if not retrieved:
            return "No relevant context found."
        
        context_blocks = []
        for score, meta in retrieved:
            context_blocks.append(
                f"[Source: {meta['source']} | Score: {score:.2f}]\n{meta['text']}"
            )
        
        context = "\n\n".join(context_blocks)
        logger.info(f"Retrieved {len(retrieved)} documents")
        return context


class DirectAnswerTool(Tool):
    """Tool for generating direct answers without retrieval"""
    def __init__(self):
        super().__init__(
            name="direct_answer",
            description="Generate a direct answer to the question without retrieving context. Use this for general knowledge or simple queries."
        )
    
    def execute(self, query: str) -> str:
        """Return a marker indicating direct answer should be used"""
        return f"DIRECT_ANSWER_MODE: {query}"


class SingleLevelAgent:
    """
    Single-level RAG Agent that decides which tool to use and processes the result.
    No framework dependencies - pure Python implementation.
    """
    
    def __init__(self, retriever: SemanticRetriever, llm: LLMController):
        self.retriever = retriever
        self.llm = llm
        
        # Initialize tools
        self.tools: Dict[str, Tool] = {
            "retrieve_context": RAGTool(retriever),
            "direct_answer": DirectAnswerTool()
        }
        
        self.conversation_history: List[Dict[str, str]] = []
        self.max_iterations = 1  # Single-level means only one decision cycle
    
    def _get_tool_definitions(self) -> str:
        """Generate tool definitions for the LLM"""
        tools_desc = "Available Tools:\n"
        for tool_name, tool in self.tools.items():
            tools_desc += f"- {tool_name}: {tool.description}\n"
        return tools_desc
    
    def _build_decision_prompt(self, user_query: str) -> str:
        """Build a prompt for the agent to decide which tool to use"""
        tools_def = self._get_tool_definitions()
        
        decision_prompt = f"""You are an intelligent agent that decides which tool to use for answering questions.

{tools_def}

User Query: {user_query}

Analyze the query and respond with ONLY valid JSON in this exact format:
{{
    "reasoning": "Brief explanation of why you chose this tool",
    "tool_choice": "retrieve_context OR direct_answer",
    "tool_input": {{"query": "The query to pass to the tool"}}
}}

Rules:
1. Choose "retrieve_context" if the query requires specific information, facts, or knowledge from a database
2. Choose "direct_answer" for general knowledge, common sense, or simple questions
3. Always provide valid JSON
4. The tool_input must be a valid JSON object"""
        
        return decision_prompt
    
    def _parse_tool_decision(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM's decision about which tool to use"""
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r"```json|```", "", raw_output).strip()
            
            decision = json.loads(cleaned)
            
            # Validate required fields
            if "tool_choice" not in decision or "tool_input" not in decision:
                logger.error("Invalid decision format: missing required fields")
                return None
            
            if decision["tool_choice"] not in self.tools:
                logger.error(f"Unknown tool: {decision['tool_choice']}")
                return None
            
            return decision
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool decision: {e}")
            return None
    
    def _build_final_answer_prompt(self, user_query: str, tool_result: str) -> str:
        """Build prompt for generating the final answer"""
        final_prompt = f"""Based on the following information, provide a helpful answer to the user's question.

User Question: {user_query}

Retrieved Context/Information:
{tool_result}

Respond with ONLY valid JSON in this exact format:
{{
    "reply": "Your helpful answer here",
    "word_count": <number of words in reply>
}}

Rules:
1. If the context is "No relevant context found.", politely inform the user you cannot answer
2. If context is provided, use it to answer the question
3. Always provide the response in valid JSON format
4. Count the words in your reply accurately"""
        
        return final_prompt
    
    def process(self, user_query: str) -> Dict[str, Any]:
        """
        Main agent processing loop - single level means one tool decision and one answer generation.
        
        Flow:
        1. Decide which tool to use
        2. Execute the tool
        3. Generate final answer
        """
        logger.info(f"Agent processing query: {user_query}")
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Step 1: Get tool decision from LLM
        decision_prompt = self._build_decision_prompt(user_query)
        logger.info("Step 1: Getting tool decision from LLM")
        
        decision_raw = self.llm.client.generate(decision_prompt)
        logger.info(f"Raw decision: {decision_raw}")
        
        decision = self._parse_tool_decision(decision_raw)
        
        if not decision:
            # Fallback to retrieve_context if parsing fails
            logger.warning("Tool decision parsing failed, using retrieve_context as fallback")
            decision = {
                "tool_choice": "retrieve_context",
                "tool_input": {"query": user_query},
                "reasoning": "Fallback due to parsing error"
            }
        
        logger.info(f"Tool chosen: {decision['tool_choice']}")
        logger.info(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        
        # Step 2: Execute the chosen tool
        tool_name = decision["tool_choice"]
        tool = self.tools[tool_name]
        
        logger.info(f"Step 2: Executing tool '{tool_name}'")
        
        # Extract tool input, handling both direct strings and dict formats
        tool_input = decision["tool_input"]
        if isinstance(tool_input, dict):
            tool_result = tool.execute(**tool_input)
        else:
            tool_result = tool.execute(query=tool_input)
        
        logger.info(f"Tool result length: {len(tool_result)} chars")
        
        # Step 3: Generate final answer
        logger.info("Step 3: Generating final answer")
        
        final_prompt = self._build_final_answer_prompt(user_query, tool_result)
        final_raw = self.llm.client.generate(final_prompt)
        logger.info(f"Raw final answer: {final_raw}")
        
        # Parse final answer
        try:
            cleaned = re.sub(r"```json|```", "", final_raw).strip()
            final_answer = json.loads(cleaned)
            
            if "reply" not in final_answer or "word_count" not in final_answer:
                logger.error("Invalid final answer format")
                final_answer = {
                    "reply": final_raw,
                    "word_count": len(final_raw.split())
                }
        except json.JSONDecodeError:
            logger.error("Failed to parse final answer as JSON")
            final_answer = {
                "reply": final_raw,
                "word_count": len(final_raw.split())
            }
        
        # Store agent's response
        self.conversation_history.append({
            "role": "agent",
            "content": final_answer.get("reply", "")
        })
        
        # Compile full result
        result = {
            "user_query": user_query,
            "tool_chosen": tool_name,
            "tool_reasoning": decision.get("reasoning", ""),
            "retrieved_context": tool_result if tool_name == "retrieve_context" else None,
            "reply": final_answer.get("reply", ""),
            "word_count": final_answer.get("word_count", 0),
            "conversation_history": self.conversation_history
        }
        
        logger.info(f"Agent response generated: {len(final_answer.get('reply', ''))} chars")
        return result
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the conversation history"""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
