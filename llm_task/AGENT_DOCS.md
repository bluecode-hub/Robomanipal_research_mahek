# Single-Level RAG Agentic System

A framework-free, pure Python implementation of a single-level RAG (Retrieval-Augmented Generation) agent that makes intelligent decisions about which tools to use for answering questions.

## Architecture

### System Components

```
User Query
    ↓
Agent Decision Layer
    ├─ Analyzes query
    └─ Selects appropriate tool
    ↓
Tool Execution Layer
    ├─ RAG Tool (retrieve_context)
    │  └─ Retrieves relevant documents from knowledge base
    │
    └─ Direct Answer Tool (direct_answer)
       └─ Generates answers without retrieval
    ↓
Answer Generation Layer
    └─ Generates final response using tool output
    ↓
User Response
```

## Single-Level Explanation

A **single-level agent** makes ONE tool decision per query:
1. **Analyze** the query and decide which tool is best suited
2. **Execute** the chosen tool
3. **Generate** the final answer based on tool output

This is different from multi-level agents which would loop and make multiple decisions.

## Core Concepts

### 1. Tools
Tools are the actions an agent can take. Currently implemented:

#### `retrieve_context`
- **Purpose**: Retrieve relevant information from the knowledge base
- **When to use**: Factual questions, specific information lookups, domain-specific queries
- **Input**: Query string
- **Output**: Formatted context from retrieved documents

#### `direct_answer`
- **Purpose**: Generate answers using general knowledge
- **When to use**: General knowledge questions, conversational queries, simple facts
- **Input**: Query string
- **Output**: Marker for direct answer mode

### 2. Agent Workflow

**Step 1: Tool Selection**
- Agent receives user query
- Sends query to LLM with tool definitions
- LLM decides which tool is most appropriate
- LLM provides reasoning for the decision

**Step 2: Tool Execution**
- Chosen tool is executed with relevant parameters
- Tool returns structured output

**Step 3: Answer Generation**
- Tool output is combined with original query
- LLM generates final answer
- Response is formatted and returned

### 3. Conversation History
- All user queries and agent responses are logged
- Can be retrieved and cleared as needed
- Useful for understanding agent behavior

## Usage

### Basic Usage

```python
from indexer import build_index
from retriever import SemanticRetriever
from agent import SingleLevelAgent
from llm.client import GeminiClient
from llm.controller import LLMController

# Initialize
embedder, store = build_index()
retriever = SemanticRetriever(embedder, store)
client = GeminiClient()
llm = LLMController(client)

# Create agent
agent = SingleLevelAgent(retriever, llm)

# Process query
result = agent.process("What is the capital of France?")
print(result['reply'])
```

### Result Structure

```python
{
    "user_query": "The user's question",
    "tool_chosen": "retrieve_context or direct_answer",
    "tool_reasoning": "Why this tool was chosen",
    "retrieved_context": "Retrieved documents (if tool_chosen == 'retrieve_context')",
    "reply": "The agent's answer",
    "word_count": 42,
    "conversation_history": [...]
}
```

### Interactive Mode

Run the interactive agent:
```bash
python agent_main.py
```

Commands:
- Enter questions normally
- `quit` - Exit the program
- `history` - View conversation history
- `clear` - Clear conversation history

## Implementation Details

### Key Classes

#### `Tool` (Base Class)
Abstract base class for all tools with:
- `name`: Tool identifier
- `description`: What the tool does
- `execute(**kwargs)`: Execute the tool

#### `RAGTool`
Implements retrieval-based context gathering using the existing `SemanticRetriever`.

#### `DirectAnswerTool`
Simple tool that marks queries for direct answer generation.

#### `SingleLevelAgent`
Main agent class that:
- Manages tool selection and execution
- Builds decision prompts for LLM
- Parses LLM responses
- Maintains conversation history
- Generates final answers

### JSON-Based Communication

All LLM interactions use JSON for structured communication:

**Tool Decision Format:**
```json
{
    "reasoning": "Why this tool was chosen",
    "tool_choice": "retrieve_context or direct_answer",
    "tool_input": {"query": "The query string"}
}
```

**Final Answer Format:**
```json
{
    "reply": "The answer to the user's question",
    "word_count": 42
}
```

## Features

✅ **No External Framework** - Pure Python, no LangChain/AutoGen dependencies
✅ **Single-Level Decision Making** - One tool choice per query
✅ **Tool-Based Architecture** - Extensible tool system
✅ **Logging** - Complete logging of all agent decisions and LLM interactions
✅ **Error Handling** - Graceful fallbacks for parsing failures
✅ **Conversation History** - Track all interactions
✅ **JSON Validation** - Structured output validation
✅ **Modularity** - Easy to add new tools

## Extending the System

### Adding a New Tool

```python
class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculate",
            description="Perform mathematical calculations"
        )
    
    def execute(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Invalid expression"

# Register in SingleLevelAgent.__init__:
self.tools["calculate"] = CalculatorTool()
```

### Customizing Decision Logic

Modify `_build_decision_prompt()` in `SingleLevelAgent` to change how the LLM decides on tools.

### Custom Answer Generation

Modify `_build_final_answer_prompt()` to customize the final answer generation.

## Logging

The system uses Python's logging module with the following coverage:
- Agent initialization
- Tool decisions and reasoning
- Tool execution details
- LLM requests and responses
- Error cases and fallbacks
- Answer generation

Check logs for debugging tool selection issues.

## Performance Considerations

- **API Calls**: Each query makes 2 LLM API calls (tool decision + answer generation)
- **Retrieval**: Semantic similarity search is O(n) where n = number of documents
- **Memory**: Conversation history grows with each interaction

## Limitations

1. **Single-Level Only** - Cannot chain multiple tool calls in one session
2. **No Tool Chaining** - Cannot use output of one tool as input to another
3. **Fixed Tool Set** - Tools defined at agent initialization
4. **No State Between Queries** - Each query is independent

## Future Enhancements

- Multi-level agent with tool chaining
- Conditional tool execution
- Tool output validation
- Custom tool parameters
- Agent reasoning tree visualization
- Token counting for cost optimization
