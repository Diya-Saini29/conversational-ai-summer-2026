# PHASE 5.1: LANGGRAPH - CONVERSATIONAL AI SYSTEMS
## Building Production-Grade Agentic AI with LangGraph

---

## What is LangGraph?

### Definition
LangGraph is a framework for building **stateful, multi-actor applications** using LLMs as nodes in a graph structure.

```
LangChain:  Linear chains (A → B → C)
LangGraph:  Complex flows with decisions, loops, parallel paths
            (A → {B or C} → {loop back or end})
```

### Why LangGraph Over LangChain?

| Aspect | LangChain | LangGraph |
|---|---|---|
| **Control Flow** | Sequential chains | Conditional, parallel, looping |
| **State Management** | Limited | Full state control |
| **Agent Logic** | Basic agents | Complex multi-step reasoning |
| **Conversational** | Stateless | Full conversation state |
| **Memory** | Manual | Built-in persistence |
| **Debugging** | Hard to trace | Visual graph + tracing |

---

## Core Concept: Graph-Based Execution

### What is a Graph?

```
Traditional Code:
input → function1() → function2() → function3() → output

LangGraph (Graph):
        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  Retrieve   │
        │  Documents │
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │  Conditional Check  │
        │  (Relevant found?)  │
        └──┬──────────────┬───┘
    YES │               │ NO
        │               │
    ┌───▼──┐       ┌────▼────┐
    │Generate│      │Web Search│
    │Answer  │      │          │
    └───┬────┘      └────┬─────┘
        │                │
        └────────┬───────┘
                 │
          ┌──────▼──────┐
          │   END       │
          └─────────────┘
```

**Key Components:**
- **Nodes:** Functions/LLMs that do work
- **Edges:** Connections between nodes (control flow)
- **State:** Data passed between nodes
- **Conditional Edges:** Routes based on conditions

---

## Building Block 1: State Management

### What is State?

State = All information needed to make decisions in your agent.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Input
    messages: Annotated[list, add_messages]
    
    # Context
    documents: list
    current_query: str
    
    # Decision variables
    needs_web_search: bool
    iteration_count: int
    
    # Output
    final_answer: str
```

### State Flow Through Graph

```
User Input: "What is machine learning?"
     ↓
[State created: messages=["What is..."], documents=[], iteration_count=0]
     ↓
[Node 1: Retrieve] → State: documents=[doc1, doc2, doc3]
     ↓
[Node 2: Check] → State: needs_web_search=False (docs found)
     ↓
[Node 3: Generate] → State: final_answer="Machine learning is..."
     ↓
Output
```

### Key Concept: Immutable State Updates

```python
# LangGraph state updates are functional (immutable)
def retrieve_node(state):
    docs = retrieve_documents(state["messages"][-1].content)
    # Return updates (don't modify state directly)
    return {
        "documents": docs,  # Add to state
        "iteration_count": state["iteration_count"] + 1
    }
```

---

## Building Block 2: Nodes

### What is a Node?

A node is a function that:
1. Takes state as input
2. Does some work (LLM call, retrieval, etc)
3. Returns updated state

```python
from langchain_core.messages import AIMessage

def llm_node(state):
    """Node that calls LLM"""
    messages = state["messages"]
    documents = state["documents"]
    
    # Build prompt with context
    prompt = f"""Based on these documents:
    {documents}
    
    Answer: {messages[-1].content}"""
    
    # Call LLM
    response = llm.invoke(prompt)
    
    # Update state
    return {
        "messages": [AIMessage(content=response)],
        "final_answer": response
    }

def retrieve_node(state):
    """Node that retrieves documents"""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    
    return {
        "documents": docs
    }
```

### Types of Nodes

| Node Type | Purpose | Example |
|---|---|---|
| **Retrieval** | Fetch relevant data | Vector DB search |
| **LLM** | Call language model | Generate response |
| **Tool** | Execute tools | Calculator, API calls |
| **Condition** | Make decisions | Check relevance |
| **Processing** | Data transformation | Parse, format |

---

## Building Block 3: Edges & Conditional Edges

### Simple Edge (A → B)

```python
graph.add_edge("retrieve_node", "llm_node")
# Always go from retrieve to LLM
```

### Conditional Edge (A → {B or C})

```python
def route_decision(state):
    """Decide which node to go to next"""
    documents = state["documents"]
    
    if documents:  # Found relevant docs
        return "generate_answer"
    else:  # No docs found
        return "web_search"

# Add conditional edge
graph.add_conditional_edges(
    "retrieve_node",
    route_decision,
    {
        "generate_answer": "llm_node",
        "web_search": "web_search_node"
    }
)
```

### Loop Back (For Iteration)

```python
def check_answer_quality(state):
    """Decide if answer needs refinement"""
    answer = state["final_answer"]
    iteration = state["iteration_count"]
    
    if iteration < 3 and answer_needs_improvement(answer):
        return "refine"  # Loop back
    else:
        return "end"  # Exit

graph.add_conditional_edges(
    "generate_node",
    check_answer_quality,
    {
        "refine": "retrieve_node",  # Loop back
        "end": END
    }
)
```

---

## Building Block 4: Conditional Workflows

### What Are Conditional Workflows?

Agents that make decisions based on state.

```
User: "What's the weather AND stock price?"
     ↓
[Check]: Need 2 tools?
     ├─→ YES → [Weather Tool] + [Stock Tool] (parallel)
     └─→ NO → [Single Tool]
     ↓
[Combine results]
     ↓
Answer
```

### Example: Smart RAG Agent

```python
def route_query(state):
    query = state["messages"][-1].content
    
    if "search" in query.lower():
        return "web_search"
    elif "calculate" in query.lower():
        return "calculator"
    else:
        return "retrieve_docs"

# Add routing
graph.add_conditional_edges(
    "start",
    route_query,
    {
        "web_search": "web_search_node",
        "calculator": "calculator_node",
        "retrieve_docs": "retrieve_node"
    }
)
```

---

## Building Block 5: Iterative Workflows

### What Are Iterative Workflows?

Agents that refine answers through multiple iterations.

```
Attempt 1: Generate answer
     ↓
[Check]: Good enough?
     ├─→ YES → Return
     └─→ NO → Refine

Attempt 2: Refine based on feedback
     ↓
[Check]: Good enough?
     ├─→ YES → Return
     └─→ NO → Refine again
```

### Example: Multi-Pass Agent

```python
def should_refine(state):
    """Decide if answer needs refinement"""
    answer = state.get("answer", "")
    attempts = state.get("attempts", 0)
    
    # Stop after 3 attempts or if satisfied
    if attempts >= 3:
        return "end"
    
    if is_answer_sufficient(answer):
        return "end"
    else:
        return "refine"

def refine_node(state):
    """Improve the answer"""
    previous_answer = state["answer"]
    query = state["messages"][-1].content
    
    refine_prompt = f"""
    Previous answer: {previous_answer}
    Original question: {query}
    
    Provide a better, more complete answer.
    """
    
    refined = llm.invoke(refine_prompt)
    
    return {
        "answer": refined,
        "attempts": state.get("attempts", 0) + 1
    }
```

---

## Building Block 6: Multi-Turn Conversations

### Problem: Stateless Chatbots

```
Turn 1: User: "I'm interested in AI"
        Bot: "AI is fascinating!"
        
Turn 2: User: "Tell me more"
        Bot: "More about what? (Lost context!)
```

### Solution: Message History in State

```python
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]  # ALL messages
    documents: list
    final_answer: str

# add_messages function automatically appends new messages
# Maintains full conversation history

def conversation_node(state):
    all_messages = state["messages"]  # Full history
    # Use full history for context-aware responses
    response = llm.invoke(all_messages)
    
    return {
        "messages": [AIMessage(content=response)]
    }
```

---

## Building Block 7: Tools in Agents

### What Are Tools?

Functions that agents can call to gather information or take action.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_documents(query: str) -> str:
    """Search internal documents"""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression"""
    return str(eval(expression))

@tool
def web_search(query: str) -> str:
    """Search the web"""
    results = search_api.search(query)
    return results

tools = [search_documents, calculate, web_search]
```

### Agent With Tools

```python
from langgraph.prebuilt import create_react_agent

# Create agent that decides which tool to use
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant. Use tools to help answer."
)

# Agent automatically:
# 1. Decides which tool to use
# 2. Calls the tool
# 3. Processes result
# 4. Generates response
```

### Tool Loop

```
User: "Calculate 2+2 and search for AI definition"
     ↓
[Agent thinks]: Need calculator AND search_documents
     ↓
[Tool Call 1]: calculate("2+2") → "4"
[Tool Call 2]: search_documents("AI definition") → "AI is..."
     ↓
[Agent combines]: "2+2 = 4. AI is..."
     ↓
Response
```

---

## Building Block 8: Memory - The Heart of Conversational AI

### Problem: Why LLMs Don't Have Memory

```python
# LLMs are stateless functions
def llm(prompt):
    return response  # Only uses this one prompt

# Each call is independent
response1 = llm("Hi, I'm Alice")  # Doesn't remember Alice
response2 = llm("What's my name?")  # Can't answer (no memory!)
```

### Solution: Explicit Memory Management

```
Turn 1: User: "Hi, I'm Alice"
        System: Store in memory: {"name": "Alice"}
        Response: "Nice to meet you, Alice!"

Turn 2: User: "What's my name?"
        System: Retrieve from memory: name="Alice"
        Response: "Your name is Alice!"
```

---

## Building Block 9: Short-Term Memory (Within Conversation)

### What is Short-Term Memory?

Information relevant to the current conversation session.

```
Session 1:
- User topic: Machine Learning
- Preferences: Code examples
- Questions asked: 3
- Items discussed: [Transformers, CNNs, RNNs]

(Session ends)

Session 2: New session, short-term memory cleared
```

### Implementation

```python
from langchain.memory import ConversationBufferMemory

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]  # Short-term memory
    user_preferences: dict
    conversation_context: str

def remember_context(state):
    """Update short-term memory based on conversation"""
    messages = state["messages"]
    
    # Extract preferences from recent messages
    recent = messages[-5:]  # Last 5 messages
    preferences = extract_preferences(recent)
    
    return {
        "user_preferences": preferences,
        "conversation_context": summarize_context(recent)
    }
```

### What to Store

```
Short-term memory includes:
✅ Current topic
✅ User preferences stated in this session
✅ Previous questions in this session
✅ Decisions made in this session

NOT stored:
❌ Personal info (first name, email)
❌ Preferences from past sessions
❌ Historical data
```

---

## Building Block 10: Long-Term Memory (Across Sessions)

### What is Long-Term Memory?

Information that persists across multiple sessions.

```
Session 1 (Monday):
- User: "I work in healthcare"
- System: Store: {"industry": "healthcare"}

Session 2 (Wednesday):
- User: "Give me healthcare AI trends"
- System: Retrieves: industry="healthcare"
- Response: Uses healthcare context (remembered!)
```

### Implementation

```python
import json
from datetime import datetime

class LongTermMemory:
    def __init__(self, storage_file="memory.json"):
        self.storage = storage_file
        self.data = self.load()
    
    def store_memory(self, user_id: str, key: str, value):
        """Store information for next session"""
        if user_id not in self.data:
            self.data[user_id] = {}
        
        self.data[user_id][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def retrieve_memory(self, user_id: str, key: str):
        """Retrieve information from previous sessions"""
        if user_id in self.data and key in self.data[user_id]:
            return self.data[user_id][key]["value"]
        return None
    
    def load(self):
        try:
            with open(self.storage, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save(self):
        with open(self.storage, 'w') as f:
            json.dump(self.data, f)

# Usage
memory = LongTermMemory()

# In conversation
def use_long_term_memory(state):
    user_id = state.get("user_id")
    
    # Retrieve previous preferences
    industry = memory.retrieve_memory(user_id, "industry")
    
    if industry:
        # Use industry context
        prompt = f"Tailor response for {industry} industry"
    
    # After response, store new info
    if extracted_industry := extract_industry(state["messages"]):
        memory.store_memory(user_id, "industry", extracted_industry)
    
    return state
```

### What to Store

```
Long-term memory includes:
✅ User profile (industry, role, goals)
✅ Preferences learned over time
✅ Topics discussed before
✅ Important facts about user

NOT stored:
❌ Sensitive data (passwords, cards)
❌ Temporary preferences
❌ Real-time data
```

---

## Building Block 11: Persistence

### What is Persistence?

Saving agent state to disk so it survives app restart.

```
Turn 1 (Monday 9am):
User: "Analyze this dataset"
Agent: Does analysis, saves state to database

Turn 2 (Monday 3pm - after app restart):
User: "Continue from where we left"
Agent: Loads saved state, continues (NO lost context!)
```

### Implementation

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

# Create checkpointer
checkpointer = SqliteSaver.from_conn_string(":memory:")

# Build graph with checkpointing
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Add checkpointing
graph = graph_builder.compile(checkpointer=checkpointer)

# Run with persistence
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke(state, config=config)

# Next time with same thread_id, state is loaded
result2 = graph.invoke(new_input, config=config)  # State preserved!
```

### Benefits

```
✅ Multi-session conversations
✅ Resume interrupted tasks
✅ User isolation (thread_id per user)
✅ Debugging (replay conversations)
✅ Analytics (track conversation history)
```

---

## Building Block 12: Streamlit UI

### Why Streamlit?

Simple Python web UI for chatbot demos.

```python
import streamlit as st
from langgraph.graph import StateGraph

st.title("🤖 Conversational AI Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Ask anything..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    
    # Run agent
    state = {"messages": st.session_state.messages}
    result = st.session_state.graph.invoke(state)
    
    # Add response
    st.session_state.messages.append(
        {"role": "assistant", "content": result["final_answer"]}
    )
    
    st.rerun()
```

### Run It

```bash
streamlit run app.py
# Opens http://localhost:8501
```

---

## Building Block 13: RAG with LangGraph

### RAG Agent Architecture

```
User Query
    ↓
[Retrieve Documents]
    ↓
[Check Relevance]
    ├→ Relevant found
    │    ↓
    │   [Generate with context]
    │    ↓
    │   Answer
    │
    └→ Not relevant
         ↓
        [Web search]
         ↓
        [Generate]
         ↓
        Answer
```

### Implementation

```python
def retrieve_node(state):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"documents": docs}

def check_relevance(state):
    """Decide if retrieved docs are relevant"""
    if state["documents"]:
        return "generate"
    else:
        return "web_search"

def generate_node(state):
    """Generate answer with retrieved docs"""
    docs = state["documents"]
    query = state["messages"][-1].content
    
    prompt = f"""Based on these documents:
    {docs}
    
    Answer the question: {query}"""
    
    response = llm.invoke(prompt)
    return {"final_answer": response}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("web_search", web_search_node)

graph.add_edge("retrieve", "check_relevance")
graph.add_conditional_edges(
    "retrieve",
    check_relevance,
    {"generate": "generate", "web_search": "web_search"}
)
graph.add_edge("generate", END)
graph.add_edge("web_search", "generate")

compiled_graph = graph.compile()
```

---

## Complete Conversational AI System

### Full Architecture

```
                    ┌──────────────────┐
                    │   USER INPUT     │
                    └────────┬─────────┘
                             │
                    ┌────────▼────────┐
                    │ Load Long-term  │
                    │ Memory          │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Check Message   │
                    │ Type (Question, │
                    │ Command, etc)   │
                    └────┬────────┬───┘
            ┌─────────────┤        └──────────────┐
            │             │                       │
        ┌───▼─────┐  ┌───▼──────┐         ┌──────▼────┐
        │Retrieve │  │Calculator│         │Tool Call  │
        │from RAG │  │Tool      │         │Handler    │
        └───┬─────┘  └───┬──────┘         └──────┬────┘
            │             │                       │
            └─────────────┼───────────────────────┘
                          │
                  ┌───────▼────────┐
                  │Update Short-   │
                  │term Memory     │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │Generate        │
                  │Response        │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │Update Long-    │
                  │term Memory     │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │Save State      │
                  │(Persistence)   │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │Return Response │
                  └────────────────┘
```

---

## Memory Tricks - LangGraph

| Concept | Trick |
|---|---|
| **Node** | Function that does work + returns state updates |
| **Edge** | Connection between nodes (control flow) |
| **Conditional Edge** | Route based on state (decision point) |
| **State** | All information shared between nodes |
| **Short-term Memory** | What user said in THIS session |
| **Long-term Memory** | Who user IS (persistent across sessions) |
| **Persistence** | Save state to database (survive restarts) |
| **Tools** | Functions agents can call to gather info |
| **Iterative** | Loop until satisfied (multi-pass refinement) |

---

## One-Liners - LangGraph

- **LangGraph:** Graph-based framework for building stateful, multi-step AI agents
- **Node:** Function in the graph that performs work and updates state
- **Edge:** Directed connection showing flow between nodes
- **Conditional Edge:** Routes to different nodes based on state conditions
- **State:** Shared data structure passed between all nodes
- **Short-term Memory:** Conversation history within single session
- **Long-term Memory:** User/context information persisting across sessions
- **Persistence:** Saving agent state to database for resuming conversations
- **Tools:** External functions agents can invoke (search, calculate, API calls)
- **Iterative:** Agent loops to refine answer until satisfied

---

## Interview Questions - LangGraph

### Q1: How does LangGraph differ from LangChain?
**A:** "LangChain handles linear chains (A→B→C). LangGraph handles complex control flow: conditional paths, loops, parallel execution, and full state management. LangGraph is better for agents."

### Q2: What is state in LangGraph?
**A:** "State is a TypedDict that holds all data shared between nodes. Each node receives state, does work, and returns updates. State persists across the entire graph execution."

### Q3: How do you implement conditional routing?
**A:** "Use `add_conditional_edges()`. Provide a function that examines state and returns which node to route to. Example: if documents found, go to 'generate', else go to 'web_search'."

### Q4: What's the difference between short-term and long-term memory?
**A:** "Short-term memory is the current conversation (message history). Long-term memory is user profile/preferences that persist across sessions. Both essential for real conversational AI."

### Q5: Why is persistence important?
**A:** "Without persistence, if app restarts, user loses context. With persistence (saved to database), agent can resume conversation from exact state. Critical for production systems."

### Q6: How would you build a multi-turn agent?
**A:** "Use message history in state with add_messages. Each turn, append new message to history. Agent can access full history for context. Persistence ensures history survives app restart."

### Q7: What problem do tools solve?
**A:** "Tools let agents take action beyond just generating text. Agent can decide which tool to use (calculator, search, API call) to gather information or complete tasks."

### Q8: Explain iterative workflows.
**A:** "Agent makes first attempt, checks quality, loops back if not satisfied. Example: Generate answer → Check relevance → If poor, retrieve more docs → Refine → Loop again until good."

---

## Common Patterns

### Pattern 1: Simple Q&A with Retrieval
```
User Query → Retrieve Docs → Generate Answer → Response
```

### Pattern 2: Conditional Tool Routing
```
Query → Check Type → Route to Tool → Execute → Generate Response
```

### Pattern 3: Iterative Refinement
```
Generate → Check Quality → If Bad → Refine → Loop
```

### Pattern 4: Multi-turn with Memory
```
Load Long-term Memory → Process Query → Generate → Save Long-term Memory
```

### Pattern 5: RAG Agent with Fallback
```
Retrieve → Check Relevance → If Good: Generate | If Bad: Web Search → Generate
```

---



**Ready to build your Conversational AI Capstone!** 🚀

---
