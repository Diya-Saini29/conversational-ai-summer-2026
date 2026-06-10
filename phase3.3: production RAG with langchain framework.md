# PHASE 3.3: LANGCHAIN FRAMEWORK
## Production-Grade RAG Systems with LangChain

---

## What is LangChain?

### Definition
LangChain is a framework that **abstracts away complexity** of building LLM applications.

```
Without LangChain (RAG from Scratch):
- Load document → manual code
- Split text → manual code
- Create embeddings → manual code
- Store in vector DB → manual code
- Retrieve → manual code
- Build prompt → manual code
- Call LLM → manual code
- Parse response → manual code

Total: ~200 lines of boilerplate code

With LangChain:
- Use DocumentLoader → 2 lines
- Use TextSplitter → 2 lines
- Use Embeddings → 2 lines
- Use VectorStore → 2 lines
- Use Retriever → 2 lines
- Use Chains → 2 lines
- Use LLM → 2 lines

Total: ~20 lines of clean code
```

### Core Philosophy
**"Build LLM applications by chaining reusable components"**

---

## LangChain Architecture

```
┌─────────────────────────────────────────────┐
│           LangChain Framework               │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────┐  ┌──────────────┐        │
│  │  Models     │  │  Prompt Tmpl │        │
│  ├─────────────┤  ├──────────────┤        │
│  │ OpenAI      │  │ Formatting   │        │
│  │ Claude      │  │ Variables    │        │
│  │ HuggingFace │  │ Few-shot     │        │
│  └─────────────┘  └──────────────┘        │
│       ↑                 ↑                   │
│       └────────┬────────┘                  │
│                │                            │
│          ┌─────▼─────┐                    │
│          │  Chains   │  (Orchestration)   │
│          └─────┬─────┘                    │
│                │                            │
│    ┌───────────┼───────────┐              │
│    │           │           │              │
│    ↓           ↓           ↓              │
│ ┌──────┐  ┌────────┐  ┌─────────┐       │
│ │Memory│  │Retriever│ │Tools    │       │
│ └──────┘  └────────┘  └─────────┘       │
│    │           │           │              │
│    └───────────┼───────────┘              │
│                │                            │
│          ┌─────▼──────┐                   │
│          │  Agents    │ (Reasoning)       │
│          └────────────┘                   │
│                                              │
└─────────────────────────────────────────────┘
```

---

## Core Components

### 1. LLMs (Language Models)
**What:** Interface to different language models

**Supported Models:**
```python
from langchain.llms import OpenAI, HuggingFace, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic

# OpenAI
llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

# Claude (Anthropic)
llm = ChatAnthropic(model="claude-2", temperature=0.5)

# HuggingFace (local)
llm = HuggingFace(model_name="gpt2")
```

**Key Difference: LLMs vs ChatModels**
```
LLM:            Takes text input → Returns text
                "What is AI?" → "AI is..."

ChatModel:      Takes messages → Returns message
                [{"role": "user", "content": "What is AI?"}] 
                → Message(content="AI is...")
```

### 2. Prompts & Prompt Templates
**What:** Structured way to build prompts

```python
from langchain.prompts import PromptTemplate

# Basic prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)

result = prompt.format(topic="Machine Learning")
# Output: "Explain Machine Learning in simple terms"
```

**Advanced: Few-Shot Examples**
```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "positive"},
    {"input": "sad", "output": "negative"}
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Word: {input}\nSentiment: {output}"
    ),
    suffix="Word: {word}\nSentiment:",
    input_variables=["word"]
)

# Gives examples before asking question
```

### 3. Chains
**What:** Sequence of calls to LLMs and other tools

**Simple Chain (Prompt → LLM)**
```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")
```

**RAG Chain**
```python
from langchain.chains import RetrievalQA

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # how to handle retrieved docs
    retriever=retriever   # your vector store retriever
)

answer = rag_chain.run("What is machine learning?")
```

**Chain Types (how to handle multiple documents):**
```
"stuff":       Put all docs in prompt (simple, but limited by context)
"map_reduce":  Summarize each doc, then summarize summaries
"refine":      Iteratively refine answer as you process docs
"map-rerank":  Rank docs by relevance, use top one
```

### 4. Document Loaders
**What:** Load documents from various sources

```python
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    DirectoryLoader,
    WebBaseLoader,
    ArxivLoader
)

# Load single file
loader = TextLoader("document.txt")
docs = loader.load()  # [Document(page_content="...", metadata={...})]

# Load entire directory
loader = DirectoryLoader(
    "documents/",
    glob="**/*.txt",
    loader_cls=TextLoader
)
docs = loader.load()

# Load from website
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# Load PDF
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()
```

### 5. Text Splitters
**What:** Break documents into chunks (we learned this in Phase 3.2)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
# Applies to each document
```

### 6. Embeddings
**What:** Convert text to vectors (we learned this in Phase 3.2)

```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed text
vector = embeddings.embed_query("Hello world")
# vector = [0.2, -0.5, 0.8, ...]
```

### 7. Vector Stores
**What:** Store and retrieve embeddings (we learned this in Phase 3.2)

```python
from langchain.vectorstores import Chroma, Pinecone, FAISS

# Create vector store and add documents
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Or load existing
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Search
results = vector_store.similarity_search("machine learning", k=3)
```

### 8. Retrievers
**What:** Interface to vector store for document retrieval

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Simple retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Return top 3
)

# Advanced: Compression retriever (filters irrelevant docs)
compressor = LLMChainExtractor.from_llm(llm, verbose=True)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 9. Memory
**What:** Keep conversation history for multi-turn interactions

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Simple memory (stores all messages)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

memory.save_context(
    {"input": "Hi"},
    {"output": "Hello! How can I help?"}
)

# Summary memory (summarizes old messages)
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history"
)
```

### 10. Agents
**What:** LLM that can decide to use tools autonomously

```python
from langchain.agents import AgentType, initialize_agent, Tool

tools = [
    Tool(
        name="Search",
        func=search_docs,
        description="Search documents"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Do math"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent decides which tool to use!
result = agent.run("What is 2+2 and explain AI")
# Agent: "I need calculator for math, and search for AI explanation"
```

---

## Complete RAG Application with LangChain

### Step-by-Step Implementation

```python
# 1. Setup
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# 2. Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 4. Create embeddings
embeddings = OpenAIEmbeddings()

# 5. Store in vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 6. Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 7. Initialize LLM
llm = OpenAI(temperature=0.3)

# 8. Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 9. Query!
result = rag_chain({"query": "What is machine learning?"})
print(result["result"])  # Answer
print(result["source_documents"])  # Sources used
```

---

## Comparison: RAG from Scratch vs LangChain

| Aspect | From Scratch | LangChain |
|---|---|---|
| **Lines of Code** | ~200 | ~30 |
| **Documentation** | Limited | Excellent |
| **Flexibility** | Maximum | Good (can customize) |
| **Learning Curve** | Steep | Gentle |
| **Production-Ready** | Manual optimization | Built-in best practices |
| **Error Handling** | Manual | Automatic |
| **Model Support** | Manual integration | 50+ pre-built |

---

## Memory Tricks - LangChain

| Component | Memory Trick |
|---|---|
| **LLM** | Language model interface (OpenAI, Claude, etc) |
| **Prompt Template** | Reusable prompt with variables |
| **Chain** | Sequence of operations (Prompt → LLM → Output) |
| **Loader** | Read documents from various sources |
| **Splitter** | Break documents into chunks |
| **Embeddings** | Convert text to vectors |
| **Vector Store** | Database for similarity search |
| **Retriever** | Fetch relevant documents |
| **Memory** | Remember conversation history |
| **Agent** | LLM that uses tools autonomously |

---

## One-Liners - LangChain

- **LangChain:** Framework for building LLM applications with reusable components
- **LLM:** Interface to language models (OpenAI, Claude, HuggingFace)
- **Prompt Template:** Reusable prompt structure with variable placeholders
- **Chain:** Orchestrates multiple steps (load → split → embed → retrieve → generate)
- **Retriever:** Fetches relevant documents from vector store for context
- **Agent:** Autonomous tool-using entity that decides what to do
- **Memory:** Stores conversation history for multi-turn interactions
- **Chain Type:** How to handle multiple retrieved documents (stuff, map-reduce, refine)

---

## Interview Questions - LangChain

### Q1: Why use LangChain instead of building from scratch?
**A:** "LangChain reduces boilerplate from ~200 to ~30 lines. It provides pre-built components for loaders, splitters, embeddings, vector stores, chains, and agents. This lets you focus on application logic rather than integration plumbing."

### Q2: What's the difference between LLM and ChatModel?
**A:** "LLM takes text input and returns text (legacy). ChatModel takes structured messages with roles and returns messages. ChatModel is preferred for conversation-like interfaces."

### Q3: Explain chain_type in RAG (stuff vs map_reduce vs refine)
**A:** 
- Stuff: Put all retrieved docs in prompt (fast, limited by context)
- Map-reduce: Summarize each doc independently, then summarize summaries (handles many docs)
- Refine: Iteratively improve answer as you process each doc (good for coherence)

### Q4: When would you use an Agent over a Chain?
**A:** "Use Chain when you know the exact steps. Use Agent when the LLM should decide dynamically. Example: Chain for RAG (retrieve → generate always). Agent for multi-tool interaction where LLM decides whether to search, calculate, or reason."

### Q5: How does LangChain handle memory?
**A:** "Two types: Buffer memory (stores all messages - simple but uses more tokens) and Summary memory (summarizes old messages to save tokens). Both keep conversation context for multi-turn interactions."

### Q6: How would you add custom tools to an Agent?
**A:** "Create Tool objects with name, func, and description. Pass them to initialize_agent. The agent reads descriptions and decides when to use each tool based on the query."

### Q7: What's the difference between retriever and vector_store?
**A:** "Vector store is storage + search. Retriever is an interface that can wrap vector store or use other strategies. You can have custom retrievers that combine multiple sources or apply compression."

---

## Common Mistakes with LangChain

```
❌ Using wrong chain type for your data size
   - "stuff" fails with 100+ documents

❌ Not setting appropriate search_kwargs
   - vector_store.as_retriever() needs k parameter

❌ Forgetting to persist vector store
   - persist_directory must be set to save

❌ Using high temperature for RAG
   - Should use 0.3-0.5 for factual answers

❌ Not including source documents in output
   - Always return_source_documents=True for citations

❌ Using generic embeddings for specialized domains
   - Consider fine-tuned embeddings for legal/medical text

❌ Creating vector store every time (inefficient)
   - Load existing if available, don't recreate
```

---

## Advanced LangChain Patterns

### Pattern 1: Custom Chain
```python
from langchain.schema import BaseMemory
from langchain.chains import LLMChain

class CustomChain:
    def __init__(self, llm, prompt, retriever):
        self.llm = llm
        self.prompt = prompt
        self.retriever = retriever
    
    def run(self, query):
        # Custom logic
        docs = self.retriever.get_relevant_documents(query)
        result = self.llm(self.prompt.format(
            query=query,
            context="\n".join([d.page_content for d in docs])
        ))
        return result
```

### Pattern 2: Multi-step RAG
```python
from langchain.chains import SequentialChain

# Step 1: Rewrite query
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

# Step 2: Retrieve
retriever = vector_store.as_retriever()

# Step 3: Generate answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Combine
full_chain = SequentialChain(
    chains=[rewrite_chain, qa_chain],
    verbose=True
)
```

### Pattern 3: Agent with Tools
```python
from langchain.agents import Tool, AgentType, initialize_agent

tools = [
    Tool(
        name="DocumentSearch",
        func=search_documents,
        description="Search company documents"
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="Search the internet"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Perform calculations"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

---

## Code Concepts for Implementation

When using LangChain, you'll typically:

1. **Choose your LLM**
   ```python
   llm = ChatOpenAI(model="gpt-4", temperature=0.3)
   ```

2. **Load documents**
   ```python
   loader = DirectoryLoader("docs/", glob="**/*.pdf")
   docs = loader.load()
   ```

3. **Process documents**
   ```python
   splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
   chunks = splitter.split_documents(docs)
   ```

4. **Create embeddings**
   ```python
   embeddings = OpenAIEmbeddings()
   ```

5. **Store in vector DB**
   ```python
   vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
   ```

6. **Create retriever**
   ```python
   retriever = vector_store.as_retriever(search_kwargs={"k": 3})
   ```

7. **Build RAG chain**
   ```python
   rag = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
   ```

8. **Query**
   ```python
   answer = rag.run("Your question here")
   ```

---

## Next Steps After LangChain

1. **Add LangSmith observability** (track performance)
2. **Implement evaluation** (measure quality)
3. **Add vector database persistence** (production)
4. **Wrap in API** (FastAPI) ← Phase 6
5. **Deploy** (Docker + Cloud) ← Phase 6

---

## Quick Reference

### Import Cheat Sheet
```python
# Models
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic

# Document handling
from langchain.document_loaders import TextLoader, PDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings & Storage
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, FAISS

# Chains & RAG
from langchain.chains import LLMChain, RetrievalQA, SequentialChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Agents
from langchain.agents import initialize_agent, Tool, AgentType

# Memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
```

---

**Master LangChain, and you can build production LLM applications in minutes!** 🚀
