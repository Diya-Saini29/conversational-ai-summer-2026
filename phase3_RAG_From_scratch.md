# PHASE 3.2: RAG FROM SCRATCH
## Retrieval Augmented Generation - Complete Notes

---

## The Problem: LLM Limitations

### Issue 1: Knowledge Cutoff
```
GPT-4 training data ends in April 2024
User asks in June 2024: "Who won the World Cup?"
LLM: "I don't know" or hallucination
```

### Issue 2: Hallucination
```
User: "What is the capital of Atlantis?"
LLM: "The capital of Atlantis is Poseidon City"
Reality: Atlantis is fictional!
LLM made up an answer (hallucination)
```

### Issue 3: Domain-Specific Knowledge
```
User: "What does our company policy say about remote work?"
LLM: Cannot answer - not in training data
Need: Access to company documents
```

### Solution: RAG (Retrieval Augmented Generation)

---

## What is RAG?

### Simple Definition
**RAG = Search your documents + Use results to generate answer**

```
User Question: "What is our remote work policy?"
     ↓
[1. Retrieve] Search company docs → Find relevant policy
     ↓
[2. Augment] Add retrieved docs to prompt
     ↓
[3. Generate] Pass augmented prompt to LLM → Get answer
     ↓
Answer: "Our remote work policy allows..."
```

### Why RAG Works
```
Without RAG:
User: "Company policy on remote work?"
LLM: *makes up answer* "We don't allow remote work"
Reality: Company allows 3 days/week remote

With RAG:
User: "Company policy on remote work?"
Retriever: *finds actual policy document*
Prompt: "Based on this document: [actual policy], answer: ..."
LLM: "Based on the policy, you can work remotely 3 days/week"
```

---

## RAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG SYSTEM                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Documents] ─→ [Split] ─→ [Embed] ─→ [Vector DB]    │
│                                            ↑             │
│                                            │             │
│                                         [Search]         │
│                                            ↑             │
│  [User Query] ──→ [Embed] ──────────────────┘          │
│       ↓                                                   │
│  [Retrieved Docs] ─→ [Augment Prompt] ─→ [LLM] ─→ Answer│
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role | Example |
|---|---|---|
| **Document** | Source of truth | PDF, text file, webpage |
| **Chunking** | Break into pieces | 512 tokens per chunk |
| **Embedding** | Convert to vectors | 768-dim vector |
| **Vector DB** | Store & search | Chroma, Pinecone |
| **Query** | User question | "What's the policy?" |
| **Retrieval** | Find relevant docs | Return top-3 chunks |
| **Augmentation** | Add context to prompt | Prompt + docs |
| **Generation** | LLM answers | "Based on docs..." |

---

# STEP 1: DOCUMENT LOADING

## What is a Document?
Any text source: PDF, website, book, notes, etc.

### Document Types & How to Load Them

#### Text Files (.txt)
```python
with open('document.txt', 'r') as f:
    content = f.read()
    # content = "This is the document text..."
```

#### PDFs
```python
from PyPDF2 import PdfReader
pdf = PdfReader('document.pdf')
text = ""
for page in pdf.pages:
    text += page.extract_text()
    # Extract all pages
```

#### Websites
```python
from urllib.request import urlopen
html = urlopen('https://example.com').read()
# Parse HTML to extract text
```

#### CSV/Structured Data
```python
import pandas as pd
df = pd.read_csv('data.csv')
# Convert rows to text format
```

### Key Concept: Document Metadata
```
Document = Content + Metadata

Example:
Content: "Machine learning is a subset of AI..."
Metadata: 
  - source: "ai_textbook.pdf"
  - page: 42
  - author: "John Smith"
  - date: "2024-01-15"

Why metadata matters?
- Can filter by source
- Track where answer came from
- Cite the document
```

---

# STEP 2: CHUNKING

## Problem: Documents Are Too Large
```
Document: "AI Textbook" (500 pages, 150,000 tokens)
LLM context window: 4,096 tokens
Cannot fit entire document!

Solution: Break into smaller chunks
```

## What is a Chunk?
A small piece of document that fits in LLM context.

```
Document:
"Machine learning is a subset of artificial intelligence.
It focuses on learning from data. AI is used in many applications.
Transformers are neural networks. Attention is key mechanism."

Chunks (500 chars each):
Chunk 1: "Machine learning is a subset of artificial intelligence.
         It focuses on learning from data."
         
Chunk 2: "AI is used in many applications. Transformers are 
         neural networks."
         
Chunk 3: "Attention is key mechanism."
```

## Chunking Strategies

### Strategy 1: Fixed Size Chunks
```
Break document every N characters/tokens

Size: 512 tokens per chunk
Overlap: 50 tokens (to maintain context)

Chunk 1: [tokens 0-512]
Chunk 2: [tokens 462-974]  (50 token overlap)
Chunk 3: [tokens 924-1436]
```

**Pros:** Simple, fast  
**Cons:** May split important context

### Strategy 2: Semantic Chunking
```
Break at logical boundaries (paragraphs, sentences)

Example:
Paragraph 1 → Chunk 1
Paragraph 2 → Chunk 2
Section break → New chunk

Benefits: Maintains semantic meaning
```

### Strategy 3: Recursive Chunking
```
Try to chunk by paragraph first
If paragraph > max_size, then chunk by sentence
If sentence > max_size, then chunk by character

Respects document structure while respecting size limit
```

## Recommended Approach
```python
# Use RecursiveCharacterTextSplitter
# Tries: paragraph → sentence → character
# Splits on: "\n\n" → "\n" → " " → ""

chunk_size = 512        # tokens
chunk_overlap = 50      # maintain context between chunks
```

## Chunking Parameters & Their Impact

| Parameter | Small Value | Large Value | Tradeoff |
|---|---|---|---|
| **Chunk Size** | More chunks, more retrieval queries | Fewer chunks, might miss context | Balance retrieval & context |
| **Overlap** | Less redundancy | More redundancy, ensures context | Usually 50-100 tokens |

### Example: Different Chunk Sizes
```
Document: 10,000 tokens

Chunk Size 256:
- Number of chunks: 40
- Retrieval: Fast, precise
- Risk: May split important context

Chunk Size 1024:
- Number of chunks: 10
- Retrieval: Slower search
- Benefit: Keeps context together
```

---

# STEP 3: EMBEDDINGS

## What is an Embedding?
A vector (list of numbers) that represents text meaning.

```
Text:      "Machine learning is great"
Embedding: [0.2, -0.5, 0.8, 0.1, ..., 0.3]  (768 dimensions)
           ↑    ↑    ↑   ↑         ↑
           dimension values that encode meaning
```

## Why Embeddings?
```
Reason 1: Vectors can be compared mathematically
         "Machine learning" and "ML" should have similar vectors
         
Reason 2: Find similar meaning without exact word match
         Query: "AI techniques"
         Document: "Deep learning methods" ← similar embedding!
         
Reason 3: Store in vector database for fast search
```

## Embedding Models

### Popular Models

| Model | Dimensions | Speed | Quality | Use Case |
|---|---|---|---|---|
| **Sentence-Transformers (all-MiniLM-L6-v2)** | 384 | Very fast | Good | RAG, semantic search |
| **OpenAI text-embedding-3-small** | 1536 | Fast | Excellent | Production RAG |
| **Cohere embed-english-v3.0** | 1024 | Fast | Excellent | Commercial use |
| **Local: BERT** | 768 | Medium | Good | Privacy-critical |

### How to Use Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed documents
doc_embedding = model.encode("Machine learning is great")
# Output: array of 384 numbers

# Embed query
query_embedding = model.encode("What is machine learning?")
# Output: array of 384 numbers

# Compare (similarity)
similarity = dot_product(doc_embedding, query_embedding)
# High similarity = related meaning!
```

## Embedding Properties

### Property 1: Semantic Similarity
```
"cat" embedding is closer to "dog" embedding
than to "car" embedding

Distance (cat, dog):    0.1 (close, similar)
Distance (cat, car):    0.8 (far, different)
```

### Property 2: Compositionality
```
"not good" embedding ≠ "good" embedding
"very good" embedding ≠ "good" embedding
Context matters!
```

### Property 3: Directionality
```
"machine learning" → "ML"   : same direction
"ML" → "machine learning"   : same direction
Symmetrical relationships preserved
```

---

# STEP 4: VECTOR DATABASE

## What is a Vector Database?
Specialized database optimized for storing and searching vectors.

```
Regular Database:
Store: "machine learning"
Search: Find exact match or keyword

Vector Database:
Store: [0.2, -0.5, 0.8, 0.1, ..., 0.3]
Search: Find most similar vectors (cosine similarity)
```

## Why Vector Database?
```
Without Vector DB:
- Store 1000 embeddings in memory
- Search takes O(n) = 1000 comparisons
- Slow for large documents

With Vector DB:
- Store 1000 embeddings indexed
- Search takes O(log n) or faster with indexing
- Fast similarity search with optimization
```

## Vector DB Comparison

| DB | Best For | Speed | Cost | Complexity |
|---|---|---|---|---|
| **Chroma** | Local development | Very fast | Free | Easy |
| **FAISS** | Large-scale local | Very fast | Free | Medium |
| **Pinecone** | Production cloud | Fast | Paid | Easy |
| **Weaviate** | Open-source cloud | Fast | Free/Paid | Medium |
| **Milvus** | Enterprise | Very fast | Free | Hard |

### Using Chroma (Easiest for Learning)
```python
import chromadb

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="my_docs")

# Add documents
collection.add(
    ids=["doc1", "doc2"],
    documents=["Machine learning is...", "Deep learning is..."],
    embeddings=[[0.2, -0.5, ...], [0.1, 0.3, ...]],
    metadatas=[
        {"source": "textbook.pdf"},
        {"source": "paper.pdf"}
    ]
)

# Search
results = collection.query(
    query_embeddings=[[0.2, -0.4, ...]],
    n_results=3
)
# Returns top 3 similar documents
```

## Vector Similarity Metrics

### Cosine Similarity (Most Common)
```
Measures angle between vectors
Range: -1 to 1
Higher = more similar

cos_sim(A, B) = (A · B) / (||A|| × ||B||)

Example:
A = [1, 0, 1]  (document)
B = [1, 0, 1]  (query)
cos_sim = 1.0  (identical!)

A = [1, 0, 1]
B = [0, 1, 0]
cos_sim = 0.0  (orthogonal, no similarity)
```

### Euclidean Distance
```
Measures straight-line distance between vectors
Range: 0 to ∞
Lower = more similar

dist(A, B) = √((a1-b1)² + (a2-b2)² + ... + (an-bn)²)
```

### Which to Use?
- **Cosine:** For text (direction matters more than magnitude)
- **Euclidean:** For numeric data (distance matters)
- **Most RAG systems use Cosine Similarity**

---

# STEP 5: RETRIEVAL

## What is Retrieval?
Finding the most relevant documents for a query.

### Retrieval Process
```
User Query: "What is machine learning?"
         ↓
Embed Query: [0.2, -0.5, 0.8, ...]
         ↓
Search Vector DB: Compare with all document embeddings
         ↓
Rank by Similarity: 
  - doc1: similarity 0.89
  - doc2: similarity 0.76
  - doc3: similarity 0.42
         ↓
Return Top-K: Return top 3 (or configurable number)
         ↓
Retrieved Docs: [doc1, doc2, doc3]
```

## Retrieval Parameters

### Parameter 1: K (Number of Results)
```
K = 3: Retrieve top 3 documents
K = 5: Retrieve top 5 documents
K = 10: Retrieve top 10 documents

Tradeoff:
- Small K: Faster, might miss relevant docs
- Large K: Slower, more context (might confuse LLM)
- Typical: K = 3 to 5
```

### Parameter 2: Similarity Threshold
```
Min Similarity: 0.7

Query: "machine learning"
Results:
- doc1: 0.89 ✓ (above threshold, include)
- doc2: 0.76 ✓ (above threshold, include)
- doc3: 0.42 ✗ (below threshold, exclude)

Benefit: Don't pass irrelevant docs to LLM
```

### Parameter 3: Reranking
```
Without Reranking:
1. Retrieve using embedding similarity (fast)
2. Pass top-K to LLM

With Reranking:
1. Retrieve top-K using embedding similarity (fast)
2. Use different model to rerank (more accurate)
3. Pass top-K reranked to LLM

Example: Retrieve 10, rerank to 3
```

## Advanced Retrieval Strategies

### Multi-Query Retrieval
```
Problem: Single query might miss relevant docs
Solution: Generate multiple query variations

Original Query: "What is machine learning?"

Generated Variations:
1. "What is ML and how does it work?"
2. "How does machine learning differ from AI?"
3. "Machine learning algorithms"

Retrieve with all 3 → Combine results → More coverage
```

### Hybrid Retrieval
```
Combine dense (embedding) + sparse (keyword) search

Step 1: Dense search - "find semantically similar"
        Returns: [doc1, doc2, doc3]

Step 2: Sparse search - "find keyword matches"
        Returns: [doc2, doc4, doc5]

Step 3: Merge results
        Final: [doc1, doc2, doc3, doc4, doc5]

Benefit: Get semantic + exact match results
```

---

# STEP 6: AUGMENTATION

## What is Augmentation?
Adding retrieved documents to the prompt.

### Prompt Structure

#### Without Augmentation
```
Prompt: "What is machine learning?"
```

#### With Augmentation
```
Prompt:
---
Context from documents:
[Retrieved Doc 1]: "Machine learning is a subset of AI..."
[Retrieved Doc 2]: "ML algorithms learn from data..."
[Retrieved Doc 3]: "Common ML techniques include..."

Question: "What is machine learning?"
---
```

### Prompt Template
```
SYSTEM: You are a helpful assistant.
Answer based on the provided context.

CONTEXT:
{retrieved_documents}

QUESTION:
{user_query}

ANSWER:
```

### Why Augmentation Order Matters
```
Lost in the Middle Effect:
When too many documents, LLM forgets information 
in the middle of context

Test: 10 documents, relevant info in middle
Result: LLM often ignores middle documents

Solution: Rerank by relevance, most relevant first
```

---

# STEP 7: GENERATION

## What is Generation?
Using LLM to answer based on augmented prompt.

### Generation Process
```
Augmented Prompt (with context)
         ↓
LLM Model (Claude, GPT-4, etc)
         ↓
Response Generation (token by token)
         ↓
Final Answer
```

### Prompt Engineering in RAG
```
Good Prompt:
"Based ONLY on the provided documents, answer: ..."
(Enforces grounding in context)

Bad Prompt:
"Answer this question"
(LLM might use training knowledge, hallucinate)

Good Prompt:
"If the answer is not in the documents, say 'I don't know'"
(Prevents hallucination)
```

### Temperature & Top-K in Generation
```
Temperature: Controls randomness
- 0.0: Deterministic (same answer every time)
- 0.7: Balanced (default)
- 1.0+: Creative (different answers each time)

For RAG: Use lower temperature (0.3-0.5)
Reason: Want factual, consistent answers
```

---

# STEP 8: COMPLETE RAG PIPELINE

## Code Flow
```python
# 1. LOAD DOCUMENTS
from pypdf import PdfReader
pdf = PdfReader('document.pdf')
text = ""
for page in pdf.pages:
    text += page.extract_text()

# 2. CHUNK
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
# chunks = ["Machine learning is...", "Deep learning is...", ...]

# 3. EMBED
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [model.encode(chunk) for chunk in chunks]

# 4. STORE IN VECTOR DB
import chromadb
client = chromadb.Client()
collection = client.create_collection(name="docs")
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

# 5. USER QUERY
query = "What is machine learning?"

# 6. RETRIEVE
query_embedding = model.encode(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
retrieved_docs = results['documents'][0]

# 7. AUGMENT PROMPT
augmented_prompt = f"""
Context: {' '.join(retrieved_docs)}
Question: {query}
Answer:
"""

# 8. GENERATE
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": augmented_prompt}]
)
answer = response.choices[0].message.content

print(answer)
```

## Full RAG Flow Diagram
```
┌──────────────┐
│  Documents   │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│    Chunking      │ (512 tokens/chunk)
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Embeddings      │ (384 dims)
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Vector Store    │ (Chroma, Pinecone)
└──────┬───────────┘
       │
       ├─────────────────────────┐
       │                         │
   [Query]                  [Documents]
       │                         │
       ↓                         ↓
   [Embed]              [Retrieve Top-K]
       │                         │
       └────────────┬────────────┘
                    ↓
           [Augment Prompt]
                    ↓
              [LLM (Claude)]
                    ↓
              [Final Answer]
```

---

# STEP 9: RAG EVALUATION

## Why Evaluate RAG?
```
Question: "What is machine learning?"

Bad RAG answer: "I don't know" (missing docs)
Good RAG answer: "Machine learning is a subset of AI..."

How to measure quality?
```

## Evaluation Metrics

### Metric 1: Retrieval Quality
```
Question: "What is the capital of France?"
Retrieved docs: [
  doc1: "Paris is a city in France" ✓ RELEVANT
  doc2: "France is in Europe" ✗ NOT RELEVANT
]

Precision: 1/2 = 50% (1 out of 2 retrieved are relevant)
Recall: 1/1 = 100% (found 1 out of 1 relevant docs available)
F1: 2/(1/0.5 + 1/1) = 66% (harmonic mean)

Interpretation:
- High precision: Most retrieved docs are relevant
- High recall: Found most relevant docs
- High F1: Good balance
```

### Metric 2: Answer Quality (Relevance)
```
Question: "What is machine learning?"

LLM Answer: "Machine learning is a type of artificial intelligence..."

Evaluator checks: Is the answer relevant to the question?
Score: 0-1 (or 0-5)
1.0 = Perfectly relevant
0.0 = Not relevant at all
```

### Metric 3: Faithfulness
```
Question: "What's our remote work policy?"

Retrieved doc: "Employees can work remotely 3 days/week"
LLM Answer: "Employees can work remotely 5 days/week"

Faithfulness: 0 (contradicts source!)

Should answer only use information from docs
```

### Metric 4: Hallucination Rate
```
Question: "What's the CEO's name?"
Retrieved doc: "CEO is John Smith"
LLM Answer: "CEO is Jane Doe"

Hallucination detected: LLM invented false information
Lower hallucination rate = better
```

## How to Evaluate (Simple Approach)
```
1. Create test set:
   Q1: "What is X?"
   Q2: "How does Y work?"
   ...
   Q10: "Define Z"

2. Run through RAG system

3. Manual evaluation:
   - Is answer correct? Yes/No
   - Is answer based on docs? Yes/No
   - Any hallucinations? Yes/No

4. Calculate metrics:
   Accuracy = correct answers / total questions
```

---

## Memory Tricks - RAG Fundamentals

| Concept | Memory Trick |
|---|---|
| **RAG** | Retrieve + Augment + Generate (3 steps) |
| **Chunking** | Break docs into bite-sized pieces |
| **Embedding** | Numbers that represent meaning |
| **Vector DB** | Fast similarity search warehouse |
| **Retrieval** | Find most relevant chunks |
| **Augmentation** | Add context to prompt |
| **Generation** | LLM answers with context |
| **Cosine Similarity** | Angle between vectors (0 to 1 range) |
| **K parameter** | How many results to retrieve |

---

## One-Liners - RAG System

- **RAG:** Retrieve relevant docs, add to prompt, generate answer
- **Chunking:** Breaking documents into manageable pieces (typically 512 tokens)
- **Embedding:** Converting text into vectors that capture meaning
- **Vector Database:** Optimized storage for fast similarity search
- **Retrieval:** Finding most relevant documents using vector similarity
- **Augmentation:** Adding retrieved docs to LLM prompt for context
- **Generation:** LLM generates answer based on augmented prompt
- **Cosine Similarity:** Measures angle between vectors (0=different, 1=same)
- **Chunk Size:** Balance between context preservation and number of chunks

---

## Interview Questions - RAG

### Q1: Explain RAG in one sentence
**A:** "RAG retrieves relevant documents from a knowledge base, adds them to the LLM prompt for context, and generates an answer grounded in those documents."

### Q2: Why chunk documents?
**A:** "LLM context windows are limited. Chunking breaks documents into pieces that fit in context (usually 512 tokens) while maintaining semantic meaning."

### Q3: What's the tradeoff between chunk size?
**A:** "Smaller chunks = more retrieval precision but more chunks to search. Larger chunks = fewer to search but might lose context. Typical: 256-1024 tokens."

### Q4: How does embedding solve the matching problem?
**A:** "Embeddings convert text to vectors where semantically similar words have similar vectors. This allows finding relevant docs without exact keyword matching."

### Q5: Why use vector database instead of searching all docs?
**A:** "Vector databases use indexing and optimization algorithms to search efficiently. Searching all docs is O(n). Vector DB with indexing is O(log n) or better."

### Q6: What's cosine similarity and why use it for text?
**A:** "Cosine similarity measures the angle between vectors (0-1). We use it for text because direction of the vector matters more than magnitude - two sentences can be identical in meaning but have different lengths."

### Q7: What causes RAG to fail?
**A:** "Three main causes: 1) Bad retrieval - relevant docs not found, 2) Noisy context - too many irrelevant docs confuse LLM, 3) LLM hallucination - ignores docs and makes up answer."

### Q8: How would you evaluate RAG quality?
**A:** "Three metrics: 1) Retrieval precision/recall - are retrieved docs relevant? 2) Answer relevance - does answer match question? 3) Faithfulness - does answer stick to retrieved docs?"

---

## Code Concepts for Phase 3.2

When implementing RAG from scratch, you'll code:

1. **Document Loading**
   ```python
   # Read PDF, TXT, or other document formats
   def load_document(file_path):
       # Load and return text
   ```

2. **Chunking**
   ```python
   def chunk_text(text, chunk_size=512, overlap=50):
       # Split text into overlapping chunks
       return chunks
   ```

3. **Embedding**
   ```python
   def embed_text(text):
       # Convert text to vector
       return embedding_vector
   ```

4. **Vector Store**
   ```python
   def store_embeddings(chunks, embeddings):
       # Store in vector database
       # Support similarity search
   ```

5. **Retrieval**
   ```python
   def retrieve(query, k=3):
       # Embed query
       # Find top-k similar chunks
       return retrieved_chunks
   ```

6. **Augmentation**
   ```python
   def augment_prompt(query, context):
       # Build prompt with context
       return augmented_prompt
   ```

7. **Generation**
   ```python
   def generate_answer(prompt, llm_model):
       # Call LLM with augmented prompt
       return answer
   ```

---

## Next Steps

After Phase 3.2 (RAG from Scratch):
1. You understand RAG end-to-end (manual implementation)
2. You've built a working system without frameworks
3. Ready for Phase 3.3: Use LangChain for production RAG
4. LangChain will automate and optimize what you manually coded

---

## Quick Reference

### RAG Pipeline Summary
```
Document → Chunk → Embed → Store → Retrieve → Augment → Generate → Answer
```

### Key Numbers
```
Chunk size:     512 tokens (typical)
Chunk overlap:  50 tokens
Embedding dim:  384-1536 (depends on model)
K (top results): 3-5 (usually)
Temperature:    0.3-0.5 (for factual answers)
```

### Common Mistakes
```
❌ Too large chunks: Context gets confused
❌ No overlap: Context lost between chunks
❌ Too small K: Miss relevant documents
❌ High temperature: Hallucinations increase
❌ Bad retrieval: Irrelevant docs break answer
```

---

**Master this, and you understand how LLMs access external knowledge!** 🚀
