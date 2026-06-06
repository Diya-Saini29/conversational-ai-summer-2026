# PHASE 2: UCS664 NATURAL LANGUAGE PROCESSING
## Complete Notes for NLP & Transformers

---

# PHASE 2.1: NLP FUNDAMENTALS
## Video: Krish Naik - "Complete NLP Machine Learning In One Shot"

---

## What is Natural Language Processing (NLP)?

### Definition
NLP is the field of AI that helps computers understand, interpret, and generate human language (text and speech).

### Real-World Examples
| Application | How NLP Works |
|---|---|
| **Gmail spam filter** | Reads email text → classifies as spam/not spam |
| **Google Translate** | Reads English → understands meaning → generates Spanish |
| **Chatbots** | Reads user message → understands intent → generates response |
| **Sentiment analysis** | Reads review → determines if positive/negative |
| **Named entity recognition** | Reads "Apple CEO Tim Cook" → identifies PERSON, COMPANY, POSITION |

### Why is NLP Hard?
- **Ambiguity:** "Bank" = financial institution OR river bank?
- **Context matters:** "I read a book yesterday" vs "I red a red car"
- **Slang & typos:** "ur", "lol", "gonna"
- **Multiple meanings:** Same word, different context

---

## Step 1: Text Preprocessing
### Why Preprocess?
Raw text is messy. Preprocessing cleans it so the model can learn better.

### Preprocessing Steps

#### 1.1 Lowercasing
```
Input:  "Apple is a GREAT company!"
Output: "apple is a great company!"
Why:    Treats "Apple" and "apple" as same word
```

#### 1.2 Removing Punctuation
```
Input:  "Hello, how are you? I'm fine!"
Output: "Hello how are you I m fine"
Why:    Punctuation doesn't add meaning for most tasks
```

#### 1.3 Removing Special Characters & Numbers
```
Input:  "Email: user123@gmail.com, Price: $99.99"
Output: "Email user gmail Price"
Why:    Reduces noise, focuses on meaningful words
```

#### 1.4 Whitespace Handling
```
Input:  "Hello    world"
Output: "Hello world"
Why:    Multiple spaces are unnecessary
```

---

## Step 2: Tokenization
### What is Tokenization?
Breaking text into smaller pieces (tokens) that a model can understand.

### Types of Tokenization

#### Word Tokenization
```
Input:  "I love machine learning"
Output: ["I", "love", "machine", "learning"]

Input:  "It's a beautiful day!"
Output: ["It", "'s", "a", "beautiful", "day", "!"]
```

#### Sentence Tokenization
```
Input:  "Hello. How are you? I'm fine."
Output: ["Hello.", "How are you?", "I'm fine."]
```

#### Character Tokenization
```
Input:  "Hello"
Output: ["H", "e", "l", "l", "o"]
Why:    Useful for languages without clear word boundaries (Chinese, Japanese)
```

### Key Concept
- **Vocabulary:** Set of all unique words in your dataset
- **Vocab size:** If you have 10,000 unique words → vocab size = 10,000
- **Out of Vocabulary (OOV):** Words not in vocabulary → marked as `<UNK>` (unknown)

---

## Step 3: Stopword Removal
### What are Stopwords?
Common words that appear frequently but add little meaning.

### Examples
```
Common stopwords: a, an, the, is, are, and, or, but, in, on, at, to, for, of, with
```

### Decision: Remove or Keep?
| Scenario | Decision | Example |
|---|---|---|
| **Sentiment Analysis** | Remove | "not good" → if you remove "not", meaning changes! Consider keeping. |
| **Spam Detection** | Remove | Email filter doesn't need "the", "a" |
| **Machine Translation** | Keep | "The book" has structure that matters in target language |
| **Chatbot Intent** | Keep | "What is AI?" - "is" helps identify question type |

### Code Concept
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# stopwords: {'a', 'an', 'the', 'and', 'or', ...}
```

---

## Step 4: Stemming vs Lemmatization
### Problem They Solve
```
Different forms of same word:
- run, running, runs
- happy, happier, happily
- connection, connecting, connected

Without stemming/lemmatization:
→ Model treats each as different word
→ Wastes vocabulary space
→ Doesn't capture that they're related

With stemming/lemmatization:
→ Reduces to root word
→ Captures semantic relationship
```

### Stemming
**What:** Chops off word endings using rules.

**Example:**
```
running  → runn (remove "ing")
happiness → happi (remove "ness")
connection → connect (remove "ion")
```

**Algorithms:**
- **Porter Stemmer** (most common)
- **Snowball** (improved version)

**Pros & Cons:**
| Pros | Cons |
|---|---|
| Fast | Crude (sometimes wrong) |
| Works well in practice | Can create non-words ("happi") |
| Reduces vocabulary | Loses some meaning |

### Lemmatization
**What:** Uses dictionary + grammar to find root word.

**Example:**
```
running → run (dictionary: "running" is verb form of "run")
better → good (dictionary: "better" is comparative of "good")
am, are, is → be (all forms of "be")
```

**Pros & Cons:**
| Pros | Cons |
|---|---|
| Accurate (real words) | Slower (uses dictionary lookup) |
| Preserves meaning better | Requires language dictionary |
| Handles irregular words | More complex |

### Stemming vs Lemmatization Comparison
```
Word              Stemming    Lemmatization
running           runn        run
happily           happi       happy
better            better      good
am/are/is         am/are/is   be
```

### When to Use What?
| Task | Choice | Why |
|---|---|---|
| Search engines | Stemming | Speed matters, exactness less |
| Chatbots | Lemmatization | Understanding meaning is critical |
| Information retrieval | Stemming | Fast, good enough |
| Machine translation | Lemmatization | Grammar matters for accuracy |

---

## Step 5: Feature Extraction - Turning Text Into Numbers

### Why Numbers?
**Machine learning models work with numbers, not text.**

```
Model input:  must be numbers
Text input:   "I love AI"
Feature extraction: "I love AI" → [0, 1, 0.5, 0.8, ...]
Model output: [0.9] (probability)
```

### Method 1: Bag of Words (BoW)

**Concept:** Count how many times each word appears.

**Example:**
```
Vocabulary: ["I", "love", "machine", "learning", "AI"]

Sentence 1: "I love AI"
BoW:        [1, 1, 0, 0, 1]
             (I appears 1x, love 1x, machine 0x, learning 0x, AI 1x)

Sentence 2: "I love machine learning"
BoW:        [1, 1, 1, 1, 0]
```

**Pros:** Simple, interpretable  
**Cons:** Loses word order ("dog bites man" vs "man bites dog" = same BoW)

### Method 2: TF-IDF (Term Frequency - Inverse Document Frequency)

**Problem it solves:**
- Bag of Words treats all words equally
- "the" appears in every document but isn't important
- "AI" appears in few documents but is very important

**Solution:** TF-IDF gives higher score to rare but important words.

**Formula:**
```
TF-IDF = TF × IDF

TF (Term Frequency):     How often word appears in document
                         TF = (count of word) / (total words in document)

IDF (Inverse Doc Freq):  How rare the word is across all documents
                         IDF = log(total documents / documents with word)

Interpretation:
- High TF-IDF  = word is frequent in doc AND rare across corpus (IMPORTANT)
- Low TF-IDF   = word is rare in doc OR common across corpus (UNIMPORTANT)
```

**Example:**
```
Document 1: "machine learning is powerful"
Document 2: "machine learning is exciting"
Document 3: "deep learning is complex"

Word: "learning"
- Appears in all 3 docs
- TF-IDF will be lower (common word)

Word: "powerful"
- Appears in only 1 doc
- TF-IDF will be higher (rare, important word)
```

**Code Concept:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)  # Converts text to TF-IDF scores
```

### Method 3: Word Embeddings (Word2Vec)

**Concept:** Represent each word as a vector of numbers that capture meaning.

**Key Idea:** Words with similar meanings should have similar vectors.

**Example:**
```
king = [0.2, 0.5, 0.8, 0.1, ...]  (300 dimensions)
queen = [0.25, 0.48, 0.78, 0.15, ...] (similar to king!)
man = [0.1, 0.4, 0.7, 0.05, ...]
woman = [0.3, 0.6, 0.9, 0.25, ...]

Vector similarity:
- king is more similar to queen than to man
- This captures semantic meaning!
```

**Famous Word2Vec Property:**
```
king - man + woman ≈ queen
[0.2, 0.5, 0.8, 0.1] - [0.1, 0.4, 0.7, 0.05] + [0.3, 0.6, 0.9, 0.25] ≈ [0.25, 0.48, 0.78, 0.15]
```

**Algorithms:**
- **Skip-gram:** Predicts context words from target word
- **CBOW:** Predicts target word from context words

**Code Concept:**
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=300)
word_vector = model['machine']  # Get vector for "machine"
similarity = model.similarity('machine', 'learning')  # 0-1 scale
```

---

## Comparison: BoW vs TF-IDF vs Word2Vec

| Feature | BoW | TF-IDF | Word2Vec |
|---|---|---|---|
| **What it captures** | Word counts | Word importance | Word meaning |
| **Vector size** | Vocab size (large) | Vocab size (large) | Fixed (usually 300) |
| **Order matters?** | No | No | Yes (through training) |
| **Speed** | Very fast | Fast | Slow (requires training) |
| **Best for** | Basic tasks | Information retrieval | Deep learning, semantic search |
| **Example** | Search filtering | Search ranking | LLM embeddings |

---

## Step 6: Sentiment Analysis (Application)

### What is It?
Determining if text expresses positive, negative, or neutral emotion.

### Approaches

#### 1. Lexicon-Based (Dictionary approach)
```
Positive words: ["good", "great", "excellent", "love"]
Negative words: ["bad", "awful", "terrible", "hate"]

Text: "I love this movie, but the ending was terrible"
Positive score: 1 (for "love")
Negative score: 1 (for "terrible")
Result: Neutral or Mixed sentiment
```

#### 2. Machine Learning-Based
```
Train on labeled data:
Text: "I love AI" → Label: POSITIVE
Text: "I hate bugs" → Label: NEGATIVE
Text: "The weather is neutral" → Label: NEUTRAL

Model learns patterns and predicts on new text
```

**Code Concept:**
```python
from textblob import TextBlob
blob = TextBlob("I love this!")
sentiment = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
```

---

## NLP Pipeline Summary

```
Raw Text
   ↓
[1. Lowercasing & Remove Special Chars]
   ↓
[2. Tokenization]
   ↓
[3. Stopword Removal] (optional)
   ↓
[4. Stemming/Lemmatization]
   ↓
[5. Feature Extraction] (BoW / TF-IDF / Word2Vec)
   ↓
[6. Machine Learning Model]
   ↓
Prediction (Classification / Sentiment / etc)
```

---

## Memory Tricks - NLP Fundamentals

| Concept | Memory Trick |
|---|---|
| **Tokenization** | Breaking text into tokens = "Token-ization" |
| **Stemming** | Chops like a STEM of a plant = crude |
| **Lemmatization** | Uses dictionary = more accurate |
| **TF-IDF** | Rare words get higher score = "Term Frequency × Inverse Doc Frequency" |
| **Word2Vec** | Words with similar meaning → similar vectors |
| **Stopwords** | Common words that stop the signal = remove for cleaner text |
| **BoW** | Count words like ingredients in a bag |

---

## One-Liners - NLP Fundamentals

- **NLP:** Teaching computers to understand human language
- **Tokenization:** Breaking text into words/sentences
- **Stemming:** Crude chopping of word endings
- **Lemmatization:** Dictionary-based root word finding
- **TF-IDF:** Gives high score to rare but important words
- **Word2Vec:** Vectors that capture word meaning/similarity
- **Sentiment Analysis:** Detecting if text is positive/negative/neutral

---

# PHASE 2.2: TRANSFORMERS & ATTENTION MECHANISM
## Video: codebasics - "Transformers Explained"

---

## The Problem: Sequence-to-Sequence Learning

### Traditional RNNs
```
Input:  "I love machine learning"
RNN processes: I → I love → I love machine → I love machine learning
Problem: By the time it reaches "learning", it forgot about "I" (vanishing gradient)
Also: Must process sequentially (slow)
```

### Solution: Attention Mechanism
Instead of forcing the model to remember everything, let it focus on relevant words.

---

## Attention Mechanism

### What is Attention?
**Core Idea:** When translating "I love cats", focus on relevant parts of input.

```
French translation: "J'aime les chats"
Translation step 1: Translate "I" → focus on "I" (100%), "love" (10%), "cats" (5%)
Translation step 2: Translate "love" → focus on "I" (10%), "love" (100%), "cats" (20%)
Translation step 3: Translate "cats" → focus on "I" (5%), "love" (20%), "cats" (100%)
```

### Attention Formula

```
Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V

Where:
Q = Query (what we're looking for)
K = Key (what's available to look at)
V = Value (what information to extract)
d_k = dimension of key (for scaling)
```

**Interpretation:**
1. **Q × K^T:** Compute similarity between query and each key
2. **softmax:** Convert similarities to probabilities (sum to 1)
3. **Multiply by V:** Weight the values by attention scores
4. **Result:** Weighted sum of values (attended information)

---

## Detailed Attention Example

### Setup
```
Sentence: "I love machine learning"
Words: ["I", "love", "machine", "learning"]
Word embeddings (simplified):
- "I":         [1.0, 0.1, 0.2]
- "love":      [0.2, 1.0, 0.1]
- "machine":   [0.1, 0.2, 1.0]
- "learning":  [0.1, 0.3, 0.9]
```

### Step 1: Create Q, K, V
```
From embeddings, create:
Query (Q):  [words transformed to find what to focus on]
Key (K):    [words transformed to be found]
Value (V):  [actual information to extract]

In practice: Q = word_embedding × W_Q (learned weight matrix)
             K = word_embedding × W_K
             V = word_embedding × W_V
```

### Step 2: Compute Attention Scores
```
Scores = Q × K^T

For "I" (row 0):
Similarity with "I":        high (itself)
Similarity with "love":     low
Similarity with "machine":  low
Similarity with "learning": low
```

### Step 3: Apply Softmax
```
Raw scores:   [10.0, 1.0, 0.5, 0.3]
After softmax: [0.99, 0.005, 0.003, 0.002]
(Sum to 1, first word gets most attention)
```

### Step 4: Multiply by Values
```
Attention output = [0.99 * V[I] + 0.005 * V[love] + 0.003 * V[machine] + 0.002 * V[learning]]
                 = Mostly "I"'s information, small parts of others
```

---

## Self-Attention

### What is Self-Attention?
Attention where Query, Key, Value all come from **same sequence**.

```
Input: "I love machine learning"

Self-attention computes:
"I" attends to:       ["I" (high), "love" (low), "machine" (low), "learning" (low)]
"love" attends to:    ["I" (medium), "love" (high), "machine" (low), "learning" (low)]
"machine" attends to: ["I" (low), "love" (medium), "machine" (high), "learning" (high)]
"learning" attends to:["I" (low), "love" (low), "machine" (high), "learning" (high)]

Result: Each word learns which other words to focus on!
```

### Key Insight
Self-attention captures **long-range dependencies** without sequential processing.

```
Old (RNN): "I" → "love" → "machine" → "learning" (must go step by step)
New (Self-Attention): "I" directly attends to "learning" (one step!)
```

---

## Multi-Head Attention

### Problem with Single Attention Head
One attention head captures one type of relationship.

```
Single head might focus on:
- Grammatical relationships
- But miss semantic relationships
```

### Solution: Multiple Heads
Use multiple attention heads in parallel, each learning different patterns.

```
Head 1: Focuses on grammatical relationships
        "I" → [highly attends to "love" (verb follows I)]
        
Head 2: Focuses on semantic relationships
        "I" → [attends to "learning" (same topic)]
        
Head 3: Focuses on other patterns
        ...

Combine: Concatenate all heads and pass through another matrix
         Result: Rich representation capturing multiple relationships
```

### Code Concept
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        self.num_heads = 8
        self.d_k = d_model // num_heads  # 512 / 8 = 64 per head
        
    def forward(self, Q, K, V):
        # Split into 8 heads
        # Compute attention for each head
        # Concatenate results
        # Pass through final linear layer
```

---

## Positional Encoding

### Problem
Self-attention has no notion of word position.

```
Sentence 1: "I love cats"
Sentence 2: "cats love I"

Attention mechanism sees:
[I, love, cats] vs [cats, love, I]
Both have same words → same embeddings → same output!
But meaning is different!
```

### Solution: Positional Encoding
Add position information to embeddings.

```
Word embedding:        [0.1, 0.2, 0.3, 0.4]
Positional encoding:   [0.9, 0.1, 0.8, 0.2]  (depends on position)
Combined:              [1.0, 0.3, 1.1, 0.6]  (word + position info)
```

### Formula
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
pos = position in sequence (0, 1, 2, ...)
i = dimension index
d_model = embedding dimension
```

**Intuition:**
- Lower dimensions: Oscillate slowly (capture long-range positions)
- Higher dimensions: Oscillate quickly (capture local positions)

---

## Transformer Architecture

### Full Transformer Block

```
Input
  ↓
[Multi-Head Self-Attention]
  ↓
[Add & Layer Norm] (residual connection)
  ↓
[Feed Forward Network]  (2 linear layers with activation)
  ↓
[Add & Layer Norm] (residual connection)
  ↓
Output
```

### Why Layer Normalization?
```
Without: Activations can explode or vanish
With: Keeps activations in healthy range (mean 0, std 1)
```

### Why Residual Connections?
```
Without: Deep networks suffer from vanishing gradients
With: Gradients can flow directly through connection
```

---

## Encoder vs Decoder

### Encoder
```
Input: "I love machine learning"
↓
[Self-Attention: each word looks at all words]
↓
[Feed Forward]
↓
Output: Rich representations of each word
```

**Key characteristic:** Can look at future words (entire sentence available)

### Decoder
```
Input: [START] "J" "aime"
↓
[Masked Self-Attention: "J" can only look at [START]
                        "aime" can only look at [START], "J"]
↓
[Cross-Attention: looks at encoder outputs]
↓
Output: Next word probability
```

**Key characteristic:** Cannot look at future words (generating left-to-right)

### Cross-Attention
```
Query (Q):   from decoder
Key (K):     from encoder
Value (V):   from encoder

Allows decoder to focus on relevant parts of input!
```

---

## Putting It Together: Full Transformer

### Translation Example: English → French

```
Input: "I love cats"

ENCODER:
[Self-Attention: each word learns to focus on relevant words]
[Feed Forward: non-linear transformation]
Output: [encoded_I, encoded_love, encoded_cats]

DECODER (generates one word at a time):

Step 1: Generate first word
  Input: [START]
  Masked-SA: [START] attends to [START]
  Cross-A: attends to encoder outputs
  Output: "J" (with probability 0.9)
  
Step 2: Generate second word
  Input: [START], "J"
  Masked-SA: "J" attends to [START], "J"
  Cross-A: attends to encoder outputs
  Output: "aime" (with probability 0.85)
  
Step 3: Generate third word
  Input: [START], "J", "aime"
  Masked-SA: "aime" attends to [START], "J", "aime"
  Cross-A: attends to encoder outputs
  Output: "les" (with probability 0.8)

Step 4: Generate [END] token → stop

Result: "J aime les chats" ✓
```

---

## Comparison: RNN vs Transformer

| Feature | RNN | Transformer |
|---|---|---|
| **Speed** | Sequential (slow) | Parallel (fast) |
| **Long-range deps** | Hard (vanishing gradient) | Easy (attention) |
| **Training** | Slower on GPUs | Highly parallelizable |
| **Inference** | Slow (must generate token-by-token) | Faster |
| **Memory** | Low | Higher (stores attention matrices) |
| **Interpretability** | Black box | Can visualize attention |

---

## Memory Tricks - Transformers

| Concept | Memory Trick |
|---|---|
| **Attention** | "Pay attention to relevant parts" |
| **Self-Attention** | Word looks at itself and neighbors |
| **Multi-Head** | Multiple perspectives on the text |
| **Positional Encoding** | "Where am I in the sequence?" |
| **Encoder** | Reads and understands input |
| **Decoder** | Generates output word-by-word |
| **Cross-Attention** | Decoder looks at encoder |
| **Masked Attention** | Can't peek at future words |

---

## One-Liners - Transformers

- **Attention:** Mechanism to focus on relevant information
- **Self-Attention:** Word learns to focus on other words in sequence
- **Multi-Head Attention:** Multiple perspectives on text (8 heads parallel)
- **Positional Encoding:** Adds position information to word embeddings
- **Transformer:** Encoder-Decoder architecture with attention (no RNN needed)
- **Masked Attention:** Prevents decoder from seeing future words
- **Cross-Attention:** Decoder attends to encoder outputs

---

## Code Concepts for Phase 2.2

When you code along, you'll implement:

1. **Attention Function**
   ```python
   def attention(Q, K, V):
       scores = Q @ K.T / sqrt(d_k)
       weights = softmax(scores)
       return weights @ V
   ```

2. **Multi-Head Attention**
   ```python
   class MultiHeadAttention:
       def split_heads(self, x):
           # Split into num_heads
       def attention(self, Q, K, V):
           # Compute for each head
       def combine_heads(self, x):
           # Concatenate heads
   ```

3. **Positional Encoding**
   ```python
   def positional_encoding(seq_length, d_model):
       # Create PE matrix using sin/cos formulas
       return pe  # shape: (seq_length, d_model)
   ```

4. **Transformer Block**
   ```python
   class TransformerBlock:
       def forward(self, x):
           # Self-attention
           # Layer norm + residual
           # Feed forward
           # Layer norm + residual
           return x
   ```

---

## Interview Questions - Phase 2

### NLP Fundamentals
Q: "Explain the difference between stemming and lemmatization"
A: "Stemming chops word endings using rules (fast but crude). Lemmatization uses dictionary to find root word (accurate but slower). Example: 'running' → stemming: 'runn', lemmatization: 'run'"

Q: "Why do we remove stopwords?"
A: "Stopwords are common words like 'the', 'a' that appear everywhere. Removing them reduces noise and focuses on meaningful words. But context matters - in sentiment analysis, 'not' is important to keep."

Q: "What's the difference between TF-IDF and Word2Vec?"
A: "TF-IDF treats each word independently and gives scores based on frequency. Word2Vec creates dense vectors that capture semantic meaning - similar words have similar vectors. TF-IDF is sparse, Word2Vec is dense."

### Transformers
Q: "Explain self-attention in one sentence"
A: "Self-attention allows each word to learn which other words in the sequence are most relevant to understand it, all in parallel."

Q: "Why do we need positional encoding?"
A: "Without position information, 'I love cats' and 'cats love I' would have the same representation. Positional encoding adds position-dependent information to distinguish word order."

Q: "What's the advantage of multi-head attention?"
A: "Different heads can learn different types of relationships - one might focus on grammar, another on semantics, etc. Combining them captures richer representations."

Q: "How is transformer different from RNN?"
A: "Transformers use self-attention instead of sequential processing. This allows parallel computation (much faster) and better long-range dependency capture (no vanishing gradient)."

---

## Quick Summary Tables

### NLP Pipeline Steps
```
1. Preprocessing  → Cleaning (lowercase, remove special chars)
2. Tokenization   → Breaking into words/sentences
3. Stopword       → Removing common words (optional)
4. Stemming/Lem   → Finding root words
5. Feature Extr   → Converting to numbers (BoW, TF-IDF, Word2Vec)
6. Modeling       → Machine learning/deep learning
```

### Feature Extraction Methods
```
BoW:        Count-based, loses order, simple
TF-IDF:     Importance-weighted, loses order, interpretable
Word2Vec:   Dense vectors, captures semantics, requires training
```

### Transformer Components
```
Self-Attention  → Each word attends to relevant words
Multi-Head      → Multiple attention perspectives
Position Enc    → Adds position information
Feed Forward    → Non-linear transformation
Residual/LN     → Stability and gradient flow
```

---

## Next Steps

After mastering Phase 2:
1. You understand NLP fundamentals (text processing)
2. You understand transformers (modern architecture)
3. Ready for Phase 3: LLMs and RAG systems use transformers!
