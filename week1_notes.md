# Week 1 Notes – Math Refresh (3Blue1Brown)

## Video 1: But what is a neural network?

### What is a neural network?
- A function that maps inputs to outputs
- Learns from examples, not programmed with rules
- Example: Handwriting recognition – looks at pixel values → outputs which digit (0-9)

### Structure of a neural network
| Layer | What it does |
|-------|--------------|
| Input layer | Receives raw data (e.g., pixel values of an image) |
| Hidden layers | Process the information (can be 1 or many) |
| Output layer | Produces final result (e.g., "this is a 3") |

### What is a neuron?
- Each neuron holds a number between 0 and 1
- 0 = not activated | 1 = fully activated
- Example: In handwriting recognition, some neurons detect edges, some detect curves, some detect loops

### Weights and biases (The "learning" part)
| Term | What it means | Analogy |
|------|---------------|---------|
| Weight | How important is this connection? | Volume knob (0 to 1) |
| Bias | Shift the output up or down | Baseline sensitivity |

### Activation function (Sigmoid)
- Takes any number → squashes it between 0 and 1
- Why? Neurons only understand 0-to-1 values

---

## Video 2: Gradient descent – how neural networks learn

### What is "learning"?
- Adjusting weights and biases to make better predictions
- Start with random weights → make a guess → see how wrong → adjust → repeat

### Loss function (Cost function)
- Measures how wrong the network's prediction is
- Low loss = good prediction | High loss = bad prediction
- Goal: Minimize the loss

### Cross-entropy loss (appears in your plan)
- Used when the output is a probability distribution
- Example: [0.1, 0.7, 0.2] for 3 categories
- If correct answer is category 2: cross-entropy = -log(0.7)

### Gradient descent
- Algorithm that finds the smallest loss value
- Think of hiking down a mountain in fog – take small steps downhill
- "Gradient" = direction of steepest ascent (go the opposite way)

### Learning rate
- How big each step is
- Too small → learns very slowly
- Too large → might overshoot and miss the minimum

---

## Math Concepts from Your Plan (Explained Simply)

### Cosine similarity
| Aspect | Explanation |
|--------|-------------|
| What it does | Measures how similar two vectors are |
| Range | -1 (opposite) to 0 (unrelated) to 1 (identical) |
| Formula | cos(θ) = (A·B) / (||A|| × ||B||) |
| In AI | Used to find similar sentences, images, or user preferences |

### Softmax
| Aspect | Explanation |
|--------|-------------|
| What it does | Turns raw scores into probabilities that sum to 1 |
| Example | Scores [2, 1, 0.5] → Softmax → [0.59, 0.24, 0.17] |
| Why needed | Neural networks output raw numbers. Softmax makes them readable as "confidence percentages" |

### Probability
| Aspect | Explanation |
|--------|-------------|
| What it is | Likelihood of an event happening |
| Range | 0 (impossible) to 1 (certain) |
| In LLMs | The model predicts "The next word has 70% chance of being 'cat'" |

### Cross-entropy (continued from above)
| Aspect | Explanation |
|--------|-------------|
| What it measures | How different two probability distributions are |
| Lower value | Predictions are close to reality |
| In training | The loss function tells the model "you were wrong by this much" |

---

## Memory Tricks for Interviews

| Question | Answer to memorize |
|----------|-------------------|
| "What is softmax?" | "Turns scores into probabilities that sum to 1" |
| "What does cosine similarity measure?" | "How similar two vectors are – from -1 to 1" |
| "What is cross-entropy loss?" | "Measures how wrong the prediction is" |
| "What do weights and biases do?" | "Weights control importance. Biases shift the output." |

---

## Quick Summary – One Line Each

- **Neural network:** A function that learns from examples
- **Weights:** Volume knobs for connections between neurons
- **Biases:** Baseline sensitivity of each neuron
- **Softmax:** Converts raw scores to probabilities
- **Cross-entropy:** Measures prediction error
- **Gradient descent:** Algorithm for finding minimum error
- **Cosine similarity:** Measures similarity between vectors
