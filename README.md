# Quantum Accelerated Spam Mail Checker

## 1. Project Overview
This project demonstrates the power of **Quantum Computing** applied to a real-world cybersecurity problem: **Spam Detection**.

We implement two distinct Quantum Algorithms to solve this problem:
1.  **Quantum Support Vector Classifier (QSVC)**: Uses a **Quantum Kernel** to map data into a high-dimensional Hilbert Space for classification.
2.  **Grover's Search Algorithm**: Uses a **Quantum Oracle** to search for "Spam Keywords" in a superposition of states, providing a quadratic speedup over classical search.

---

## 2. The Pipeline: From Text to Qubits

How do we turn an SMS message like *"Win FREE cash now!"* into something a Quantum Computer can understand?

### Step 1: Preprocessing (`src/preprocess.py`)
*   **Input**: Raw text strings.
*   **Action**: 
    *   Convert to lowercase ("Win" $\to$ "win").
    *   Remove punctuation ("now!" $\to$ "now").
*   **Output**: Cleaned text tokens.

### Step 2: Classical Feature Extraction (`src/features.py`)
*   **TF-IDF (Term Frequency - Inverse Document Frequency)**:
    *   Converts text into numbers based on how "important" a word is.
    *   *Example*: "the" has low importance, "cash" has high importance.
*   **PCA (Principal Component Analysis)**:
    *   **Crucial Step**: Quantum computers today have few qubits (we use 2-4).
    *   TF-IDF creates ~1000 features. PCA compresses this down to **2 features** (for 2 qubits).
*   **Scaling**:
    *   Maps these 2 numbers to the range $[0, 1]$ or $[0, 2\pi]$ so they can be input as **rotation angles** for quantum gates.

---

## 3. Algorithm 1: Quantum Support Vector Classifier (QSVC)
Implemented in: `src/quantum_model.py`

### Theory
*   **The Problem**: Some data is not separable by a straight line in 2D (classical space).
*   **The Quantum Solution**: Map the data to a higher dimension (Hilbert Space) where it *is* linearly separable.
*   **ZZFeatureMap**: We use a specific quantum circuit that entangles qubits.
    *   $|\phi(x)\rangle = U_{\Phi(x)} |0\rangle$
    *   The "distance" between two data points is calculated as the overlap of their quantum states: $|\langle \phi(x) | \phi(y) \rangle|^2$.

### Usage
This method **Learns**. It takes 70% of the data, finds the decision boundary, and predicts the rest.

---

## 4. Algorithm 2: Grover's Search Classifier (Hybrid)
Implemented in: `src/grover_classifier.py`

### Theory
*   **Grover's Algorithm** is famous for finding an item in an unsorted database of size $N$ in $O(\sqrt{N})$ steps (Classical computers take $O(N)$).
*   **Our Application**: We treat "Searching for a Spam Keyword" as the database search problem.

### The "Oracle" (The Brain)
We constructed a **Smart Oracle** that recognizes 30+ keywords, categorized by intent:
*   **Financial**: `cash`, `prize`, `bonus`
*   **Urgency**: `act`, `now`, `immediate`
*   **Security**: `verify`, `password`, `login`

### The Process
1.  Take an email.
2.  Create a **Quantum Superposition** of all words in the email.
3.  Apply the **Oracle**: "Mark" the state if it is a Spam Keyword.
4.  Apply **Grover Diffusion**: Amplify the probability of the marked state.
5.  **Measure**: If we measure a keyword, the email is **SPAM**.

### Results
We ran this on the **Full Dataset (5,572 messages)**.
*   **Accuracy**: ~82%
*   **Spam Recall**: **87%** (It found almost all the spam!)
*   **Speed**: Slow on simulation (hours), but theoretically blindingly fast on real quantum hardware.

---

## 5. How to Run

### Installation
```bash
pip install numpy pandas scikit-learn qiskit qiskit-machine-learning qiskit-algorithms
```

### Running Grover (The Full Simulation)
To run the Grover Search on a sample of the dataset:
```bash
python3 src/main.py --grover --samples 200
```
*   `--samples`: Number of messages to process (Max 5572).

### Running Benchmarks
To see the speed/complexity comparison:
```bash
python3 src/compare_all.py
```

---

## 6. Understanding the Output

```text
              precision    recall  f1-score   support
        Spam       0.44      0.87      0.58       224
```

*   **Precision (0.44)**: "When we claimed it was spam, we were right 44% of the time." (Lots of False Positives - cautious).
*   **Recall (0.87)**: "We found 87% of all the actual spam." (Very few missed spam emails).
*   **F1-Score**: The balance between the two.

## 7. Dataset
**UCI SMS Spam Collection**: A public set of SMS labeled messages that have been collected for mobile phone spam research.

