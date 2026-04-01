# Quantum-Based Spam Detection

This project demonstrates a high-performance **Quantum Spam Classifier** using two distinct architectures: **Quantum SVM (QSVC)** for learning-based detection and **Grover's Search** for zero-shot keyword identification.

---

## Final Performance (500-Sample Benchmark)
We benchmarked the system on **500 balanced SMS samples** (350 Training / 150 Testing).

| Algorithm        | Accuracy | Speed (Total) | Complexity / Steps       |
|------------------|----------|---------------|--------------------------|
| **Classical SVM**| 90.0%    | < 0.01 s      | ~105,000 Operations      |
| **Quantum SVM**  | **91.3%**| 266.23 s      | 525,000 Gate Depth       |
| **Grover Search**| 82.7%    | 63.16 s       | 1,554 Gates (Zero Shot)  |

---

## Key Architectural Insights

### 1. Quantum SVM: Higher Dimensional Precision
The **Quantum Support Vector Classifier** achieved the highest accuracy (91.3%). By using a **ZZFeatureMap**, it maps text features into a high-dimensional Hilbert space that classical computers struggle to simulate, allowing for a more nuanced decision boundary between "Ham" and "Spam".

### 2. Grover's Search: The Zero-Shot Advantage
Unlike the SVMs which *require* 350 training labels to work, Grover's Search is **Zero-Shot**. 
- It functions purely through a mathematical **Quantum Oracle** and **QRAM**.
- It identifies spam based on inherent patterns without ever seeing a "training label".
- This represents a fundamental shift in how we think about classification: moving from "Training" to "Quantum Database Querying".

### 3. Dynamic Circuit Architecture
A unique feature of this implementation is that the **Quantum Circuit is not static**. 
-   **Adaptive Qubits**: The number of address qubits is dynamically calculated as $log_2(\text{Email Length})$. A short 4-word SMS uses 2 address qubits; a longer message uses more.
-   **On-the-Fly QRAM**: For every email, the system builds a custom **QRAM loader** that maps the specific sequence of words into a quantum superposition.
-   **Zero-Shot Flexibility**: This allows the classifier to analyze any text without retraining, as the "Intelligence" is encoded in the mathematical relationship between the QRAM and the Oracle.

---

## Optimizations & Innovations

- **BBHT-Optimized Iteration**: Replaced static Grover iterations with the **Boyer-Brassard-Høyer-Tapp** algorithm to handle unknown spam keyword counts across different emails.
- **MCMT Efficiency**: Used Multi-Controlled Multi-Target (MCMT) gates to reduce transpilation overhead, making 500-sample simulations possible on standard hardware.
- **Vocabulary safe-word mapping**: Optimized the QRAM data-bus by mapping all non-suspicious terms to a single `__safe__` ID, drastically reducing qubit requirements.

---

## How to Run

### Install Dependencies
```bash
pip install numpy pandas scikit-learn qiskit qiskit-machine-learning qiskit-algorithms
```

### Run Circuit Visualization
To export a high-level architectural diagram of the Grover circuit (including QRAM, Oracle, and Diffusion blocks):
```bash
python3 src/visualize_grover.py
```
*   Output: `grover_circuit.txt` (ASCII Architecture)

### Run Scale-Testing (Grover Only)
```bash
python3 src/main.py --grover --sample 100
```
