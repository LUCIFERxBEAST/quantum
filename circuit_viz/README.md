# Grover's Search: Quantum Circuit Architecture

This directory contains a high-fidelity visual breakdown of the Quantum Search Algorithm used in the spam classification pipeline.

## 🛠️ Dynamic Architecture Overview
The quantum circuit is **not static**. It is recompiled for every email to ensure optimal qubit usage and sub-linear search complexity.

### 1. Adaptive Qubit Scaling
The circuit automatically calculates the required number of **Address Qubits** based on the email's word count ($log_2(N)$). 
-   A 4-word SMS (like the one in these diagrams) uses **2 Address Qubits**.
-   A 20-word message would use **5 Address Qubits**.
This ensures that the quantum search space is exactly as large as it needs to be, increasing simulation speed and accuracy.

---

## 🧩 Circuit Components

### [qram.png] - The Data Loader
**What it does**: Maps the sequence of words from the email into a quantum superposition.
-   **Data Context**: Generated using the email: `"win free prize now"`.
-   **Mechanism**: Uses Multi-Controlled Multi-Target (MCMT) gates to write word IDs into the Data Register at specific Address locations.

### [oracle.png] - The Phase Inverter
**What it does**: "Marks" the quantum states that represent spam keywords.
-   **Keywords**: `"win"`, `"free"`, `"prize"`, `"urgent"`.
-   **Mechanism**: Performs a Phase Flip (Z-gate logic) whenever the Data Register bits match a suspicious keyword ID.

### [qram_inverse.png] - The Unloader
**What it does**: Reverses the QRAM operation to clean up the Data Register.
-   **Why?**: This isolates the "Spam Signal" onto the Address Register so the Diffusion operator can amplify it.

### [diffusion.png] - The Amplifier
**What it does**: Amplifies the probability of the "Marked" states (the spam keywords) while suppressing the "Safe" states.
-   **Mechanism**: Reflections about the mean amplitude of the address superposition.

---

## 📦 Visualization Data
The diagrams in this folder were generated from the following representative sample:
-   **Email**: `"win free prize now"`
-   **Keywords**: `["win", "free", "prize", "urgent"]`
-   **Total Qubits**: 5 (2 Address + 3 Data)
-   **Circuit Depth**: Optimized via MCMT gate compression.

## 🔬 The Complete Gate & Symbol Glossary
This guide explains every symbol seen in the high-fidelity PNGs and the [exact_architecture.txt](exact_architecture.txt) report.

### 1. The Qubits (q_0 to q_4)
-   **q_0, q_1**: The **Address Register**. These qubits identify which "slot" in the email we are looking at (0, 1, 2, or 3).
-   **q_2, q_3, q_4**: The **Data Register**. These qubits store the unique ID of the word found at that address (e.g., "win" = 1).

### 2. Single-Qubit Gates
-   **`U(π, 0, π)` (X-Gate)**: The Standard NOT gate. It flips $|0\rangle \to |1\rangle$. Used to set up address patterns.
-   **`H` (Hadamard)**: Creates **Superposition**. It turns a definite state into a "maybe," allowing the computer to check every word in the email simultaneously.
-   **`T` & `T†` (T-Dagger)**: These are $\pi/4$ phase rotations. They are the "atoms" of complex logic, used to coordinate the timing of bit-flips in multi-controlled gates.
-   **`U(π/2, 0, π)`**: A specific phase-rotation gate used during the **Diffusion** (amplification) step to prepare the qubits for the final probability boost.

### 3. Multi-Qubit Interaction (The Vertical Wires)
-   **`■` with `X` (CNOT / CX)**: The "Controlled-NOT" gate. If the qubit with the dot is $|1\rangle$, the qubit with the `X` flips. This is how the QRAM "reads" data based on an address.
-   **`■` with `Z` (MCZ)**: The "Controlled-Phase" gate. It marks a state with a negative sign if a specific keyword ID is detected. This is the core of the **Oracle**.

### 4. Logic Control
-   **`Barrier` (░)**: These grey dashed lines are non-physical. They are used to separate the 4 logical phases:
    1.  **QRAM**: Load data.
    2.  **Oracle**: Mark spam.
    3.  **QRAM Inverse**: Clean up data.
    4.  **Diffusion**: Amplify the result.

---

## 🚀 Comparison: Logical vs. Physical
-   **Logical (High-Level)**: You might think of "Load Word 1".
-   **Physical (What you see here)**: To "Load Word 1", the computer must perform a sequence of `H`, `T`, and `CX` gates to manipulate the quantum wave-function. 

The diagrams in this folder show the **Physical** reality of the algorithm.
