import time
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from qiskit.circuit.library import ZZFeatureMap, GroverOperator
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# --- 1. CLASSIFICATION COMPARISON ---
def run_classical_svm(n_samples=50, n_features=2):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0, random_state=42)
    clf = SVC(kernel='rbf')
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    return end - start

def run_quantum_svm(n_samples=50, n_features=2):
    # Simulation only
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0, random_state=42)
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2, entanglement='linear')
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    start = time.time()
    # Just evaluating one kernel entry to estimate 'step' time would be too fast, 
    # so we evaluate the whole matrix for the small subset
    _ = kernel.evaluate(x_vec=X)
    end = time.time()
    return end - start

# --- 2. SEARCH COMPARISON ---
def run_classical_search(n_items=1024):
    # Worst case: item is at the end or random check
    # Average case for unstructured search is N/2
    start = time.time()
    target = n_items - 1
    for i in range(n_items):
        if i == target:
            break
    end = time.time()
    return end - start, n_items / 2  # Time, Theoretical Steps

def run_grover_simulation(n_qubits=4):
    # N = 2^n_qubits
    N = 2**n_qubits
    # Theoretical steps = (pi/4) * sqrt(N)
    steps = int((math.pi / 4) * math.sqrt(N))
    
    # We will simulate the circuit construction and run time
    # Note: Simulation time explodes with N, but 'steps' (circuit depth) grows slowly
    
    start = time.time()
    oracle = QuantumCircuit(n_qubits)
    # Mark state '1' * n_qubits (e.g., '1111')
    oracle.h(n_qubits-1)
    oracle.mcx(list(range(n_qubits-1)), n_qubits-1)
    oracle.h(n_qubits-1)
    
    grover_op = GroverOperator(oracle)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(steps):
        qc.append(grover_op, range(n_qubits))
    qc.measure_all()
    
    sampler = StatevectorSampler()
    job = sampler.run([qc])
    _ = job.result()
    end = time.time()
    
    return end - start, steps

def main():
    print("================================================================")
    print("COMPREHENSIVE ALGORITHM COMPARISON: CLASSICAL vs QUANTUM")
    print("================================================================")
    print("Note: 'Time(Sim)' is the wall-clock time on this classical CPU.")
    print("      'Steps(Theory)' is the algorithmic complexity.")
    print("----------------------------------------------------------------")
    
    # 1. Classification
    print("\n[1] CLASSIFICATION (Distinguishing Spam vs Ham)")
    print("Task: Training a model on 50 samples with 4 features (qubits).")
    
    t_class_svm = run_classical_svm(n_samples=50, n_features=4)
    t_quant_svm = run_quantum_svm(n_samples=50, n_features=4)
    
    print(f"{'Algorithm':<20} | {'Time (Sim) [s]':<15} | {'Theory Complexity':<25}")
    print("-" * 65)
    print(f"{'Classical SVM':<20} | {t_class_svm:.6f}{'':<9} | O(N_samples^2 * N_features)")
    print(f"{'Quantum SVM':<20} | {t_quant_svm:.6f}{'':<9} | O(N_samples^2) *Best Case")
    
    # 2. Search
    print("\n[2] SEARCH (Finding a specific Spam Keyword)")
    print("Task: Search in a database of N items.")
    print(f"{'Database Size (N)':<18} | {'Method':<12} | {'Time(Sim)':<10} | {'Steps (Theory)':<15}")
    print("-" * 65)
    
    for n_qubits in [4, 6, 8, 10]:
        N = 2**n_qubits
        
        # Classical
        t_c, steps_c = run_classical_search(N)
        print(f"{N:<18} | {'Linear':<12} | {t_c:.6f}   | {steps_c:<15.0f} (~N/2)")
        
        # Quantum
        try:
            t_q, steps_q = run_grover_simulation(n_qubits)
            print(f"{N:<18} | {'Grover':<12} | {t_q:.6f}   | {steps_q:<15.0f} (~√N)")
        except Exception as e:
            print(f"{N:<18} | {'Grover':<12} | {'Failed':<10} | {'-':<15}")

    print("\n----------------------------------------------------------------")
    print("INTERPRETATION:")
    print("1. Search: Notice how 'Steps (Theory)' for Grover grows much slower than Linear Search.")
    print("   - Linear: 1024 -> 512 steps")
    print("   - Grover: 1024 -> ~25 steps")
    print("   (Simulation time is high because your CPU is working hard to simulate quantum superposition!)")

if __name__ == "__main__":
    main()
